package supervisor

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// drain keeps partial bytes until LF, including split UTF-8. Oversized records
// are reported once and discarded until LF. EOF without LF is explicitly marked.
func drain(r io.Reader, max int, emit func([]byte, bool)) {
	buf := make([]byte, 32<<10)
	line := make([]byte, 0, 4096)
	overflow := false
	for {
		n, err := r.Read(buf)
		for _, b := range buf[:n] {
			if b == '\n' {
				if !overflow {
					emit(bytes.TrimSuffix(line, []byte{'\r'}), true)
				}
				line = line[:0]
				overflow = false
			} else if !overflow {
				if len(line) == max {
					emit(nil, false)
					line = line[:0]
					overflow = true
				} else {
					line = append(line, b)
				}
			}
		}
		if err != nil {
			if len(line) > 0 && !overflow {
				emit(line, false)
			}
			return
		}
	}
}

type Logs struct {
	mu      sync.Mutex
	dir     string
	file    *os.File
	size    int64
	lines   []string
	next    int
	limit   int64
	lastErr error
}

// Keep this tested three-file rotator: the CLI support bundle consumes exactly
// stderr.0.log, stderr.1.log and stderr.2.log (src/cli/support_bundle.cpp).
// Lumberjack's timestamped backup names would silently drop archived GUI logs
// from diagnose --bundle. These are raw child stderr lines, so slog adds no value.
func NewLogs(dir string) (*Logs, error) { return newLogs(dir, 10<<20) }
func newLogs(dir string, limit int64) (*Logs, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	l := &Logs{dir: dir, limit: limit}
	f, err := os.OpenFile(filepath.Join(dir, "stderr.0.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	l.file = f
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	l.size = st.Size()
	return l, nil
}
func (l *Logs) rotate() error {
	if err := l.file.Close(); err != nil {
		return err
	}
	old := filepath.Join(l.dir, "stderr.2.log")
	if err := os.Remove(old); err != nil && !os.IsNotExist(err) {
		return err
	}
	for n := 1; n >= 0; n-- {
		err := os.Rename(filepath.Join(l.dir, fmtLog(n)), filepath.Join(l.dir, fmtLog(n+1)))
		if err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	f, err := os.OpenFile(filepath.Join(l.dir, "stderr.0.log"), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	l.file = f
	l.size = 0
	return nil
}
func fmtLog(n int) string { return []string{"stderr.0.log", "stderr.1.log", "stderr.2.log"}[n] }
func (l *Logs) Append(b []byte, complete bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	s := strings.ToValidUTF8(string(b), "�")
	if b == nil {
		s = "[log record exceeded 64 KiB]"
	}
	if !complete {
		s += " [truncated]"
	}
	if len(l.lines) < 2000 {
		l.lines = append(l.lines, s)
	} else {
		l.lines[l.next] = s
		l.next = (l.next + 1) % 2000
	}
	data := []byte(s + "\n")
	for len(data) > 0 && l.lastErr == nil {
		if l.size >= l.limit {
			l.lastErr = l.rotate()
			if l.lastErr != nil {
				break
			}
		}
		n := int(l.limit - l.size)
		if n > len(data) {
			n = len(data)
		}
		written, err := l.file.Write(data[:n])
		l.size += int64(written)
		l.lastErr = err
		data = data[written:]
		if written == 0 && err == nil {
			l.lastErr = io.ErrShortWrite
		}
	}
}
func (l *Logs) Ring() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.lines) < 2000 {
		return append([]string(nil), l.lines...)
	}
	a := append([]string(nil), l.lines[l.next:]...)
	return append(a, l.lines[:l.next]...)
}
func (l *Logs) Err() error   { l.mu.Lock(); defer l.mu.Unlock(); return l.lastErr }
func (l *Logs) Close() error { l.mu.Lock(); defer l.mu.Unlock(); return l.file.Close() }
