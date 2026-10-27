//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/sys/windows"
)

// ---- read-only process list (M2) ----

// ProcInfo is one running process whose image name matches a forbidden pattern.
type ProcInfo struct {
	PID          uint32 `json:"pid"`
	Name         string `json:"name"`
	CreationTime string `json:"process_creation_time,omitempty"`
	Path         string `json:"path,omitempty"`
}

// matchProc reports whether an image name matches one of the patterns
// (case-insensitive, path.Match wildcards such as llama-*.exe).
func matchProc(name string, patterns []string) bool {
	n := strings.ToLower(name)
	for _, p := range patterns {
		p = strings.ToLower(strings.TrimSpace(p))
		if p == "" {
			continue
		}
		if ok, _ := path.Match(p, n); ok {
			return true
		}
	}
	return false
}

// listProcs reads the process table (Toolhelp snapshot, no handle with write
// rights, nothing is stopped) and returns the matching processes.
func listProcs(patterns []string) ([]ProcInfo, error) {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(snap)
	var e windows.ProcessEntry32
	e.Size = uint32(sizeofProcessEntry32())
	var out []ProcInfo
	for err = windows.Process32First(snap, &e); err == nil; err = windows.Process32Next(snap, &e) {
		name := windows.UTF16ToString(e.ExeFile[:])
		if !matchProc(name, patterns) {
			continue
		}
		ct, _ := creationTimeOf(e.ProcessID)
		image, _ := imagePathOf(e.ProcessID)
		out = append(out, ProcInfo{PID: e.ProcessID, Name: name, CreationTime: ct, Path: image})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out, nil
}

// splitTestFixtures separates matching processes whose executable lives under
// the user's temporary folder. Go tests (t.TempDir, os.MkdirTemp) build their
// fake CLIs there, also named bubo.exe; they hold no GPU memory and a real
// server never runs from that folder. A process whose path cannot be read
// stays blocking.
func splitTestFixtures(list []ProcInfo) (blocking, fixtures []ProcInfo) {
	temp := filepath.Clean(os.TempDir()) + string(filepath.Separator)
	for _, p := range list {
		if p.Path != "" && len(p.Path) > len(temp) && strings.EqualFold(p.Path[:len(temp)], temp) {
			fixtures = append(fixtures, p)
			continue
		}
		blocking = append(blocking, p)
	}
	return blocking, fixtures
}

func sizeofProcessEntry32() uintptr {
	var e windows.ProcessEntry32
	return unsafeSizeof(e)
}

func procText(list []ProcInfo) string {
	var parts []string
	for _, p := range list {
		parts = append(parts, fmt.Sprintf("%s pid %d created %s", p.Name, p.PID, p.CreationTime))
	}
	return strings.Join(parts, "; ")
}

// ---- file hashes and pins (B1) ----

func hashFile(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// binaryHashes hashes every .exe and .dll directly inside dir (not recursive).
func binaryHashes(dir string) (map[string]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, e := range entries {
		n := strings.ToLower(e.Name())
		if e.IsDir() || !(strings.HasSuffix(n, ".exe") || strings.HasSuffix(n, ".dll")) {
			continue
		}
		h, herr := hashFile(filepath.Join(dir, e.Name()))
		if herr != nil {
			return nil, herr
		}
		out[e.Name()] = h
	}
	return out, nil
}

// parsePins reads "name=sha256,name2=sha256".
func parsePins(s string) (map[string]string, error) {
	pins := map[string]string{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, sum, ok := strings.Cut(part, "=")
		sum = strings.ToLower(strings.TrimSpace(sum))
		if !ok || name == "" || len(sum) != 64 {
			return nil, fmt.Errorf("pin %q is not name=<64 hex characters>", part)
		}
		if _, err := hex.DecodeString(sum); err != nil {
			return nil, fmt.Errorf("pin %q: %v", part, err)
		}
		pins[strings.TrimSpace(name)] = sum
	}
	return pins, nil
}

// ---- disk cache must be off (M3) ----

// diskCacheState looks at every effective setting whose key contains "cache".
// It returns the rows as text and the names of rows that are not off.
func diskCacheState(eff map[string]json.RawMessage) (rows map[string]string, on []string) {
	rows = map[string]string{}
	for k, raw := range eff {
		if !strings.Contains(strings.ToLower(k), "cache") {
			continue
		}
		var row settingRow
		if json.Unmarshal(raw, &row) != nil {
			rows[k] = "unreadable"
			on = append(on, k)
			continue
		}
		rows[k] = rowText(raw)
		if !isOffValue(row.Effective) {
			on = append(on, k)
		}
	}
	sort.Strings(on)
	return rows, on
}

func isOffValue(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case bool:
		return !x
	case float64:
		return x == 0
	case string:
		switch strings.ToLower(strings.TrimSpace(x)) {
		case "", "off", "none", "disabled", "false", "0", "no":
			return true
		}
	}
	return false
}

// ---- stop refused before Manage (m2) ----

const manageRefusalText = "Select Manage this server before Stop"

// manageRefusal accepts only the supervisor's typed refusal, returned quickly.
// A timeout, a context cancel or any other error is not a refusal.
func manageRefusal(err error, took time.Duration) (bool, string) {
	switch {
	case err == nil:
		return false, "Stop was accepted for a server that was not managed"
	case err == errWaitTimeout:
		return false, "Stop did not answer in time (a hang is not a refusal)"
	case !strings.Contains(err.Error(), manageRefusalText):
		return false, fmt.Sprintf("Stop returned %q, not the Manage refusal", err.Error())
	case took > 5*time.Second:
		return false, fmt.Sprintf("the refusal took %s (limit 5 s)", took.Round(time.Millisecond))
	}
	return true, fmt.Sprintf("refused in %s: %v", took.Round(time.Millisecond), err)
}

func sortedKeys(m map[string]string) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// lockSummary reads a lock file for the record only (the tool never changes it).
func lockSummary(p string) map[string]any {
	lines, exists, err := lockOwnerLines(p)
	m := map[string]any{"path": p, "exists": exists}
	if err != nil {
		m["error"] = err.Error()
	}
	if len(lines) > 0 {
		m["lines"] = lines
	}
	return m
}
