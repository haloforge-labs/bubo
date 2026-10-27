//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Step status values. NOT RUN is never a pass.
const (
	Pass   = "PASS"
	Fail   = "FAIL"
	NotRun = "NOT RUN"
)

type Check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

type StepResult struct {
	Step       int            `json:"step"`
	Name       string         `json:"name"`
	Status     string         `json:"status"`
	Verdict    string         `json:"verdict"`
	Reason     string         `json:"reason,omitempty"`
	StartedAt  string         `json:"started_at,omitempty"`
	DurationMS int64          `json:"duration_ms"`
	Checks     []Check        `json:"checks,omitempty"`
	Data       map[string]any `json:"data,omitempty"`
}

type PIDRecord struct {
	PID          uint32 `json:"pid"`
	CreationTime string `json:"process_creation_time"`
	Role         string `json:"role"`
	AddedAt      string `json:"added_at"`
	// Filled by the cleanup proof.
	AliveAtEnd *bool  `json:"alive_at_end,omitempty"`
	EndAction  string `json:"end_action,omitempty"`
}

type Report struct {
	Tool       string       `json:"tool"`
	Mode       string       `json:"mode"`
	StartedAt  string       `json:"started_at"`
	FinishedAt string       `json:"finished_at,omitempty"`
	Exe        string       `json:"exe"`
	Manifest   string       `json:"manifest"`
	Port       int          `json:"port"`
	Overall    string       `json:"overall"`
	ExitCode   int          `json:"exit_code"`
	Interrupt  bool         `json:"interrupted,omitempty"`
	Steps      []StepResult `json:"steps"`
	PIDs       []PIDRecord  `json:"pids"`
}

// Recorder owns the transcript, the JSON report and the one-line verdicts.
// Every text line passes through redact, so a planted key can never reach a
// file this tool writes.
type Recorder struct {
	mu         sync.Mutex
	out        string
	console    io.Writer
	transcript *os.File
	report     Report
	secrets    []string
}

func NewRecorder(out string, console io.Writer) (*Recorder, error) {
	if err := os.MkdirAll(out, 0o700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(out, "transcript.txt"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return &Recorder{out: out, console: console, transcript: f, report: Report{Tool: "bubo-smoke", StartedAt: stamp()}}, nil
}

func stamp() string { return time.Now().UTC().Format("2006-01-02T15:04:05.000Z") }

func (r *Recorder) AddSecret(s string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if s != "" {
		r.secrets = append(r.secrets, s)
	}
}
func (r *Recorder) redact(s string) string {
	for _, k := range r.secrets {
		s = strings.ReplaceAll(s, k, "[redacted]")
	}
	return s
}

// Logf writes one transcript line and echoes it to the console.
func (r *Recorder) Logf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	line := r.redact(fmt.Sprintf(format, args...))
	line = strings.ReplaceAll(strings.ReplaceAll(line, "\r", " "), "\n", " | ")
	fmt.Fprintf(r.transcript, "%s %s\n", stamp(), line)
	r.transcript.Sync()
	if r.console != nil {
		fmt.Fprintln(r.console, line)
	}
}

func (r *Recorder) AddStep(s StepResult) {
	r.mu.Lock()
	r.report.Steps = append(r.report.Steps, s)
	r.mu.Unlock()
	r.Flush()
}

// Flush rewrites report.json (temp file, then rename), so a crash leaves the
// last complete report.
func (r *Recorder) Flush() {
	r.mu.Lock()
	defer r.mu.Unlock()
	b, err := json.MarshalIndent(r.report, "", "  ")
	if err != nil {
		return
	}
	b = []byte(r.redact(string(b)))
	b = append(b, '\n')
	tmp := filepath.Join(r.out, "report.json.tmp")
	if os.WriteFile(tmp, b, 0o600) == nil {
		os.Rename(tmp, filepath.Join(r.out, "report.json"))
	}
}

func (r *Recorder) Close() {
	r.Flush()
	r.mu.Lock()
	r.transcript.Close()
	r.mu.Unlock()
}

// Step is the working record of one step. Fail checks decide the status.
type Step struct {
	rec     *Recorder
	res     StepResult
	start   time.Time
	summary string
	done    bool
}

func (r *Recorder) Begin(n int, name string) *Step {
	s := &Step{rec: r, start: time.Now(), res: StepResult{Step: n, Name: name, StartedAt: stamp(), Data: map[string]any{}}}
	r.Logf("[step %d] %s: start", n, name)
	return s
}

// Check records one named condition. It returns ok so callers can stop early.
func (s *Step) Check(name string, ok bool, format string, args ...any) bool {
	detail := fmt.Sprintf(format, args...)
	s.res.Checks = append(s.res.Checks, Check{Name: name, OK: ok, Detail: detail})
	mark := "ok  "
	if !ok {
		mark = "FAIL"
	}
	s.rec.Logf("[step %d]   %s %s: %s", s.res.Step, mark, name, detail)
	return ok
}
func (s *Step) Set(key string, v any)              { s.res.Data[key] = v }
func (s *Step) Summary(format string, args ...any) { s.summary = fmt.Sprintf(format, args...) }

func (s *Step) failed() (string, bool) {
	for _, c := range s.res.Checks {
		if !c.OK {
			return c.Name + ": " + c.Detail, true
		}
	}
	return "", false
}

// End finishes the step: FAIL when any check failed, else PASS.
func (s *Step) End() StepResult {
	if s.done {
		return s.res
	}
	s.done = true
	s.res.DurationMS = time.Since(s.start).Milliseconds()
	if why, bad := s.failed(); bad {
		s.res.Status, s.res.Reason = Fail, why
		s.res.Verdict = fmt.Sprintf("step %d %s: FAIL (%s)", s.res.Step, s.res.Name, why)
	} else {
		s.res.Status = Pass
		s.res.Verdict = fmt.Sprintf("step %d %s: PASS (%s)", s.res.Step, s.res.Name, s.summary)
	}
	s.rec.Logf("%s", s.res.Verdict)
	s.rec.AddStep(s.res)
	return s.res
}

// Abort ends the step as FAIL with one reason (a timeout, a panic).
func (s *Step) Abort(reason string) StepResult {
	if !s.done {
		s.Check("step", false, "%s", reason)
	}
	return s.End()
}

func (r *Recorder) NotRun(n int, name, reason string) StepResult {
	res := StepResult{Step: n, Name: name, Status: NotRun, Reason: reason, Verdict: fmt.Sprintf("step %d %s: NOT RUN: %s", n, name, reason)}
	r.Logf("%s", res.Verdict)
	r.AddStep(res)
	return res
}
