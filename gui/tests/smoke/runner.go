//go:build windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"bubo/gui/supervisor"
)

var errWaitTimeout = errors.New("timed out")

// memCredentials keeps the supervisor away from Windows Credential Manager.
// The smoke never turns on local network access, so no key is ever stored.
type memCredentials struct {
	mu sync.Mutex
	m  map[string]string
}

func (c *memCredentials) Read(t string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.m[t]; ok {
		return v, nil
	}
	return "", errors.New("not found")
}
func (c *memCredentials) Write(t, s string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[string]string{}
	}
	c.m[t] = s
	return nil
}
func (c *memCredentials) Delete(t string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.m, t)
	return nil
}

type Runner struct {
	cfg      Config
	rec      *Recorder
	led      *Ledger
	scrubbed []string
	ctx      context.Context
	client   *http.Client

	sup     *supervisor.Supervisor
	stopObs func()
	supN    int
	views   viewCapture

	// Facts handed from one step to the next.
	endpoint         string
	model            string
	instanceID       string
	base             *supervisor.Status
	baseRing         int
	chats            []chatResult
	planted          string   // fake key planted in the step 7 server and used by step 9
	forced           []string // PIDs that had to be stopped by exact PID outside a normal stop
	shutdownFailures []string
}

type stepFn func(ctx context.Context, s *Step)

// exec runs one step with a deadline and a panic guard.
func (r *Runner) exec(n int, name string, timeout time.Duration, fn stepFn) StepResult {
	if r.ctx.Err() != nil {
		return r.rec.NotRun(n, name, "interrupted before this step")
	}
	s := r.rec.Begin(n, name)
	sctx, cancel := context.WithTimeout(r.ctx, timeout)
	defer cancel()
	func() {
		defer func() {
			if p := recover(); p != nil {
				s.Check("panic", false, "%v", p)
			}
		}()
		fn(sctx, s)
	}()
	if _, bad := s.failed(); !bad {
		if r.ctx.Err() != nil {
			s.Check("interrupted", false, "Ctrl+C during the step")
		} else if sctx.Err() != nil {
			s.Check("step deadline", false, "step ran longer than %s", timeout)
		}
	}
	return s.End()
}

func (r *Runner) runAll(ctx context.Context) int {
	r.ctx = ctx
	r.client = &http.Client{}
	cfg := r.cfg
	defer r.finish()

	switch cfg.mode() {
	case ModePreflightOnly:
		return r.partial(1, "preflight", 3*time.Minute, r.preflight, "PREFLIGHT")
	case ModeOnlyStep9:
		return r.partial(9, "log export", 4*time.Minute, r.stepLogExport, "STEP9")
	}
	pre := r.exec(1, "preflight", 3*time.Minute, r.preflight)
	var ready bool
	if pre.Status != Pass {
		for _, st := range []struct {
			n    int
			name string
		}{{2, "start"}, {3, "request"}, {4, "status counters"}, {5, "stall watchdog"}, {6, "stop and restart"}, {7, "reconnect"}} {
			r.rec.NotRun(st.n, st.name, "preflight did not pass")
		}
		r.rec.NotRun(9, "log export", "preflight did not pass")
		return r.cleanupAndExit()
	}

	s2 := r.exec(2, "start", cfg.LoadTimeout+4*time.Minute, r.stepStart)
	ready = s2.Status == Pass
	if !ready {
		r.settle("step 2 failed")
		for _, st := range []struct {
			n    int
			name string
		}{{3, "request"}, {4, "status counters"}, {5, "stall watchdog"}, {6, "stop and restart"}, {7, "reconnect"}} {
			r.rec.NotRun(st.n, st.name, "step 2 did not reach a verified Ready")
		}
	} else {
		r.exec(3, "request", 2*cfg.RequestTimeout+3*time.Minute, r.stepRequest)
		// Step 4 reads counters, slots and the Pi snippet. It does not depend on
		// cache reuse, so it runs whenever both requests returned HTTP 200. A cache
		// miss stays a step 3 FAIL and shows again in step 4; it never hides it.
		if len(r.chats) == 2 {
			r.exec(4, "status counters", 3*time.Minute, r.stepCounters)
		} else {
			r.rec.NotRun(4, "status counters", "step 3 did not complete both requests with HTTP 200")
		}
		r.exec(5, "stall watchdog", cfg.IdleWatch+2*time.Minute, r.stepIdle)
		s6 := r.exec(6, "stop and restart", 2*(cfg.LoadTimeout+cfg.StopTimeout)+cfg.RequestTimeout+8*time.Minute, r.stepStopRestart)
		if s6.Status != Pass {
			r.settle("step 6 failed")
		} else {
			r.closeSup("after step 6")
		}
		if r.stillAlive() {
			r.rec.NotRun(7, "reconnect", "a recorded process is still alive after step 6")
		} else {
			s7 := r.exec(7, "reconnect", cfg.LoadTimeout+3*cfg.StopTimeout+5*time.Minute, r.stepReconnect)
			if s7.Status != Pass {
				r.settle("step 7 failed")
			}
		}
	}
	r.exec(9, "log export", 4*time.Minute, r.stepLogExport)
	return r.cleanupAndExit()
}

// partial runs one step alone (--preflight-only, --only-step9) and finishes the
// report. Its verdict is never "PASS": only a full run proves the smoke.
func (r *Runner) partial(n int, name string, timeout time.Duration, fn stepFn, label string) int {
	res := r.exec(n, name, timeout, fn)
	r.closeSup("final")
	code, overall := ExitPass, label+" PASS"
	switch {
	case r.ctx.Err() != nil:
		code, overall = ExitInterrupted, "INTERRUPTED"
		r.rec.report.Interrupt = true
	case res.Status != Pass:
		code, overall = ExitFail, label+" FAIL"
	}
	if alive := r.led.Alive(); len(alive) > 0 { // nothing should have started
		for _, p := range alive {
			action, err := terminateExact(p.PID, p.CreationTime)
			r.rec.Logf("partial: pid %d: %s %v", p.PID, action, err)
		}
		code, overall = ExitFail, label+" FAIL"
	}
	r.rec.report.Overall, r.rec.report.ExitCode, r.rec.report.FinishedAt = overall, code, stamp()
	r.rec.Logf("OVERALL: %s (%s only); exit %d; report %s", overall, r.cfg.mode(), code, filepath.Join(r.cfg.Out, "report.json"))
	return code
}

func (r *Runner) stillAlive() bool { return len(r.led.Alive()) > 0 }

// settle ends everything this run started: supervisor Exit first, then exact
// PIDs for what is left. It is used after a failure, before the next step.
func (r *Runner) settle(why string) {
	r.rec.Logf("settle: %s", why)
	r.closeSup(why)
	for _, p := range r.led.Alive() {
		action, err := terminateExact(p.PID, p.CreationTime)
		r.rec.Logf("settle: pid %d: %s %v", p.PID, action, err)
		r.forced = append(r.forced, fmt.Sprintf("%d (%s)", p.PID, p.Role))
	}
}

// cleanupAndExit is step 8 and the overall verdict.
func (r *Runner) cleanupAndExit() int {
	r.closeSup("final")
	r.stepCleanup()
	steps := r.rec.report.Steps
	sort.SliceStable(steps, func(i, j int) bool { return steps[i].Step < steps[j].Step })
	fails, notRun, passes := 0, 0, 0
	for _, s := range steps {
		switch s.Status {
		case Fail:
			fails++
		case NotRun:
			notRun++
		case Pass:
			passes++
		}
	}
	code, overall := ExitPass, "PASS"
	switch {
	case r.ctx.Err() != nil:
		code, overall = ExitInterrupted, "INTERRUPTED"
		r.rec.report.Interrupt = true
	case fails > 0:
		code, overall = ExitFail, "FAIL"
	case notRun > 0 || passes != 9:
		code, overall = ExitNotRun, "NOT RUN"
	}
	r.rec.report.Overall, r.rec.report.ExitCode, r.rec.report.FinishedAt = overall, code, stamp()
	r.rec.Logf("OVERALL: %s (%d pass, %d fail, %d not run of 9 steps); exit %d; report %s", overall, passes, fails, notRun, code, filepath.Join(r.cfg.Out, "report.json"))
	return code
}

func (r *Runner) finish() {}

// ---- supervisors ----

func (r *Runner) newSup(tag string, observe bool) (*supervisor.Supervisor, error) {
	r.supN++
	c := supervisor.Config{
		Exe: r.cfg.Exe, Port: r.cfg.Port,
		LogDir:       filepath.Join(r.cfg.Out, fmt.Sprintf("supervisor-logs-%d-%s", r.supN, tag)),
		SettingsPath: filepath.Join(r.cfg.Out, "gui-settings.json"), // never the owner's real settings
		LoadTimeout:  r.cfg.LoadTimeout, StopTimeout: r.cfg.StopTimeout, DrainTimeout: 2 * time.Second,
		QueryTimeout: r.cfg.QueryTimeout, PollInterval: r.cfg.PollInterval,
		InitialPrepared: true,
		Keys:            supervisor.NewKeyManagerFor(&memCredentials{}, "bubo-smoke-unused"),
	}
	if r.cfg.MVP {
		c.InitialPrepared = false // bootstrap must prove readiness, never assume it
		if !r.cfg.Fake {
			c.GPU = supervisor.NewGPUSource()
		}
	}
	sup, err := supervisor.New(c)
	if err != nil {
		return nil, err
	}
	r.sup = sup
	octx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	r.stopObs = func() { cancel(); <-done }
	go func() {
		defer close(done)
		if !observe {
			<-octx.Done()
			return
		}
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-octx.Done():
				return
			case <-t.C:
				v := r.view(sup)
				// Only a process this supervisor launched is owned: Starting,
				// Ready and Stopping carry its own identity.
				if (v.State == supervisor.Starting || v.State == supervisor.Ready || v.State == supervisor.Stopping) && v.Instance != nil {
					r.led.Add(v.Instance.PID, v.Instance.CreationTime, "supervisor-owned bubo serve")
				}
			}
		}
	}()
	return sup, nil
}

// closeSup exits the current supervisor and waits for it. A supervisor that
// owns a server stops it gracefully first.
func (r *Runner) closeSup(why string) {
	if r.sup == nil {
		return
	}
	select {
	case rep := <-r.sup.Exit():
		r.rec.Logf("supervisor exit (%s): err=%v", why, rep.Err)
		if rep.Err != nil {
			r.shutdownFailures = append(r.shutdownFailures, fmt.Sprintf("%s: %v", why, rep.Err))
		}
	case <-time.After(r.cfg.StopTimeout + 30*time.Second):
		r.rec.Logf("supervisor exit (%s): no reply within the deadline", why)
		r.shutdownFailures = append(r.shutdownFailures, why+": supervisor exit timed out")
	}
	r.stopObs()
	r.sup, r.stopObs = nil, nil
}

func (r *Runner) waitView(ctx context.Context, sup *supervisor.Supervisor, timeout time.Duration, cond func(supervisor.View) (bool, error)) (supervisor.View, error) {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	last := ""
	for {
		v := r.view(sup)
		key := fmt.Sprintf("%s|%s|%t", v.State, v.Reason, v.Stale)
		if key != last {
			last = key
			r.rec.Logf("supervisor state: %s label=%q stale=%t", v.State, v.Reason, v.Stale)
		}
		done, err := cond(v)
		if err != nil {
			return v, err
		}
		if done {
			return v, nil
		}
		select {
		case <-ctx.Done():
			return v, ctx.Err()
		case <-deadline.C:
			return v, errWaitTimeout
		case <-tick.C:
		}
	}
}

func lastErr(v supervisor.View) string {
	if v.Failure == nil {
		return "none"
	}
	return v.Failure.Code + " from " + v.Failure.Source
}

// reply waits for an action acknowledgement.
func reply(ctx context.Context, ch <-chan supervisor.ResultReply, timeout time.Duration) error {
	select {
	case rep := <-ch:
		return rep.Err
	case <-time.After(timeout):
		return errWaitTimeout
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ---- direct CLI calls ----

type capWriter struct {
	buf bytes.Buffer
	max int
}

func (w *capWriter) Write(p []byte) (int, error) {
	if w.buf.Len()+len(p) > w.max {
		return 0, errors.New("output is larger than the cap")
	}
	return w.buf.Write(p)
}

// cli runs one bounded read-only bubo command and parses its first record.
func (r *Runner) cli(ctx context.Context, timeout time.Duration, args ...string) (supervisor.Event, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := hiddenCommand(cctx, r.cfg.Exe, args...)
	out := &capWriter{max: 4 << 20}
	cmd.Stdout = out
	err := cmd.Run()
	line := bytes.TrimSpace(out.buf.Bytes())
	if i := bytes.IndexByte(line, '\n'); i >= 0 {
		line = bytes.TrimSpace(line[:i])
	}
	if cctx.Err() != nil {
		return supervisor.Event{}, fmt.Errorf("%s did not finish within %s", args[0], timeout)
	}
	if len(line) == 0 {
		return supervisor.Event{}, fmt.Errorf("%s printed no record (run error: %v)", args[0], err)
	}
	ev, perr := supervisor.ParseEvent(line)
	if perr != nil {
		return ev, fmt.Errorf("%s record is invalid: %v", args[0], perr)
	}
	if ev.Type == "error" {
		if ce, ok := ev.Value.(*supervisor.CLIError); ok {
			return ev, fmt.Errorf("CLI error %s (%s)", ce.Code, ce.Stage)
		}
		return ev, errors.New("CLI error")
	}
	if err != nil {
		return ev, fmt.Errorf("%s exited with an error: %v", args[0], err)
	}
	return ev, nil
}

// statusOf samples `bubo status --json` (for one instance when id is given).
func (r *Runner) statusOf(ctx context.Context, id string) (*supervisor.Status, map[string]json.RawMessage, error) {
	args := []string{"status", "--json"}
	if id != "" {
		args = append(args, "--instance-id", id)
	}
	ev, err := r.cli(ctx, r.cfg.QueryTimeout, args...)
	if err != nil {
		return nil, nil, err
	}
	st, ok := ev.Value.(*supervisor.Status)
	if !ok {
		return nil, nil, fmt.Errorf("record type %q is not a status", ev.Type)
	}
	var payload map[string]json.RawMessage
	if err = json.Unmarshal(ev.Payload, &payload); err != nil {
		return nil, nil, err
	}
	if id != "" {
		r.saveRawStatus(id, ev.Payload)
	}
	return st, payload, nil
}

// saveRawStatus appends the raw `status --json --instance-id` payload to a file
// under --out, so the step 9 walk over every output file also scans what the
// CLI really sent (N16: the report and transcript are redacted, this is not).
// A write failure is recorded in the transcript, never ignored.
func (r *Runner) saveRawStatus(id string, payload []byte) {
	f, err := os.OpenFile(filepath.Join(r.cfg.Out, "status-instance-raw.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err == nil {
		var one bytes.Buffer
		if json.Compact(&one, payload) == nil {
			one.WriteByte('\n')
			_, err = f.Write(one.Bytes())
		}
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}
	if err != nil {
		r.rec.Logf("WARNING: raw status of %s was not saved: %v", id, err)
	}
}

// ---- small helpers ----

var versionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+`)

func normPath(p string) string {
	return strings.ToLower(strings.ReplaceAll(filepath.Clean(p), "/", `\`))
}

func gib(b uint64) float64 { return float64(b) / (1 << 30) }

// dig walks nested JSON maps.
func dig(v any, path ...string) any {
	for _, k := range path {
		m, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		v = m[k]
	}
	return v
}

func decodeAny(raw json.RawMessage) any {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return nil
	}
	return v
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}
