//go:build windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// J12b additions: preflight-only, only-step9, process list, locks, pins,
// nil counters, unknown power, typed Manage refusal, planted key in the
// external server's environment.

func onlyStep(t *testing.T, rep Report, n int) {
	t.Helper()
	if len(rep.Steps) != 1 || rep.Steps[0].Step != n {
		t.Errorf("want only step %d, got %+v", n, rep.Steps)
	}
	if len(rep.PIDs) != 0 {
		t.Errorf("nothing may start: %+v", rep.PIDs)
	}
}

func dataOf(rep Report, n int) map[string]any {
	for _, s := range rep.Steps {
		if s.Step == n {
			return s.Data
		}
	}
	return nil
}

func TestPreflightOnly(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	c.PreflightOnly = true
	rep, _ := runExpect(t, c, ExitPass, [10]string{"", Pass})
	if rep.Overall != "PREFLIGHT PASS" || rep.Mode != ModePreflightOnly {
		t.Errorf("overall %q mode %q: a preflight must never read as the full PASS", rep.Overall, rep.Mode)
	}
	onlyStep(t, rep, 1)
	d := dataOf(rep, 1)
	for _, k := range []string{"host_ram_available_gib", "host_ram_total_gib", "binary_sha256", "lock_state", "power_source", "capabilities", "versions"} {
		if _, ok := d[k]; !ok {
			t.Errorf("preflight data lacks %s", k)
		}
	}
	// A pin that matches passes; a wrong pin fails.
	sum, err := hashFile(c.Exe)
	if err != nil {
		t.Fatal(err)
	}
	c2 := fakeConfig(t, "normal", "ok", nil)
	c2.PreflightOnly, c2.Pins = true, filepath.Base(c2.Exe)+"="+strings.ToUpper(sum)
	runExpect(t, c2, ExitPass, [10]string{"", Pass})
	c3 := fakeConfig(t, "normal", "ok", nil)
	c3.PreflightOnly, c3.Pins = true, filepath.Base(c3.Exe)+"="+strings.Repeat("0", 64)
	rep3, _ := runExpect(t, c3, ExitFail, [10]string{"", Fail})
	if rep3.Overall != "PREFLIGHT FAIL" || !strings.Contains(rep3.Steps[0].Reason, "pin") {
		t.Errorf("wrong pin: %q %q", rep3.Overall, rep3.Steps[0].Reason)
	}
	// A lock does not refuse a read-only preflight (it is only recorded).
	c4 := fakeConfig(t, "normal", "ok", nil)
	c4.PreflightOnly = true
	os.WriteFile(c4.LockPath, []byte("owner=somebody-else\n"), 0o600)
	rep4, _ := runExpect(t, c4, ExitPass, [10]string{"", Pass})
	if ls, _ := dataOf(rep4, 1)["lock_state"].(map[string]any); ls == nil {
		t.Error("lock state not recorded")
	}
}

func TestProcessListCheck(t *testing.T) {
	// A bystander named like a production process: image name smoketest-llama-7.exe.
	dir := t.TempDir()
	t.Setenv("BUBO_FAKE_DIR", dir) // the bystander must never append in the working directory
	// The check ignores fakes under the temp folder (splitTestFixtures); move
	// the temp folder elsewhere so this bystander counts like a real server.
	otherTemp := filepath.Join(dir, "other-temp")
	if err := os.MkdirAll(otherTemp, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TMP", otherTemp)
	t.Setenv("TEMP", otherTemp)
	byName := filepath.Join(dir, "smoketest-llama-7.exe")
	b, err := os.ReadFile(fakeExe)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(byName, b, 0o700); err != nil {
		t.Fatal(err)
	}
	by := exec.Command(byName, "grandchild")
	by.Env = os.Environ()
	if err = by.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { by.Process.Kill(); by.Wait() }()
	pats := []string{"bubo-never.exe", "smoketest-llama-*.exe"}

	c := fakeConfig(t, "normal", "ok", nil)
	c.ProcPatterns, c.PreflightOnly = pats, true
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Fail})
	if !strings.Contains(rep.Steps[0].Reason, "no bubo/gufo/llama process") {
		t.Errorf("reason %q", rep.Steps[0].Reason)
	}
	// With production up the same finding is recorded and does not fail.
	c2 := fakeConfig(t, "normal", "ok", nil)
	c2.ProcPatterns = pats
	c2.PreflightOnly, c2.ProductionUp = true, true
	rep2, _ := runExpect(t, c2, ExitPass, [10]string{"", Pass})
	list, _ := dataOf(rep2, 1)["matching_processes"].([]any)
	if len(list) != 1 {
		t.Errorf("recorded processes %v", list)
	}
	// A full run with the process present fails at step 1 and starts nothing.
	c3 := fakeConfig(t, "normal", "ok", nil)
	c3.ProcPatterns = pats
	rep3, _ := runExpect(t, c3, ExitFail, [10]string{"", Fail, NotRun})
	if len(rep3.PIDs) != 0 {
		t.Error("a server started with a matching process present")
	}
	// The bystander was not touched.
	if ct, err := creationTimeOf(uint32(by.Process.Pid)); err != nil || !isLive(uint32(by.Process.Pid), ct) {
		t.Error("the bystander was touched")
	}
	// --production-up is not allowed without --preflight-only.
	c4 := fakeConfig(t, "normal", "ok", nil)
	c4.ProductionUp = true
	var buf bytes.Buffer
	if got := run(context.Background(), c4, &buf); got != ExitUsage {
		t.Errorf("production-up without preflight-only exit %d", got)
	}
}

func TestMatchProc(t *testing.T) {
	pats := []string{"bubo.exe", "gufo.exe", "llama-*.exe"}
	for name, want := range map[string]bool{"bubo.exe": true, "BUBO.EXE": true, "gufo.exe": true, "llama-server.exe": true, "llama-bench.exe": true,
		"bubo-smoke.exe": false, "notbubo.exe": false, "llama.exe": false, "chrome.exe": false} {
		if got := matchProc(name, pats); got != want {
			t.Errorf("matchProc(%q) = %t, want %t", name, got, want)
		}
	}
}

func TestOnlyStep9(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	c.OnlyStep9 = true
	rep, _ := runExpect(t, c, ExitPass, [10]string{9: Pass})
	if rep.Overall != "STEP9 PASS" || rep.Mode != ModeOnlyStep9 {
		t.Errorf("overall %q mode %q", rep.Overall, rep.Mode)
	}
	onlyStep(t, rep, 9)
	// A leaking bundle fails it.
	c2 := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_BUNDLE_LEAK": "1"})
	c2.OnlyStep9 = true
	rep2, _ := runExpect(t, c2, ExitFail, [10]string{9: Fail})
	if rep2.Overall != "STEP9 FAIL" {
		t.Errorf("overall %q", rep2.Overall)
	}
	// It needs no free port and no lock.
	t.Setenv("BUBO_FAKE_BUNDLE_LEAK", "") // c2 set it for this test
	c3 := fakeConfig(t, "normal", "ok", nil)
	c3.OnlyStep9 = true
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", c3.Port))
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	os.WriteFile(c3.LockPath, []byte("owner=somebody-else\n"), 0o600)
	runExpect(t, c3, ExitPass, [10]string{9: Pass})
	// A production port is still refused.
	c4 := fakeConfig(t, "normal", "ok", nil)
	c4.OnlyStep9, c4.Port = true, 8083
	runExpect(t, c4, ExitRefused, [10]string{})
	// The two modes cannot be combined.
	c5 := fakeConfig(t, "normal", "ok", nil)
	c5.OnlyStep9, c5.PreflightOnly = true, true
	var buf bytes.Buffer
	if got := run(context.Background(), c5, &buf); got != ExitUsage {
		t.Errorf("combined modes exit %d", got)
	}
}

// M6: a server that echoes its environment leaks the planted key into its
// stderr file; the step 9 walk must find it.
func TestServerEnvLeakFailsStep9(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_ENV_LEAK": "1"})
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Pass, Pass, Pass, Pass, Pass, Pass, Fail})
	for _, s := range rep.Steps {
		if s.Step == 9 && !strings.Contains(s.Reason, "planted key absent from run output") {
			t.Errorf("step 9 reason %q", s.Reason)
		}
	}
	b, _ := os.ReadFile(filepath.Join(c.Out, "report.json"))
	if bytes.Contains(b, []byte("SMOKE-PLANTED")) {
		t.Error("the planted key is in report.json")
	}
	raw, _ := os.ReadFile(filepath.Join(c.Out, "external-serve.stderr.log"))
	if !bytes.Contains(raw, []byte("sk-SMOKE-PLANTED-")) {
		t.Error("the planted key never reached the external server environment")
	}
}

func TestUnknownPowerFails(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_POWER": "unknown"})
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Fail, NotRun})
	if !strings.Contains(rep.Steps[0].Reason, "AC power") {
		t.Errorf("reason %q", rep.Steps[0].Reason)
	}
	c2 := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_POWER": "unknown"})
	c2.AllowBattery = true
	runExpect(t, c2, ExitPass, allPass)
}

// m1: a counter the CLI does not report fails visibly in steps 4 and 5.
func TestNilCounterFailsVisibly(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_NULL_COUNTERS": "completed_successful_requests_total"})
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Pass, Fail, Pass, Pass, Pass, Pass, Pass})
	for _, s := range rep.Steps {
		if s.Step == 4 && !strings.Contains(s.Reason, "completed counter present") {
			t.Errorf("step 4 reason %q", s.Reason)
		}
	}
	c2 := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_NULL_COUNTERS": "decode_tokens_total"})
	rep2, _ := runExpect(t, c2, ExitFail, [10]string{"", Pass, Pass, Pass, Fail, Fail, Pass, Pass, Pass, Pass})
	for _, s := range rep2.Steps {
		if s.Step == 5 && !strings.Contains(s.Reason, "idle counters read") {
			t.Errorf("step 5 reason %q", s.Reason)
		}
	}
}

func TestManageRefusalIsTyped(t *testing.T) {
	good := fmt.Errorf("Select Manage this server before Stop")
	cases := []struct {
		name string
		err  error
		took time.Duration
		want bool
	}{
		{"typed and fast", good, 40 * time.Millisecond, true},
		{"accepted", nil, time.Millisecond, false},
		{"timeout", errWaitTimeout, 30 * time.Second, false},
		{"other error", fmt.Errorf("supervisor is busy"), time.Millisecond, false},
		{"context cancel", context.Canceled, time.Millisecond, false},
		{"typed but slow", good, 6 * time.Second, false},
	}
	for _, tc := range cases {
		if got, why := manageRefusal(tc.err, tc.took); got != tc.want {
			t.Errorf("%s: got %t (%s), want %t", tc.name, got, why, tc.want)
		}
	}
}

func TestDiskCacheState(t *testing.T) {
	mk := func(eff any) json.RawMessage {
		b, _ := json.Marshal(map[string]any{"requested": nil, "effective": eff, "source": "default", "reason": nil})
		return b
	}
	rows, on := diskCacheState(map[string]json.RawMessage{"profile": mk("default"), "disk_cache": mk("off"), "--cache-disk": mk(nil)})
	if len(on) != 0 || len(rows) != 2 {
		t.Errorf("off rows: %v %v", rows, on)
	}
	_, on = diskCacheState(map[string]json.RawMessage{"cache_disk": mk(`C:\x\cache`), "disk_cache_bytes": mk(0)})
	if len(on) != 1 || on[0] != "cache_disk" {
		t.Errorf("on rows: %v", on)
	}
	rows, on = diskCacheState(map[string]json.RawMessage{"profile": mk("default")})
	if len(rows) != 0 || len(on) != 0 {
		t.Errorf("no cache row: %v %v", rows, on)
	}
}

func TestParsePins(t *testing.T) {
	good := strings.Repeat("ab", 32)
	if p, err := parsePins("bubo.exe=" + good + ", bubo-engine.dll=" + strings.ToUpper(good)); err != nil || len(p) != 2 || p["bubo-engine.dll"] != good {
		t.Errorf("pins %v %v", p, err)
	}
	for _, bad := range []string{"bubo.exe", "bubo.exe=abc", "=" + good, "x=" + strings.Repeat("zz", 32)} {
		if _, err := parsePins(bad); err == nil {
			t.Errorf("pin %q accepted", bad)
		}
	}
}

// M4: both locks, whole-line owner match, and --require-locks.
func TestLocksGpuAndBuild(t *testing.T) {
	const owner = "owner=user-gui-smoke"
	c := fakeConfig(t, "normal", "ok", nil)
	c.AllowLockOwner, c.RequireLocks = owner, true
	_, console := runExpect(t, c, ExitRefused, [10]string{})
	if !strings.Contains(console, "does not exist") {
		t.Error(console)
	}
	// Only the GPU lock: the build lock is still missing.
	c2 := fakeConfig(t, "normal", "ok", nil)
	c2.AllowLockOwner, c2.RequireLocks = owner, true
	os.WriteFile(c2.LockPath, []byte(owner+"\n"), 0o600)
	_, console = runExpect(t, c2, ExitRefused, [10]string{})
	if !strings.Contains(console, "build lock") {
		t.Error(console)
	}
	// A build lock of another owner is refused.
	c3 := fakeConfig(t, "normal", "ok", nil)
	c3.AllowLockOwner = owner
	os.WriteFile(c3.LockPath, []byte(owner+"\n"), 0o600)
	os.WriteFile(c3.BuildLockPath, []byte("owner=cpu-build\n"), 0o600)
	_, console = runExpect(t, c3, ExitRefused, [10]string{})
	if !strings.Contains(console, "owner=cpu-build") {
		t.Error(console)
	}
	// A trailing field on the owner line breaks the whole-line match.
	c4 := fakeConfig(t, "normal", "ok", nil)
	c4.AllowLockOwner = owner
	os.WriteFile(c4.LockPath, []byte(owner+" started=now\n"), 0o600)
	runExpect(t, c4, ExitRefused, [10]string{})
	// Both locks with the exact line: the run passes and neither file changes.
	c5 := fakeConfig(t, "normal", "ok", nil)
	c5.AllowLockOwner, c5.RequireLocks = owner, true
	for _, p := range []string{c5.LockPath, c5.BuildLockPath} {
		os.WriteFile(p, []byte(owner+"\r\n"), 0o600)
	}
	runExpect(t, c5, ExitPass, allPass)
	for _, p := range []string{c5.LockPath, c5.BuildLockPath} {
		if b, _ := os.ReadFile(p); string(b) != owner+"\r\n" {
			t.Errorf("the tool changed %s", p)
		}
	}
}

// B1: the report names every exe and dll hash and the RAM after each Ready.
func TestFullRunRecordsHashesAndRAM(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	rep, _ := runExpect(t, c, ExitPass, allPass)
	if rep.Mode != ModeFull || rep.Overall != "PASS" {
		t.Errorf("mode %q overall %q", rep.Mode, rep.Overall)
	}
	if h, _ := dataOf(rep, 1)["binary_sha256"].(map[string]any); len(h) == 0 {
		t.Error("no binary hashes in step 1")
	}
	if _, ok := dataOf(rep, 2)["host_ram_available_gib_after_ready"]; !ok {
		t.Error("step 2 lacks the RAM after Ready")
	}
	if _, ok := dataOf(rep, 6)["restart_host_ram_available_gib_after_ready"]; !ok {
		t.Error("step 6 lacks the RAM after the restart Ready")
	}
	if _, ok := dataOf(rep, 7)["external_host_ram_available_gib_after_ready"]; !ok {
		t.Error("step 7 lacks the RAM after the external Ready")
	}
	if dataOf(rep, 2)["disk_cache_proof"] == nil {
		t.Error("step 2 must say the disk cache is NOT PROVEN when the CLI reports no cache row")
	}
}

// The waiter's outer time limit: a stop file acts like Ctrl+C (exit 130, every
// recorded process stopped by exact PID).
func TestStopFileInterrupts(t *testing.T) {
	c := fakeConfig(t, "hang-load", "ok", nil)
	stop := filepath.Join(t.TempDir(), "STOP")
	args := []string{"--exe", c.Exe, "--manifest", c.Manifest, "--out", c.Out, "--port", fmt.Sprint(c.Port), "--lock-path", c.LockPath,
		"--build-lock-path", c.BuildLockPath, "--proc-patterns", "smoketest-none-*.exe", "--load-timeout", "60s", "--stop-timeout", "4s",
		"--keep-env-prefix", "BUBO_FAKE_,GORACE", "--stop-file", stop}
	cmd := exec.Command(smokeExe, args...)
	cmd.Env = os.Environ()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if b, _ := os.ReadFile(filepath.Join(c.Out, "pids.jsonl")); len(b) > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := os.WriteFile(stop, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		ee, ok := err.(*exec.ExitError)
		if !ok || ee.ExitCode() != ExitInterrupted {
			t.Fatalf("exit %v, want %d", err, ExitInterrupted)
		}
	case <-time.After(90 * time.Second):
		cmd.Process.Kill()
		t.Fatal("the stop file did not end the run")
	}
	rep := loadReport(t, c)
	if rep.Overall != "INTERRUPTED" || len(rep.PIDs) != 1 || isLive(rep.PIDs[0].PID, rep.PIDs[0].CreationTime) {
		t.Errorf("overall %q pids %+v", rep.Overall, rep.PIDs)
	}
}

// J12c: the wrapper writes a two-line lock (line 1 the owner line, line 2 pid/time/mode). The tool matches whole lines,
// so the second line is safe, and neither file may change.
func TestWrapperTwoLineLockAccepted(t *testing.T) {
	const owner = "owner=user-gui-smoke"
	c := fakeConfig(t, "normal", "ok", nil)
	c.AllowLockOwner, c.RequireLocks = owner, true
	second := "pid=4242 born=2026-10-14T07:00:00.0000000Z mode=Full kind=gpu-harness wrapper_started=2026-10-14T03:00:00.0000000-04:00\n"
	for _, p := range []string{c.LockPath, c.BuildLockPath} {
		os.WriteFile(p, []byte(owner+"\n"+second), 0o600)
	}
	runExpect(t, c, ExitPass, allPass)
	for _, p := range []string{c.LockPath, c.BuildLockPath} {
		if b, _ := os.ReadFile(p); string(b) != owner+"\n"+second {
			t.Errorf("the tool changed %s", p)
		}
	}
	// Line 1 of another owner with the allowed text only in a later line still matches (any whole line); a different line 1 alone does not.
	c2 := fakeConfig(t, "normal", "ok", nil)
	c2.AllowLockOwner, c2.RequireLocks = owner, true
	os.WriteFile(c2.LockPath, []byte("owner=other\n"+second), 0o600)
	os.WriteFile(c2.BuildLockPath, []byte(owner+"\n"+second), 0o600)
	runExpect(t, c2, ExitRefused, [10]string{})
}

// N16: the raw `status --json --instance-id` payload of every instance-specific call is saved under --out.
func TestRawStatusSavedForInstanceCalls(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	runExpect(t, c, ExitPass, allPass)
	b, err := os.ReadFile(filepath.Join(c.Out, "status-instance-raw.jsonl"))
	if err != nil {
		t.Fatalf("raw status file missing: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) < 4 {
		t.Errorf("want at least 4 raw status lines (step 2, 3/4, 5, step 7 external), got %d", len(lines))
	}
	for _, l := range lines {
		var m map[string]any
		if json.Unmarshal([]byte(l), &m) != nil {
			t.Errorf("not one JSON object per line: %.80s", l)
		}
	}
}
