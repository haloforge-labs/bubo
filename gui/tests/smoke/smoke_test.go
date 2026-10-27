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

var fakeExe, smokeExe string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "bubo-smoke-tests-")
	if err != nil {
		panic(err)
	}
	fakeExe = filepath.Join(dir, "bubo.exe")
	smokeExe = filepath.Join(dir, "bubo-smoke.exe")
	for _, b := range [][2]string{{fakeExe, "./fakecli"}, {smokeExe, "./smoke"}} {
		cmd := exec.Command("go", "build", "-o", b[0], b[1])
		cmd.Dir = ".."
		cmd.Env = append(os.Environ(), "GOFLAGS=-mod=readonly")
		if out, err := cmd.CombinedOutput(); err != nil {
			fmt.Fprintf(os.Stderr, "build %s: %v\n%s", b[1], err, out)
			os.RemoveAll(dir)
			os.Exit(1)
		}
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func contractDir() string {
	p, _ := filepath.Abs(filepath.Join("..", "..", "..", "tests", "cli", "contract"))
	return p
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// fakeConfig prepares the fake CLI environment and a smoke Config for it.
func fakeConfig(t *testing.T, mode, httpMode string, extra map[string]string) Config {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("GORACE", "atexit_sleep_ms=0")
	t.Setenv("BUBO_FAKE_DIR", dir)
	t.Setenv("BUBO_FAKE_MODE", mode)
	t.Setenv("BUBO_FAKE_FIXTURES", contractDir())
	t.Setenv("BUBO_FAKE_HTTP", httpMode)
	for k, v := range extra {
		t.Setenv(k, v)
	}
	// Keep the process lifecycle isolated, with the real prepared package
	// identity for the launch-time estimate enforced by the supervisor.
	record, err := os.ReadFile(filepath.Join(contractDir(), "..", "..", "..", "gui", "tests", "scenarios", "recordings", "v12", "status-main.json"))
	if err != nil {
		t.Fatal(err)
	}
	var status struct {
		Payload struct {
			PreparedDefault string `json:"prepared_default"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(record, &status); err != nil {
		t.Fatal(err)
	}
	manifest := status.Payload.PreparedDefault
	if manifest == "" {
		t.Fatal("recorded prepared package is missing")
	}
	// Materialize the recorded package identity locally; no installed model is needed.
	packageID := filepath.Base(filepath.Dir(manifest))
	modelDir := filepath.Join(dir, "models", packageID)
	if err := os.MkdirAll(modelDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.gguf"), []byte("fake-gguf"), 0600); err != nil {
		t.Fatal(err)
	}
	marker, err := json.Marshal(map[string]any{"package_id": packageID, "files": []string{"model.gguf"}})
	if err != nil {
		t.Fatal(err)
	}
	manifest = filepath.Join(modelDir, "bubo-package.json")
	if err := os.WriteFile(manifest, marker, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_PACKAGES", "1")
	return Config{
		Exe: fakeExe, Manifest: manifest, Out: filepath.Join(dir, "out"), Port: freePort(t),
		LockPath:      filepath.Join(dir, "gpu-harness.lock"),
		BuildLockPath: filepath.Join(dir, "build.lock"),
		ProcPatterns:  []string{"smoketest-none-*.exe"}, // never matches a real process
		LoadTimeout:   20 * time.Second, StopTimeout: 4 * time.Second, RequestTimeout: 20 * time.Second,
		QueryTimeout: 10 * time.Second, PollInterval: 100 * time.Millisecond, IdleWatch: 1500 * time.Millisecond, SlotIdleWait: 2 * time.Second,
		KeepEnv: []string{"BUBO_FAKE_", "GORACE"},
	}
}

func loadReport(t *testing.T, c Config) Report {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(c.Out, "report.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Report
	if err = json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func status(r Report, n int) string {
	for _, s := range r.Steps {
		if s.Step == n {
			return s.Status
		}
	}
	return "missing"
}

// want asserts the exit code and the status of every step ("" skips a step).
func runExpect(t *testing.T, c Config, code int, steps [10]string) (Report, string) {
	t.Helper()
	var console bytes.Buffer
	got := run(context.Background(), c, &console)
	rep := loadReport(t, c)
	if got != code {
		t.Fatalf("exit %d, want %d\n%s", got, code, console.String())
	}
	for n := 1; n <= 9; n++ {
		if steps[n] != "" && status(rep, n) != steps[n] {
			t.Errorf("step %d is %s, want %s", n, status(rep, n), steps[n])
		}
	}
	if t.Failed() {
		t.Logf("console:\n%s", console.String())
	}
	if rep.Overall == "" {
		t.Error("report has no overall verdict")
	}
	return rep, console.String()
}

var allPass = [10]string{"", Pass, Pass, Pass, Pass, Pass, Pass, Pass, Pass, Pass}

func TestPassPath(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	rep, console := runExpect(t, c, ExitPass, allPass)
	if rep.Overall != "PASS" {
		t.Error("overall", rep.Overall)
	}
	if len(rep.PIDs) != 3 {
		t.Errorf("ledger holds %d pids, want 3 (start, restart, external): %+v", len(rep.PIDs), rep.PIDs)
	}
	for _, p := range rep.PIDs {
		if p.AliveAtEnd == nil || *p.AliveAtEnd || p.PID == 0 || p.CreationTime == "" {
			t.Errorf("bad ledger row %+v", p)
		}
		if isLive(p.PID, p.CreationTime) {
			t.Errorf("pid %d is alive after the run", p.PID)
		}
	}
	// One printed verdict line per step.
	for n := 1; n <= 9; n++ {
		if !strings.Contains(console, fmt.Sprintf("step %d ", n)) || !strings.Contains(console, ": PASS (") {
			t.Errorf("no verdict line for step %d", n)
		}
	}
	// No prompt text and no planted key in any output file.
	filepath.Walk(c.Out, func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi.IsDir() {
			return nil
		}
		b, _ := os.ReadFile(p)
		if bytes.Contains(b, []byte("quick brown fox")) || bytes.Contains(b, []byte("SMOKE-PLANTED")) {
			t.Errorf("%s holds prompt text or the planted key", p)
		}
		return nil
	})
	for _, name := range []string{"report.json", "transcript.txt", "pids.jsonl"} {
		if _, err := os.Stat(filepath.Join(c.Out, name)); err != nil {
			t.Errorf("missing %s", name)
		}
	}

	var s3 StepResult
	for _, s := range rep.Steps {
		if s.Step == 3 {
			s3 = s
		}
	}
	r2, _ := s3.Data["request_2"].(map[string]any)
	if r2 == nil || r2["cached_tokens"].(float64) <= 0 {
		t.Errorf("step 3 request_2 data %v", s3.Data)
	}
}

func TestRefusedPorts(t *testing.T) {
	for _, port := range []int{8081, 8082, 8083, 8084, 8085, 8086} {
		c := fakeConfig(t, "normal", "ok", nil)
		c.Port = port
		rep, _ := runExpect(t, c, ExitRefused, [10]string{})
		if rep.Overall != "REFUSED" || len(rep.Steps) != 0 || len(rep.PIDs) != 0 {
			t.Errorf("port %d: %+v", port, rep)
		}
	}
}

func TestRefusedBusyPort(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", c.Port))
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	_, console := runExpect(t, c, ExitRefused, [10]string{})
	if !strings.Contains(console, "occupied") {
		t.Error(console)
	}
}

func TestLockRules(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	if err := os.WriteFile(c.LockPath, []byte("owner=timing-arm-g2\nstarted=now\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Lock held by someone else: refused, and the message names the owner line.
	_, console := runExpect(t, c, ExitRefused, [10]string{})
	if !strings.Contains(console, "owner=timing-arm-g2") {
		t.Error("refusal does not name the lock owner:", console)
	}
	// A different owner line passed: still refused.
	c2 := fakeConfig(t, "normal", "ok", nil)
	os.WriteFile(c2.LockPath, []byte("owner=timing-arm-g2\n"), 0o600)
	c2.AllowLockOwner = "owner=user-gui-smoke"
	runExpect(t, c2, ExitRefused, [10]string{})
	// The right owner line: the run proceeds, and the lock file is untouched.
	c3 := fakeConfig(t, "normal", "ok", nil)
	os.WriteFile(c3.LockPath, []byte("owner=user-gui-smoke\n"), 0o600)
	c3.AllowLockOwner = "owner=user-gui-smoke"
	runExpect(t, c3, ExitPass, allPass)
	if b, _ := os.ReadFile(c3.LockPath); string(b) != "owner=user-gui-smoke\n" {
		t.Error("the tool changed the lock file")
	}
}

func TestUsageErrors(t *testing.T) {
	var buf bytes.Buffer
	if got := run(context.Background(), Config{}, &buf); got != ExitUsage {
		t.Errorf("empty config exit %d", got)
	}
	c := fakeConfig(t, "normal", "ok", nil)
	c.Exe = "bubo.exe"
	if got := run(context.Background(), c, &buf); got != ExitUsage {
		t.Errorf("relative exe exit %d", got)
	}
	c = fakeConfig(t, "normal", "ok", nil)
	os.MkdirAll(c.Out, 0o700)
	os.WriteFile(filepath.Join(c.Out, "old.txt"), nil, 0o600)
	if got := run(context.Background(), c, &buf); got != ExitUsage {
		t.Errorf("non-empty out exit %d", got)
	}
}

func TestNeverReady(t *testing.T) {
	c := fakeConfig(t, "hang-load", "ok", nil)
	c.LoadTimeout = 700 * time.Millisecond
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Fail, NotRun, NotRun, NotRun, NotRun, NotRun, Pass, Pass})
	if len(rep.PIDs) != 1 {
		t.Errorf("ledger %+v", rep.PIDs)
	}
	for _, p := range rep.PIDs {
		if isLive(p.PID, p.CreationTime) {
			t.Errorf("orphan pid %d", p.PID)
		}
	}
}

func TestRequestFails(t *testing.T) {
	c := fakeConfig(t, "normal", "fail", nil)
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Fail, NotRun, Pass, Fail, Pass, Pass, Pass})
	for _, s := range rep.Steps {
		if s.Step == 3 && !strings.Contains(s.Reason, "500") {
			t.Errorf("reason %q does not name HTTP 500", s.Reason)
		}
	}
}

func TestNoCacheReuseFails(t *testing.T) {
	c := fakeConfig(t, "normal", "nocache", nil)
	// M5: a cache miss is a step 3 FAIL and shows again in step 4; step 4 is not hidden.
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Fail, Fail, Pass, Pass, Pass, Pass, Pass})
	for _, s := range rep.Steps {
		if s.Step == 3 && !strings.Contains(s.Reason, "cache reuse") {
			t.Errorf("reason %q", s.Reason)
		}
		if s.Step == 4 && !strings.Contains(s.Reason, "cached counter") {
			t.Errorf("step 4 reason %q does not show the cache miss", s.Reason)
		}
	}
}

func TestSlotNeverIdleFails(t *testing.T) {
	c := fakeConfig(t, "normal", "stuck", nil)
	runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Pass, Fail, Fail, Pass, Pass, Pass, Pass})
}

func TestStopHangsIsForcedAndReported(t *testing.T) {
	c := fakeConfig(t, "hang-stop", "ok", nil)
	c.StopTimeout = time.Second
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Pass, Pass, Pass, Fail, Fail, Fail, Pass})
	if len(rep.PIDs) < 2 {
		t.Fatalf("ledger %+v", rep.PIDs)
	}
	for _, p := range rep.PIDs {
		if isLive(p.PID, p.CreationTime) {
			t.Errorf("pid %d still alive after cleanup", p.PID)
		}
	}
	for _, s := range rep.Steps {
		if s.Step == 6 && !strings.Contains(s.Reason, "Failed") {
			t.Errorf("step 6 reason %q does not show the forced stop", s.Reason)
		}
	}
}

func TestBatteryFailsPreflight(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_POWER": "battery"})
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Fail, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun})
	if len(rep.PIDs) != 0 {
		t.Error("nothing should have started", rep.PIDs)
	}
	c = fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_POWER": "battery"})
	c.AllowBattery = true
	runExpect(t, c, ExitPass, allPass)
}

func TestFreeRAMFloor(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	c.MinFreeRAMGiB = 1 << 20
	runExpect(t, c, ExitFail, [10]string{"", Fail, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun, NotRun})
}

func TestLeakedKeyInBundleFails(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", map[string]string{"BUBO_FAKE_BUNDLE_LEAK": "1"})
	rep, _ := runExpect(t, c, ExitFail, [10]string{"", Pass, Pass, Pass, Pass, Pass, Pass, Pass, Pass, Fail})
	for _, s := range rep.Steps {
		if s.Step == 9 && !strings.Contains(s.Reason, "planted key") {
			t.Errorf("reason %q", s.Reason)
		}
	}
	// The report itself must not carry the key.
	b, _ := os.ReadFile(filepath.Join(c.Out, "report.json"))
	if bytes.Contains(b, []byte("SMOKE-PLANTED")) {
		t.Error("the planted key is in report.json")
	}
}

func TestCtrlCStopsOnlyRecordedPIDs(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	// A bystander fake process that is NOT in the ledger must survive.
	by := exec.Command(fakeExe, "grandchild")
	by.Env = os.Environ()
	if err := by.Start(); err != nil {
		t.Fatal(err)
	}
	defer by.Process.Kill()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan int, 1)
	var console bytes.Buffer
	go func() { done <- run(ctx, c, &console) }()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(filepath.Join(c.Out, "report.json"))
		var rep Report
		if json.Unmarshal(b, &rep) == nil && status(rep, 2) == Pass {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	var code int
	select {
	case code = <-done:
	case <-time.After(90 * time.Second):
		t.Fatal("run did not stop after cancel")
	}
	if code != ExitInterrupted {
		t.Fatalf("exit %d\n%s", code, console.String())
	}
	rep := loadReport(t, c)
	if rep.Overall != "INTERRUPTED" || !rep.Interrupt {
		t.Error("overall", rep.Overall)
	}
	if len(rep.PIDs) == 0 {
		t.Fatal("no pids recorded")
	}
	for _, p := range rep.PIDs {
		if isLive(p.PID, p.CreationTime) {
			t.Errorf("recorded pid %d alive after Ctrl+C", p.PID)
		}
	}
	// Step 3 may already have finished when the cancel lands (the fake answers in
	// milliseconds: this was a 1-in-6 flake at 6dfb7007); the later steps must not run.
	if status(rep, 6) != NotRun || status(rep, 7) != NotRun {
		t.Errorf("step 6 is %s, step 7 is %s", status(rep, 6), status(rep, 7))
	}
	if ct, err := creationTimeOf(uint32(by.Process.Pid)); err != nil || !isLive(uint32(by.Process.Pid), ct) {
		t.Error("the bystander process was touched")
	}
}

func TestLedgerCleanupIsExactPID(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("BUBO_FAKE_DIR", dir)
	rec, err := NewRecorder(filepath.Join(dir, "out"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Close()
	led := NewLedger(rec, filepath.Join(dir, "out"))
	spawn := func() (*exec.Cmd, uint32, string) {
		cmd := exec.Command(fakeExe, "grandchild")
		cmd.Env = os.Environ()
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
		pid := uint32(cmd.Process.Pid)
		ct, err := creationTimeOf(pid)
		if err != nil {
			t.Fatal(err)
		}
		return cmd, pid, ct
	}
	_, ownPID, ownCT := spawn()
	_, otherPID, otherCT := spawn()
	_, reusedPID, _ := spawn()
	led.Add(ownPID, ownCT, "test")
	led.Add(reusedPID, "1", "test, wrong creation time (PID reuse)")
	if alive := led.Alive(); len(alive) != 1 || alive[0].PID != ownPID {
		t.Fatalf("Alive() = %+v", alive)
	}
	stopped, stuck := led.Cleanup()
	if len(stopped) != 1 || stopped[0].PID != ownPID || len(stuck) != 0 {
		t.Fatalf("stopped %+v stuck %+v", stopped, stuck)
	}
	if isLive(ownPID, ownCT) {
		t.Error("the recorded process is still alive")
	}
	if !isLive(otherPID, otherCT) {
		t.Error("an unrecorded process was killed")
	}
	if ct, err := creationTimeOf(reusedPID); err != nil || !isLive(reusedPID, ct) {
		t.Error("a process with a mismatched creation time was killed")
	}
}

func TestEnvScrubKeepsOnlyListed(t *testing.T) {
	t.Setenv("GUFO_SMOKE_PROBE", "1")
	t.Setenv("HIP_SMOKE_PROBE", "1")
	t.Setenv("BUBO_FAKE_KEEP", "1")
	t.Setenv("PATH_KEEP_SMOKE", "1")
	removed, restore := scrubEnv([]string{"BUBO_FAKE_"})
	if os.Getenv("GUFO_SMOKE_PROBE") != "" || os.Getenv("HIP_SMOKE_PROBE") != "" {
		t.Error("experiment variables survived")
	}
	if os.Getenv("BUBO_FAKE_KEEP") != "1" || os.Getenv("PATH_KEEP_SMOKE") != "1" {
		t.Error("kept variables were removed")
	}
	joined := strings.Join(removed, ",")
	if !strings.Contains(joined, "GUFO_SMOKE_PROBE") || !strings.Contains(joined, "HIP_SMOKE_PROBE") {
		t.Error("removed list", joined)
	}
	restore()
	if os.Getenv("GUFO_SMOKE_PROBE") != "1" {
		t.Error("restore did not bring the variable back")
	}
}

// The real process: refusal and usage exit codes come from main().
func TestBinaryExitCodes(t *testing.T) {
	c := fakeConfig(t, "normal", "ok", nil)
	runBin := func(args ...string) int {
		cmd := exec.Command(smokeExe, args...)
		cmd.Env = os.Environ()
		err := cmd.Run()
		if err == nil {
			return 0
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		t.Fatal(err)
		return -1
	}
	if got := runBin("--exe", c.Exe, "--manifest", c.Manifest, "--out", c.Out, "--port", "8083", "--lock-path", c.LockPath, "--build-lock-path", c.BuildLockPath); got != ExitRefused {
		t.Errorf("port 8083 exit %d", got)
	}
	if got := runBin("--bogus"); got != ExitUsage {
		t.Errorf("bad flag exit %d", got)
	}
	out2 := filepath.Join(filepath.Dir(c.Out), "out2")
	args := []string{"--exe", c.Exe, "--manifest", c.Manifest, "--out", out2, "--port", fmt.Sprint(freePort(t)), "--lock-path", c.LockPath, "--build-lock-path", c.BuildLockPath, "--proc-patterns", "smoketest-none-*.exe",
		"--load-timeout", "20s", "--stop-timeout", "4s", "--idle-watch", "1500ms", "--slot-idle-wait", "2s", "--poll-interval", "100ms", "--keep-env-prefix", "BUBO_FAKE_,GORACE"}
	if got := runBin(args...); got != ExitPass {
		b, _ := os.ReadFile(filepath.Join(out2, "transcript.txt"))
		t.Errorf("binary pass run exit %d\n%s", got, b)
	}
}

func TestCtrlCDuringLoad(t *testing.T) {
	c := fakeConfig(t, "hang-load", "ok", nil)
	c.LoadTimeout = time.Minute
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan int, 1)
	var console bytes.Buffer
	go func() { done <- run(ctx, c, &console) }()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(filepath.Join(c.Out, "pids.jsonl"))
		if len(b) > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	select {
	case code := <-done:
		if code != ExitInterrupted {
			t.Fatalf("exit %d: %s", code, console.String())
		}
	case <-time.After(90 * time.Second):
		t.Fatal("run did not stop after cancel")
	}
	rep := loadReport(t, c)
	if len(rep.PIDs) != 1 || isLive(rep.PIDs[0].PID, rep.PIDs[0].CreationTime) {
		t.Errorf("pids %+v", rep.PIDs)
	}
	if status(rep, 2) != Fail || status(rep, 7) != NotRun {
		t.Errorf("step 2 %s, step 7 %s", status(rep, 2), status(rep, 7))
	}
}
