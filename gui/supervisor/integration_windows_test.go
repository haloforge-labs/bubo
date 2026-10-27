//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

var fakeExe string
var updateHelperExe string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "bubo-supervisor-tests-")
	if err != nil {
		panic(err)
	}
	fakeExe = filepath.Join(dir, "bubo.exe")
	cmd := exec.Command("go", "build", "-race", "-o", fakeExe, "./fakecli")
	cmd.Dir = filepath.Join("..", "tests")
	cmd.Env = append(os.Environ(), "GOFLAGS=-mod=readonly")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fake build: %v\n%s", err, out)
		os.RemoveAll(dir)
		os.Exit(1)
	}
	updateHelperExe = filepath.Join(dir, "bubo-update.exe")
	cmd = exec.Command("go", "build", "-ldflags=-H=windowsgui", "-o", updateHelperExe, "./testdata/update-helper")
	out, err = cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "update helper seam build: %v\n%s", err, out)
		os.RemoveAll(dir)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}
func configFor(t *testing.T, mode string) Config {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("GORACE", "atexit_sleep_ms=0")
	t.Setenv("BUBO_FAKE_DIR", dir)
	t.Setenv("BUBO_FAKE_MODE", mode)
	t.Setenv("BUBO_FAKE_FIXTURES", contractDir())
	loadTimeout := 5 * time.Second
	if mode == "hang-load" || mode == "disagreement" {
		loadTimeout = 250 * time.Millisecond
	}
	stopTimeout := 3 * time.Second
	if mode == "lose-pipe" {
		stopTimeout = time.Second // the lost pipe is judged after the stop bound
	}
	return Config{Exe: fakeExe, Port: 19091, LogDir: filepath.Join(dir, "logs"), LoadTimeout: loadTimeout, StopTimeout: stopTimeout, DrainTimeout: 200 * time.Millisecond, QueryTimeout: 5 * time.Second, PollInterval: 40 * time.Millisecond, InitialPrepared: true}
}
func supervisorFor(t *testing.T, mode string) *Supervisor {
	t.Helper()
	s, err := New(configFor(t, mode))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.finished:
			return
		default:
		}
		reply := s.Exit()
		select {
		case <-reply:
		case <-time.After(5 * time.Second):
			t.Error("Exit did not finish")
		}
	})
	return s
}
func reply(t *testing.T, ch <-chan ResultReply) {
	t.Helper()
	select {
	case r := <-ch:
		if r.Err != nil {
			t.Fatal(r.Err)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("action reply timeout")
	}
}
func waitState(t *testing.T, s *Supervisor, state State) Snapshot {
	t.Helper()
	deadline := time.Now().Add(6 * time.Second)
	for time.Now().Before(deadline) {
		v := s.Snapshot()
		if v.State == state {
			return v
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("wanted %s, got %+v", state, s.Snapshot())
	return Snapshot{}
}
func waitNoOperation(t *testing.T, s *Supervisor) Snapshot {
	t.Helper()
	deadline := time.Now().Add(6 * time.Second)
	for time.Now().Before(deadline) {
		v := s.Snapshot()
		if v.Instance == nil && (v.State == Failed || v.State == Stopped) {
			return v
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("operation did not exit", s.Snapshot())
	return Snapshot{}
}
func TestReadyStopRestart(t *testing.T) {
	s := supervisorFor(t, "split")
	reply(t, s.Start())
	first := waitState(t, s, Ready)
	reply(t, s.Restart())
	second := waitState(t, s, Ready)
	deadline := time.Now().Add(4 * time.Second)
	for second.Generation == first.Generation && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
		second = s.Snapshot()
	}
	second = waitState(t, s, Ready)
	if second.Generation <= first.Generation || second.Instance.ID == first.Instance.ID {
		t.Fatal("restart reused identity", first, second)
	}
	s.emit(update{first.Generation, "failure", Failure{Code: "device_lost"}})
	time.Sleep(40 * time.Millisecond)
	if s.Snapshot().State != Ready {
		t.Fatal("old operation mutated state")
	}
	reply(t, s.Stop())
	v := waitState(t, s, Stopped)
	if v.Forced {
		t.Fatal("clean stop marked forced")
	}
}
func TestPrepareConsent(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Prepare())
	deadline := time.Now().Add(3 * time.Second)
	for s.Snapshot().Approval == nil && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if s.Snapshot().Approval == nil {
		t.Fatal("no consent event")
	}
	reply(t, s.Consent(true))
	waitState(t, s, Stopped)
}
func TestPrepareDecline(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Prepare())
	deadline := time.Now().Add(3 * time.Second)
	for s.Snapshot().Approval == nil && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	reply(t, s.Consent(false))
	// A declined plan is the user's answer, not a failure.
	v := waitFor(t, s, "the end of the declined plan", 6*time.Second, func(v Snapshot) bool { return v.Instance == nil && v.State != Preparing })
	if v.State != Stopped || v.LastError != nil {
		t.Fatalf("declining must end stopped, not failed: %s %+v", v.State, v.LastError)
	}
}
func TestFailuresAndForce(t *testing.T) {
	for _, mode := range []string{"before-ready", "invalid-json", "truncated", "hang-load", "hang-stop", "inherited", "disagreement", "unsupported", "bad-product", "lose-pipe"} {
		t.Run(mode, func(t *testing.T) {
			s := supervisorFor(t, mode)
			reply(t, s.Start())
			if mode == "hang-stop" {
				waitState(t, s, Ready)
				reply(t, s.Stop())
			}
			v := waitNoOperation(t, s)
			if v.State != Failed || v.LastError == nil {
				t.Fatal("failure lost", v)
			}
			if mode == "hang-stop" && !v.Forced {
				t.Fatal("hang stop was reported clean")
			}
			expected := map[string]string{"before-ready": "server_exited", "invalid-json": "protocol_mismatch", "truncated": "protocol_mismatch", "hang-load": "startup_timeout", "hang-stop": "stop_timeout", "inherited": "stop_timeout", "disagreement": "startup_timeout", "unsupported": "protocol_mismatch", "bad-product": "protocol_mismatch", "lose-pipe": "protocol_mismatch"}
			if v.LastError.Code != expected[mode] {
				t.Fatalf("wrong failure: %+v", v.LastError)
			}
			t.Logf("FIXTURE %s state=%s forced=%v error=%s", mode, v.State, v.Forced, v.LastError.Code)
		})
	}
}
func TestLostStatusIsStale(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	waitState(t, s, Ready)
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "lose-status"), nil, 0600)
	deadline := time.Now().Add(time.Second)
	for !s.Snapshot().Stale && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	v := s.Snapshot()
	if !v.Stale || v.State != Ready {
		t.Fatal("lost sample became exit", v)
	}
	os.Remove(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "lose-status"))
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}
func TestFloodHeartbeatAndBoundedLogs(t *testing.T) {
	s := supervisorFor(t, "flood")
	var maxGap atomic.Int64
	done := make(chan struct{})
	heartbeatDone := make(chan struct{})
	go func() {
		defer close(heartbeatDone)
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		last := time.Now()
		for {
			select {
			case <-tick.C:
				now := time.Now()
				gap := now.Sub(last).Nanoseconds()
				for old := maxGap.Load(); gap > old; old = maxGap.Load() {
					if maxGap.CompareAndSwap(old, gap) {
						break
					}
				}
				last = now
			case <-done:
				return
			}
		}
	}()
	reply(t, s.Start())
	waitState(t, s, Ready)
	time.Sleep(1500 * time.Millisecond)
	reply(t, s.Stop())
	waitNoOperation(t, s)
	close(done)
	<-heartbeatDone
	gap := time.Duration(maxGap.Load())
	t.Logf("HEARTBEAT max_gap=%s period=100ms", gap)
	if gap > 500*time.Millisecond {
		t.Fatal("heartbeat blocked", gap)
	}
	if len(s.Logs()) > 2000 {
		t.Fatal("ring exceeded bound")
	}
	entries, _ := filepath.Glob(filepath.Join(s.config.LogDir, "stderr.*.log"))
	if len(entries) > 3 {
		t.Fatal("too many files")
	}
	for _, p := range entries {
		st, _ := os.Stat(p)
		if st.Size() > 10<<20 {
			t.Fatal("log exceeded size", st.Size())
		}
	}
}
func readPIDs(t *testing.T) []int {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "pids.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var pids []int
	for _, s := range strings.Fields(string(b)) {
		pid, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		pids = append(pids, pid)
	}
	return pids
}
func assertNoOrphans(t *testing.T, pids []int) {
	t.Helper()
	var text []string
	for _, pid := range pids {
		text = append(text, strconv.Itoa(pid))
	}
	script := "$ids=@(" + strings.Join(text, ",") + "); $alive=@(Get-Process -Id $ids -ErrorAction SilentlyContinue); Write-Output ('EXACT_PIDS='+($ids -join ',')); Write-Output ('Get-Process alive='+$alive.Count); if($alive.Count -ne 0){exit 1}"
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).CombinedOutput()
	t.Log(strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatal("orphan check failed", err)
	}
}
func TestJobKillsChildAndGrandchild(t *testing.T) {
	s := supervisorFor(t, "spawn")
	reply(t, s.Start())
	waitState(t, s, Ready)
	deadline := time.Now().Add(2 * time.Second)
	for len(readPIDs(t)) < 3 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	pids := readPIDs(t)
	if len(pids) != 3 {
		t.Fatal("missing descendant", pids)
	}
	reply(t, s.Stop())
	v := waitNoOperation(t, s)
	if !v.Forced || v.State != Failed {
		t.Fatal("force outcome", v)
	}
	assertNoOrphans(t, pids)
	t.Logf("FORCED_STOP recorded=%v", v.Forced)
}
func TestInheritedPipesBounded(t *testing.T) {
	c := configFor(t, "inherited")
	p, err := launchOwned(c.Exe, []string{"serve", "--host", "127.0.0.1", "--port", "19091"}, c.DrainTimeout, func([]byte, bool) {}, func([]byte, bool) {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case r := <-p.Done:
		if !r.DrainTimedOut {
			t.Fatal("inherited pipes did not time out")
		}
	case <-time.After(3 * time.Second):
		p.Force()
		t.Fatal("drain hung")
	}
	pids := readPIDs(t)
	assertNoOrphans(t, pids)
}
func TestArgumentQuoting(t *testing.T) {
	c := configFor(t, "normal")
	args := []string{"", "a b", `C:\space dir\`, "é水", `a"b`, "& | ; $() ` < >", strings.Repeat(`\`, 4) + `"`}
	var got []string
	p, err := launchOwned(c.Exe, append([]string{"echo"}, args...), c.DrainTimeout, func(b []byte, _ bool) {
		if err := json.Unmarshal(b, &got); err != nil {
			t.Error(err)
		}
	}, func([]byte, bool) {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	<-p.Done
	if fmt.Sprint(got) != fmt.Sprint(args) {
		t.Fatal(got, args)
	}
}
func TestAssignmentFailure(t *testing.T) {
	c := configFor(t, "normal")
	var pid uint32
	_, err := launchOwned(c.Exe, []string{"serve"}, c.DrainTimeout, func([]byte, bool) {}, func([]byte, bool) {}, func(_ windows.Handle, h windows.Handle) error {
		pid, _ = windows.GetProcessId(h)
		return errors.New("injected assignment failure")
	})
	if err == nil {
		t.Fatal("assignment failure ignored")
	}
	if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "pids.txt")); !os.IsNotExist(err) {
		t.Fatal("suspended child ran")
	}
	assertNoOrphans(t, []int{int(pid)})
}
func startExternal(t *testing.T, c Config) (*OwnedProcess, Instance) {
	t.Helper()
	p, err := launchOwned(c.Exe, []string{"serve", "--host", "127.0.0.1", "--port", "19091"}, c.DrainTimeout, func([]byte, bool) {}, func([]byte, bool) {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Force(); <-p.Done })
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "instance.json"))
		var i Instance
		if json.Unmarshal(b, &i) == nil && i.Ready {
			return p, i
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("external did not start")
	return nil, Instance{}
}
func TestPIDReuseRejected(t *testing.T) {
	c := configFor(t, "normal")
	_, i := startExternal(t, c)
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal(err)
	}
	i.CreationTime = "1"
	if err := VerifyIdentity(i, c.Exe); err == nil {
		t.Fatal("PID reuse accepted")
	}
	t.Logf("PID_REUSE exact_pid=%d wrong_creation_time=1 rejected", i.PID)
}
func TestExternalReadOnlyAndManage(t *testing.T) {
	c := configFor(t, "normal")
	p, i := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	reply(t, s.SelectInstance(i.ID))
	v := waitState(t, s, AttachedExternal)
	if v.Managed || v.Label != "Started outside this window" {
		t.Fatal(v)
	}
	if r := <-s.Stop(); r.Err == nil {
		t.Fatal("readonly stop accepted")
	}
	reply(t, s.Exit())
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal("GUI exit touched external", err)
	}
	s, err = New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	reply(t, s.Manage())
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.Exit())
	select {
	case <-p.Done:
	case <-time.After(time.Second):
		t.Fatal("external graceful stop did not exit")
	}
}
func TestSettings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := DefaultSettings()
	if s.CloseToTray || s.StartGUIAtLogin || s.StartServerWithGUI {
		t.Fatal("opt-ins enabled")
	}
	s.LastError = &Failure{Code: "device_lost", Source: "cli", Remedy: "secret token user path", Time: time.Now().UTC()}
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	s.CloseToTray = true
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if strings.Contains(string(b), "secret") {
		t.Fatal("error was not redacted")
	}
	backup, _ := os.ReadFile(path + ".bak")
	if strings.Contains(string(backup), `"close_to_tray": true`) {
		t.Fatal("backup overwritten")
	}
	os.WriteFile(path, []byte("damaged"), 0600)
	loaded, warning := LoadSettings(path)
	if warning == "" || loaded.CloseToTray || loaded.StartGUIAtLogin || loaded.StartServerWithGUI {
		t.Fatal(loaded, warning)
	}
}
func TestLogRotation(t *testing.T) {
	dir := t.TempDir()
	l, err := newLogs(dir, 100)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 2100; n++ {
		l.Append([]byte(strings.Repeat("x", 40)), true)
	}
	if l.Err() != nil {
		t.Fatal(l.Err())
	}
	if len(l.Ring()) != 2000 {
		t.Fatal("ring size")
	}
	l.Close()
	files, _ := filepath.Glob(filepath.Join(dir, "stderr.*.log"))
	if len(files) != 3 {
		t.Fatal(files)
	}
	for _, f := range files {
		st, _ := os.Stat(f)
		if st.Size() > 100 {
			t.Fatal(st.Size())
		}
	}
}
func TestSingleInstance(t *testing.T) {
	sid, err := UserSID()
	if err != nil {
		t.Fatal(err)
	}
	suffix := ".test." + newID()
	first, primary, err := acquireSingleInstance(sid, suffix)
	if err != nil || !primary {
		t.Fatal(primary, err)
	}
	defer first.Close()
	_, primary, err = acquireSingleInstance(sid, suffix)
	if err != nil || primary {
		t.Fatal(primary, err)
	}
	select {
	case <-first.ShowWindow:
	case <-time.After(time.Second):
		t.Fatal("Show window not received")
	}
}
func TestAbandonedMutex(t *testing.T) {
	c := configFor(t, "normal")
	sid, _ := UserSID()
	suffix := ".test." + newID()
	name := "Local\\Bubo.Control." + sid + suffix
	p, err := launchOwned(c.Exe, []string{"mutex-abandon", name}, c.DrainTimeout, func([]byte, bool) {}, func([]byte, bool) {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "mutex-ready")); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	ptr, _ := windows.UTF16PtrFromString(name)
	keep, err := windows.OpenMutex(windows.SYNCHRONIZE, false, ptr)
	if err != nil {
		p.Force()
		t.Fatal(err)
	}
	defer windows.CloseHandle(keep)
	p.Force()
	<-p.Done
	first, primary, err := acquireSingleInstance(sid, suffix)
	if err != nil || !primary || !first.Abandoned {
		t.Fatal(primary, err, first)
	}
	first.Close()
}

// contractDir is the frozen CLI contract folder (schemas, fixtures, index).
func contractDir() string {
	p, err := filepath.Abs(filepath.Join("..", "..", "tests", "cli", "contract"))
	if err != nil {
		panic(err)
	}
	return p
}
