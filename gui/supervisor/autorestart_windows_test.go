//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The engine exits `bubo serve` with status 75 when it lost the GPU. The
// supervisor restarts the server once, after a short delay, with the same
// setup; a second 75 inside the window stops with the error. These tests run
// the real supervisor against the fake CLI, whose "exit-serve" file scripts
// the exit status.

func autoRestartSupervisor(t *testing.T, delay, window time.Duration) (*Supervisor, string) {
	t.Helper()
	c := configFor(t, "split")
	c.AutoRestartDelay, c.AutoRestartWindow = delay, window
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.finished:
			return
		default:
		}
		select {
		case <-s.Exit():
		case <-time.After(5 * time.Second):
			t.Error("Exit did not finish")
		}
	})
	return s, os.Getenv("BUBO_FAKE_DIR")
}

func serveLaunches(t *testing.T, dir string) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "serve-argv.txt"))
	if err != nil {
		return nil
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

func script(t *testing.T, dir, text string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "exit-serve"), []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestExit75RestartsOnceWithTheSameSetupAndSaysSo(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 300*time.Millisecond, time.Minute)
	reply(t, s.Start("--profile", "exact"))
	first := waitState(t, s, Ready)
	if first.AutoRestarted {
		t.Fatal("the user's own start is not an automatic restart")
	}
	script(t, dir, "75 once")
	failed := waitFor(t, s, "the 75 exit", 6*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	if failed.LastError == nil || failed.LastError.Code != "device_lost" {
		t.Fatalf("exit 75 must be device_lost: %+v", failed.LastError)
	}
	second := waitFor(t, s, "the restart", 6*time.Second, func(v Snapshot) bool { return v.State == Ready && v.Generation > failed.Generation })
	if !second.AutoRestarted {
		t.Fatal("the restart must be visible: AutoRestarted is false")
	}
	if second.Instance.ID == first.Instance.ID {
		t.Fatal("the restart reused the old instance")
	}
	launches := serveLaunches(t, dir)
	if len(launches) != 2 || launches[0] != launches[1] || !strings.Contains(launches[0], "--profile exact") {
		t.Fatalf("the restart must use the same setup: %q", launches)
	}
	// The window reads the fact from the View, not from the supervisor.
	if !(&ViewBuilder{}).Build(ViewInput{Snapshot: second}).AutoRestarted {
		t.Fatal("the View does not carry the automatic restart")
	}
	// Whatever the user does next ends the notice.
	reply(t, s.Stop())
	if v := waitState(t, s, Stopped); v.AutoRestarted {
		t.Fatal("a stopped server still claims an automatic restart")
	}
}

// The lifecycle stream can close before Windows reports the exit: a busy
// machine, or a large process tearing down. The exit status still decides.
func TestExit75AfterTheStreamClosesIsStillDeviceLost(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 300*time.Millisecond, time.Minute)
	reply(t, s.Start())
	waitState(t, s, Ready)
	script(t, dir, "75 once late")
	failed := waitFor(t, s, "the late 75 exit", 6*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	if failed.LastError == nil || failed.LastError.Code != "device_lost" {
		t.Fatalf("exit 75 after the stream closed must be device_lost: %+v", failed.LastError)
	}
	second := waitFor(t, s, "the restart", 6*time.Second, func(v Snapshot) bool { return v.State == Ready && v.Generation > failed.Generation })
	if !second.AutoRestarted {
		t.Fatal("the restart must be visible: AutoRestarted is false")
	}
}

func TestExit75TwiceInsideTheWindowStopsWithTheError(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 200*time.Millisecond, time.Minute)
	script(t, dir, "75")
	reply(t, s.Start())
	// The first start exits 75 at once, is restarted once, and exits 75 again.
	waitFor(t, s, "the second launch", 8*time.Second, func(v Snapshot) bool { return len(serveLaunches(t, dir)) == 2 })
	v := waitFor(t, s, "the stop after the second 75", 8*time.Second, func(v Snapshot) bool {
		return v.State == Failed && v.Instance == nil && len(serveLaunches(t, dir)) == 2
	})
	time.Sleep(900 * time.Millisecond) // several delays: no loop
	v = s.Snapshot()
	if v.State != Failed || v.LastError == nil || v.LastError.Code != "device_lost" || v.AutoRestarted || !v.LastError.Again {
		t.Fatalf("the second 75 must stop with the error, marked as again: %+v %+v", v.State, v.LastError)
	}
	if n := len(serveLaunches(t, dir)); n != 2 {
		t.Fatalf("restart loop: %d launches", n)
	}
	// The user can start again by hand; that is not limited.
	os.Remove(filepath.Join(dir, "exit-serve"))
	reply(t, s.Start())
	waitState(t, s, Ready)
}

func TestStopDuringTheDelayCancelsTheRestart(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 1500*time.Millisecond, time.Minute)
	reply(t, s.Start())
	waitState(t, s, Ready)
	script(t, dir, "75 once")
	waitFor(t, s, "the 75 exit", 6*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	<-s.Stop() // the user acts during the delay; the answer does not matter
	time.Sleep(2500 * time.Millisecond)
	if v := s.Snapshot(); v.State == Ready || v.State == Starting || v.AutoRestarted {
		t.Fatalf("Stop during the delay must not restart: %s", v.State)
	}
	if n := len(serveLaunches(t, dir)); n != 1 {
		t.Fatalf("%d launches", n)
	}
}

func TestExit75WhileTheUserStopsDoesNotRestart(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 200*time.Millisecond, time.Minute)
	reply(t, s.Start())
	waitState(t, s, Ready)
	script(t, dir, "75 on-stop")
	<-s.Stop()
	waitFor(t, s, "the end of the stop", 6*time.Second, func(v Snapshot) bool { return v.Instance == nil && (v.State == Failed || v.State == Stopped) })
	time.Sleep(900 * time.Millisecond)
	if v := s.Snapshot(); v.State == Ready || v.State == Starting || v.AutoRestarted {
		t.Fatalf("a restart after the user pressed Stop: %s", v.State)
	}
	if n := len(serveLaunches(t, dir)); n != 1 {
		t.Fatalf("%d launches", n)
	}
}

func TestOtherExitStatusesAreUnchanged(t *testing.T) {
	for _, status := range []string{"3", "1", "74", "76"} {
		t.Run(status, func(t *testing.T) {
			s, dir := autoRestartSupervisor(t, 200*time.Millisecond, time.Minute)
			reply(t, s.Start())
			waitState(t, s, Ready)
			script(t, dir, status+" once")
			v := waitFor(t, s, "the exit", 6*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
			if v.LastError == nil || v.LastError.Code != "server_exited" {
				t.Fatalf("exit %s must stay server_exited: %+v", status, v.LastError)
			}
			time.Sleep(900 * time.Millisecond)
			if v := s.Snapshot(); v.State != Failed || v.AutoRestarted {
				t.Fatalf("exit %s was restarted: %s", status, v.State)
			}
			if n := len(serveLaunches(t, dir)); n != 1 {
				t.Fatalf("%d launches", n)
			}
		})
	}
}

func TestA75AfterTheWindowRestartsAgain(t *testing.T) {
	s, dir := autoRestartSupervisor(t, 100*time.Millisecond, 600*time.Millisecond)
	reply(t, s.Start())
	waitState(t, s, Ready)
	script(t, dir, "75 once")
	waitFor(t, s, "the first restart", 6*time.Second, func(v Snapshot) bool { return v.State == Ready && v.AutoRestarted })
	time.Sleep(900 * time.Millisecond) // past the window
	script(t, dir, "75 once")
	waitFor(t, s, "the second restart", 8*time.Second, func(v Snapshot) bool { return len(serveLaunches(t, dir)) == 3 && v.State == Ready && v.AutoRestarted })
}
