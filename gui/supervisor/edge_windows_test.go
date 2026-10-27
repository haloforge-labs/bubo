//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestPrepareCancel(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Prepare())
	deadline := time.Now().Add(2 * time.Second)
	for s.Snapshot().Approval == nil && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	v := s.Snapshot()
	if len(v.PlanParts) == 0 || v.Preflight == nil {
		t.Fatal("prepare plan was not exposed", v)
	}
	reply(t, s.Stop())
	v = waitState(t, s, Stopped)
	if v.Forced {
		t.Fatal("cancel was forced")
	}
}
func TestKeepDismissAndPersistError(t *testing.T) {
	c := configFor(t, "normal")
	c.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	c.ProductVersion = "fixture-current"
	settings := DefaultSettings()
	settings.lastErrorVersion = c.ProductVersion
	settings.LastError = &Failure{Code: "device_lost", Source: "cli", Time: time.Now().UTC(), Remedy: "synthetic-secret"}
	if err := SaveSettings(c.SettingsPath, settings); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { reply(t, s.Exit()) }()
	if s.Snapshot().LastError == nil {
		t.Fatal("startup lost same-version error")
	}
	reply(t, s.Start())
	v := waitState(t, s, Ready)
	if v.LastError != nil {
		t.Fatal("successful retry retained prior error")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.DismissError())
	if s.Snapshot().LastError != nil {
		t.Fatal("Dismiss did not clear error")
	}
	saved, warning := LoadSettings(c.SettingsPath)
	if warning != "" || saved.LastError != nil {
		t.Fatal(saved, warning)
	}
}
func TestSettingsWarningStaysVisible(t *testing.T) {
	c := configFor(t, "normal")
	c.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	os.WriteFile(c.SettingsPath, []byte("invalid"), 0600)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	if s.Snapshot().SettingsWarning == "" {
		t.Fatal("damaged settings warning lost")
	}
	reply(t, s.Exit())
}
func TestExternalNoForceOnDeadline(t *testing.T) {
	c := configFor(t, "hang-stop")
	_, i := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	reply(t, s.Manage())
	r := <-s.Stop()
	if r.Err == nil {
		t.Fatal("external deadline reported clean")
	}
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal("external was terminated", err)
	}
	if s.Snapshot().Forced {
		t.Fatal("external marked forced")
	}
	reply(t, s.Exit())
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal("external terminated on Exit", err)
	}
}
func TestExternalUnsupportedManagement(t *testing.T) {
	c := configFor(t, "unsupported")
	_, i := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	v := waitState(t, s, AttachedExternal)
	if v.ManagementSupported || v.ManagementMessage == "" {
		t.Fatal(v)
	}
	if r := <-s.Manage(); r.Err == nil {
		t.Fatal("older management accepted")
	}
	reply(t, s.Exit())
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal(err)
	}
}
func TestManagedExternalRestart(t *testing.T) {
	c := configFor(t, "normal")
	_, i := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	reply(t, s.Manage())
	reply(t, s.Restart())
	v := waitState(t, s, Ready)
	if v.Instance.ID == i.ID {
		t.Fatal("external restart reused identity")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.Exit())
}
func TestKillOnJobClose(t *testing.T) {
	c := configFor(t, "spawn")
	p, err := launchOwned(c.Exe, []string{"serve", "--host", "127.0.0.1", "--port", "19091"}, c.DrainTimeout, func([]byte, bool) {}, func([]byte, bool) {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	var pids []int
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "pids.txt")); err == nil {
			pids = readPIDs(t)
			if len(pids) == 3 {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(pids) != 3 {
		p.Force()
		<-p.Done
		t.Fatal("missing descendant", pids)
	}
	p.mu.Lock()
	err = windows.CloseHandle(p.job)
	p.job = 0
	p.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.Done:
	case <-time.After(3 * time.Second):
		t.Fatal("kill-on-close did not exit")
	}
	assertNoOrphans(t, pids)
}
func TestStoppedCommandsReplyAfterExit(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Exit())
	for n := 0; n < 100; n++ {
		select {
		case r := <-s.Start():
			if r.Err == nil {
				t.Fatal("start after Exit accepted")
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("post-Exit action blocked")
		}
	}
}
func TestSnapshotIsCopy(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	v := waitState(t, s, Ready)
	v.Instance.ID = "wrong"
	v.Instance.Effective["profile"] = json.RawMessage(`"changed"`)
	actual := s.Snapshot()
	if actual.Instance.ID == "wrong" || string(actual.Instance.Effective["profile"]) == `"changed"` {
		t.Fatal("snapshot mutated reducer")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}
func TestArgumentOwnership(t *testing.T) {
	s := supervisorFor(t, "normal")
	for _, args := range [][]string{{"--host", "elsewhere"}, {"--port=19092"}, {"--yes"}, {"--instance-id", "other"}} {
		r := <-s.Start(args...)
		if r.Err == nil {
			t.Fatal("lifecycle override accepted", args)
		}
	}
}

func TestUnicodeExecutablePath(t *testing.T) {
	c := configFor(t, "normal")
	dir := filepath.Join(t.TempDir(), "space é水")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "bubo.exe")
	b, err := os.ReadFile(fakeExe)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(exe, b, 0700); err != nil {
		t.Fatal(err)
	}
	c.Exe = exe
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
		reply(t, s.Exit())
	})
	// A newly copied executable can still be in its first-run scan. Observe
	// bootstrap before timing Start. Keep the original combined Start/Ready
	// budget (four plus six seconds), with neither existing bound increased.
	stateWait := time.Now()
	startupDeadline := stateWait.Add(4*time.Second + 6*time.Second)
	waitFor(t, s, "Unicode executable discovery", 6*time.Second, func(v Snapshot) bool { return v.Initialized })
	t.Logf("Unicode discovery completed in %s", time.Since(stateWait))
	reply(t, s.Start())
	waitFor(t, s, "Unicode executable Ready", min(6*time.Second, time.Until(startupDeadline)), func(v Snapshot) bool { return v.State == Ready })
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.Exit())
}

func TestTransitionListIsCopy(t *testing.T) {
	list := Transitions()
	first := list[0]
	list[0].To = Failed
	if Transitions()[0] != first {
		t.Fatal("caller changed reducer transitions")
	}
}

func TestDiagnoseUsesOwnedContractHelper(t *testing.T) {
	s := supervisorFor(t, "normal")
	select {
	case result := <-s.Diagnose():
		if result.Err != nil || result.Event.Value.(*Diagnose).Status != "PASS" {
			t.Fatal(result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("diagnose blocked")
	}
	if s.Snapshot().State != Stopped {
		t.Fatal("diagnose changed lifecycle")
	}
}
