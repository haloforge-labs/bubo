//go:build windows

package updater

// E21w W1. A power cut, a log-off or an antivirus kill can end the update helper
// while it still waits for the GUI and the server to exit. Then active.json
// exists, and the helper wrote no journal and no result. Nothing moved, but
// pendingHandoff answered "busy" for ever, LastResult said nothing, and every
// later update failed with "Another Bubo action is still running."
//
// Ways this could fail, written before the code:
//
//	F1  the helper is gone, no journal, no result: the next update is still
//	    refused as busy (preflight and Download).
//	F2  the same state: LastResult stays silent, so the first launch tells the
//	    user nothing.
//	F3  the report repeats on every launch: it must be one result that the user
//	    dismisses once (ClearLastResult removes it and active.json).
//	F4  a LIVE helper is declared dead (it is slow, or has not yet taken its
//	    lock): the staged folder would be reused under a running helper.
//	F5  the helper identity is unknown (active.json from before the identity was
//	    recorded, or the GUI died between the launch and the write): it is
//	    declared dead on a guess.
//	F6  the recorded PID now belongs to another process (PID reuse): that process
//	    is mistaken for the helper, so the handoff looks alive for ever.
//	F7  the helper process is gone but another helper holds the install lock.
//	F8  a journal exists: the rollback_pending path changes (it must not).
//	F9  a result exists: it is overwritten by the "stopped" result.
//	F10 settling the dead handoff touches the install folder or deletes the
//	    staging folder (it must change nothing but the result file).
//	F11 launchHelper does not record the helper's identity, so none of the above
//	    can work against a real launch.
//	F12 the next Install is refused after a dead handoff was reported.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// exitedProcess returns the identity of a process that has already exited.
func exitedProcess(t *testing.T) processIdentity {
	t.Helper()
	p := processFixture(t, 10*time.Millisecond)
	id := p.identity
	if id.Created == 0 {
		t.Fatal("the fixture process has no creation time")
	}
	waitHelper(t, int(id.PID))
	return id
}

// liveProcess returns the identity of a process that is still running.
func liveProcess(t *testing.T) processIdentity {
	t.Helper()
	return processFixture(t, 2*time.Second).identity
}

// seedHandoff writes the active.json the GUI leaves when it starts a helper that
// then writes nothing.
func seedHandoff(t *testing.T, c *Client, s *Staged, id processIdentity) activeTransaction {
	t.Helper()
	a := activeTransaction{Root: s.root, Version: s.Version, Seal: s.seal,
		Helper: filepath.Join(s.root, "helper-test", "bubo-update.exe"), HelperHash: strings.Repeat("a", 64),
		HelperPID: id.PID, HelperCreated: id.Created}
	if err := writeJSON(c.activePath(), a); err != nil {
		t.Fatal(err)
	}
	return a
}

func assertStopped(t *testing.T, r *InstallResult, f *Failure) {
	t.Helper()
	if f != nil || r == nil {
		t.Fatalf("result=%+v failure=%v", r, f)
	}
	if r.Status != "failed" || r.Failure == nil || r.Failure.Code != CodeUpdateStopped {
		t.Fatalf("want a failed result with code %s, got %+v", CodeUpdateStopped, r)
	}
	if !strings.Contains(r.Failure.Message, "before it changed anything") {
		t.Fatalf("message must say nothing changed: %q", r.Failure.Message)
	}
}

func TestDeadHelperWithoutJournalOrResultIsReportedOnceAndNextUpdateWorks(t *testing.T) { // F1 F2 F3 F10 F12
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	before, err := treeSeal(c.cfg.InstallDir)
	if err != nil {
		t.Fatal(err)
	}

	r, failure := c.LastResult()
	assertStopped(t, r, failure)
	// The same answer on every call until the user dismisses it.
	r2, failure2 := c.LastResult()
	assertStopped(t, r2, failure2)

	// F12: the next update is not refused.
	if failure := c.Preflight(s); failure != nil {
		t.Fatalf("the next update is refused after a stopped handoff: %v", failure)
	}

	// F10: only the result file was written.
	after, err := treeSeal(c.cfg.InstallDir)
	if err != nil || after != before {
		t.Fatalf("install folder changed: %s %s %v", before, after, err)
	}
	if now, err := treeSeal(s.Directory); err != nil || now != s.seal {
		t.Fatalf("staged package changed: %s %v", now, err)
	}
	if exists(filepath.Join(s.root, journalName)) || exists(c.cfg.InstallDir+".previous") {
		t.Fatal("a journal or .previous appeared")
	}

	// F3: one dismissal ends the report.
	if failure := c.ClearLastResult(); failure != nil {
		t.Fatal(failure)
	}
	if r, failure := c.LastResult(); r != nil || failure != nil {
		t.Fatalf("the result came back after it was dismissed: %+v %v", r, failure)
	}
	if exists(c.activePath()) {
		t.Fatal("active.json remains after the dismissal")
	}
}

func TestDeadHelperIsNotBusyForPreflightBeforeAnyLastResultCall(t *testing.T) { // F1
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	if failure := c.Preflight(s); failure != nil {
		t.Fatalf("preflight: %v", failure)
	}
	// The report is still there for the next launch.
	r, failure := c.LastResult()
	assertStopped(t, r, failure)
}

func TestDeadHelperDoesNotBlockDownload(t *testing.T) { // F1
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	if _, failure := c.Download(t.Context(), nil); failure != nil {
		t.Fatalf("download after a stopped handoff: %v", failure)
	}
}

func TestLiveHelperIsNeverDeclaredDead(t *testing.T) { // F4
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, liveProcess(t))
	if r, failure := c.LastResult(); r != nil || failure != nil {
		t.Fatalf("a live helper was reported: %+v %v", r, failure)
	}
	if failure := c.Preflight(s); failure == nil || failure.Code != CodeBusy {
		t.Fatalf("a live helper must keep the handoff busy: %v", failure)
	}
	if exists(filepath.Join(s.root, ResultFileName)) {
		t.Fatal("a result was written under a live helper")
	}
}

func TestUnknownHelperIdentityIsNeverDeclaredDead(t *testing.T) { // F5
	for name, id := range map[string]processIdentity{
		"none":         {},
		"pid only":     {PID: exitedProcess(t).PID},
		"created only": {Created: exitedProcess(t).Created},
	} {
		t.Run(name, func(t *testing.T) {
			f, s := integrationPackage(t, "", "good")
			c := f.client
			seedHandoff(t, c, s, id)
			if r, failure := c.LastResult(); r != nil || failure != nil {
				t.Fatalf("unknown identity reported: %+v %v", r, failure)
			}
			if failure := c.Preflight(s); failure == nil || failure.Code != CodeBusy {
				t.Fatalf("unknown identity must stay busy: %v", failure)
			}
			if exists(filepath.Join(s.root, ResultFileName)) {
				t.Fatal("a result was written on a guess")
			}
		})
	}
}

func TestReusedPIDIsNotTheHelper(t *testing.T) { // F6
	f, s := integrationPackage(t, "", "good")
	c := f.client
	live := liveProcess(t)
	// The PID is alive, but it was created at another time: another process.
	seedHandoff(t, c, s, processIdentity{PID: live.PID, Created: live.Created - 12_345_678})
	r, failure := c.LastResult()
	assertStopped(t, r, failure)
}

func TestDeadHelperWhileAnotherHelperHoldsTheInstallLockIsNotDeclaredDead(t *testing.T) { // F7
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	held := make(chan struct{})
	release := make(chan struct{})
	done := make(chan struct{})
	go func() { // the mutex belongs to one OS thread
		defer close(done)
		unlock, err := lockInstall(c.cfg.InstallDir)
		if err != nil {
			t.Error(err)
			close(held)
			return
		}
		close(held)
		<-release
		unlock()
	}()
	<-held
	r, failure := c.LastResult()
	if r != nil || failure != nil {
		t.Errorf("reported while another helper holds the lock: %+v %v", r, failure)
	}
	if exists(filepath.Join(s.root, ResultFileName)) {
		t.Error("a result was written while another helper holds the lock")
	}
	close(release)
	<-done
}

func TestDeadHelperWithJournalStaysRollbackPending(t *testing.T) { // F8
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	if err := writeJSON(filepath.Join(s.root, journalName), journal{Phase: "moving_old", Version: s.Version, Seal: s.seal}); err != nil {
		t.Fatal(err)
	}
	r, failure := c.LastResult()
	if failure != nil || r == nil || r.Status != "rollback_pending" || r.Failure == nil || r.Failure.Code != CodeRollbackPending {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
	if failure := c.Preflight(s); failure == nil || failure.Code != CodeRollbackPending {
		t.Fatalf("preflight: %v", failure)
	}
	if exists(filepath.Join(s.root, ResultFileName)) {
		t.Fatal("a result was written next to a journal")
	}
}

func TestDeadHelperWithResultKeepsThatResult(t *testing.T) { // F9
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	want := InstallResult{Status: "rolled_back", Version: s.Version, Failure: fail(CodeHealth, "the new version did not start")}
	if err := writeJSON(filepath.Join(s.root, ResultFileName), want); err != nil {
		t.Fatal(err)
	}
	r, failure := c.LastResult()
	if failure != nil || r == nil || r.Status != "rolled_back" || r.Failure == nil || r.Failure.Code != CodeHealth {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
}

// F11 and the whole chain with a real launch. The stand-in helper is the fixture
// program: it starts, finds arguments it does not know and exits. It is not the
// update helper and it moves nothing.
func TestLaunchHelperRecordsItsIdentityAndAnExitedLaunchIsReported(t *testing.T) {
	f, s := integrationPackage(t, "", "good")
	c := f.client
	helperDir := filepath.Join(s.root, "helper-standin")
	if err := ensureDir(helperDir); err != nil {
		t.Fatal(err)
	}
	standin := filepath.Join(helperDir, "bubo-update.exe")
	if err := os.WriteFile(standin, fixtureGUI, 0700); err != nil {
		t.Fatal(err)
	}
	a := activeTransaction{Root: s.root, Version: s.Version, Seal: s.seal, Helper: standin, HelperHash: strings.Repeat("a", 64)}
	if err := writeJSON(c.activePath(), a); err != nil {
		t.Fatal(err)
	}
	handoff, failure := c.launchHelper(a, nil, false)
	if failure != nil {
		t.Fatal(failure)
	}
	saved, failure := c.readActive()
	if failure != nil {
		t.Fatal(failure)
	}
	if int(saved.HelperPID) != handoff.HelperPID || saved.HelperCreated == 0 {
		t.Fatalf("launchHelper must record the helper identity: handoff pid %d, saved %d/%d", handoff.HelperPID, saved.HelperPID, saved.HelperCreated)
	}
	if saved.Root != a.Root || saved.Seal != a.Seal || saved.Helper != a.Helper {
		t.Fatalf("the saved handoff changed: %+v", saved)
	}
	waitHelper(t, handoff.HelperPID)
	r, failure := c.LastResult()
	assertStopped(t, r, failure)
	if failure := c.Preflight(s); failure != nil {
		t.Fatalf("the next update is refused: %v", failure)
	}
}
