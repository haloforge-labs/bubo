//go:build windows

package updater

// The update helper runs from a copy under the transaction folder, because a
// running image cannot be deleted. After a finished update nothing removed that
// copy (about 7.7 MB of the old version's bytes). The GUI removes it once the
// helper has ended.
//
// Ways this could fail, written before the code:
//
//	H1  the copy stays after a finished update (LastResult, ClearLastResult).
//	H2  the copy of a transaction that is not finished is removed: the helper still
//	    runs, has no result yet, or the journal says Recover needs the copy.
//	H3  a helper whose identity is unknown is treated as ended.
//	H4  a file in use makes LastResult, ClearLastResult or New fail.
//	H5  a copy that could not be removed is never retried.

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

// seedHelperCopy makes a stand-in for the copied helper in the transaction folder.
func seedHelperCopy(t *testing.T, root string) string {
	t.Helper()
	dir := filepath.Join(root, "helper-test")
	if err := ensureDir(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bubo-update.exe"), []byte("old helper bytes"), 0700); err != nil {
		t.Fatal(err)
	}
	return dir
}

func seedResult(t *testing.T, s *Staged) {
	t.Helper()
	if err := writeJSON(filepath.Join(s.root, ResultFileName), InstallResult{Status: "installed", Version: s.Version}); err != nil {
		t.Fatal(err)
	}
}

func TestFinishedUpdateRemovesTheHelperCopy(t *testing.T) { // H1
	f, s := integrationPackage(t, "", "good")
	c := f.client
	gui, server := processFixture(t, 150*time.Millisecond), processFixture(t, 200*time.Millisecond)
	c.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{gui.identity.PID, server.identity.PID}, nil }
	handoff, failure := c.Install(s)
	if failure != nil {
		t.Fatal(failure)
	}
	helperDir := filepath.Dir(handoff.HelperExe)
	if !exists(handoff.HelperExe) {
		t.Fatal("the helper copy is missing while the helper runs")
	}
	waitHelper(t, handoff.HelperPID)
	r, failure := c.LastResult()
	if failure != nil || r == nil || r.Status != "installed" {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
	if exists(helperDir) {
		t.Fatal("the helper copy remains after a finished update")
	}
	if !exists(filepath.Join(s.root, ResultFileName)) {
		t.Fatal("the result file must stay until the user dismisses it")
	}
	if failure := c.ClearLastResult(); failure != nil {
		t.Fatal(failure)
	}
	if exists(c.activePath()) {
		t.Fatal("active.json remains after the dismissal")
	}
}

func TestHelperCopyIsRemovedByClearLastResultAndByTheNextStart(t *testing.T) { // H1 H5
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	seedResult(t, s)
	dir := seedHelperCopy(t, s.root)
	if failure := c.ClearLastResult(); failure != nil {
		t.Fatal(failure)
	}
	if exists(dir) {
		t.Fatal("ClearLastResult left the helper copy")
	}
	// After the dismissal active.json is gone. A copy that was in use then is
	// removed by the next start, which has no saved handoff to look at.
	dir = seedHelperCopy(t, s.root)
	if _, failure := New(c.cfg); failure != nil {
		t.Fatal(failure)
	}
	if exists(dir) {
		t.Fatal("a new client left the helper copy of a dismissed update")
	}
}

func TestHelperCopyOfAnUnfinishedUpdateIsKept(t *testing.T) { // H2 H3
	// Each case starts its own process after the package is built, so a "running"
	// helper is still running when the cleanup looks at it.
	running := func(t *testing.T) processIdentity { return liveProcess(t) }
	ended := func(t *testing.T) processIdentity { return exitedProcess(t) }
	unknown := func(t *testing.T) processIdentity { return processIdentity{} }
	pidOnly := func(t *testing.T) processIdentity { return processIdentity{PID: exitedProcess(t).PID} }
	createdOnly := func(t *testing.T) processIdentity { return processIdentity{Created: exitedProcess(t).Created} }
	for name, tc := range map[string]struct {
		id      func(*testing.T) processIdentity
		result  bool
		journal bool
	}{
		"helper runs, no result":    {id: running},
		"helper runs, has result":   {id: running, result: true},
		"helper ended, journal":     {id: ended, result: true, journal: true},
		"identity unknown, result":  {id: unknown, result: true},
		"pid only, result":          {id: pidOnly, result: true},
		"created only, result":      {id: createdOnly, result: true},
		"helper runs, journal only": {id: running, journal: true},
	} {
		t.Run(name, func(t *testing.T) {
			f, s := integrationPackage(t, "", "good")
			c := f.client
			seedHandoff(t, c, s, tc.id(t))
			if tc.result {
				seedResult(t, s)
			}
			if tc.journal {
				if err := writeJSON(filepath.Join(s.root, journalName), journal{Phase: "moving_old", Version: s.Version, Seal: s.seal}); err != nil {
					t.Fatal(err)
				}
			}
			dir := seedHelperCopy(t, s.root)
			c.LastResult()
			if _, failure := New(c.cfg); failure != nil {
				t.Fatal(failure)
			}
			if !exists(filepath.Join(dir, "bubo-update.exe")) {
				t.Fatal("the helper copy of an unfinished update was removed")
			}
		})
	}
}

func TestHelperCopyIsKeptWhileAHelperHoldsTheInstallLock(t *testing.T) { // H2
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedResult(t, s) // a dismissed update: no active.json, only the install lock shows a helper
	dir := seedHelperCopy(t, s.root)
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
	if _, failure := New(c.cfg); failure != nil {
		t.Error(failure)
	}
	kept := exists(filepath.Join(dir, "bubo-update.exe"))
	close(release)
	<-done
	if !kept {
		t.Fatal("the helper copy was removed while a helper holds the install lock")
	}
	if _, failure := New(c.cfg); failure != nil {
		t.Fatal(failure)
	}
	if exists(dir) {
		t.Fatal("the helper copy was not removed after the helper ended")
	}
}

func TestHelperCopyInUseDoesNotFailAndIsRetried(t *testing.T) { // H4 H5
	f, s := integrationPackage(t, "", "good")
	c := f.client
	seedHandoff(t, c, s, exitedProcess(t))
	seedResult(t, s)
	dir := seedHelperCopy(t, s.root)
	name, err := windows.UTF16PtrFromString(filepath.Join(dir, "bubo-update.exe"))
	if err != nil {
		t.Fatal(err)
	}
	// No FILE_SHARE_DELETE: Windows refuses to delete the file, as for a running image.
	h, err := windows.CreateFile(name, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	open := true
	defer func() {
		if open {
			windows.CloseHandle(h)
		}
	}()
	r, failure := c.LastResult()
	if failure != nil || r == nil || r.Status != "installed" {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
	if failure := c.ClearLastResult(); failure != nil {
		t.Fatalf("ClearLastResult failed because a helper file is in use: %v", failure)
	}
	if _, failure := New(c.cfg); failure != nil {
		t.Fatalf("New failed because a helper file is in use: %v", failure)
	}
	if !exists(filepath.Join(dir, "bubo-update.exe")) {
		t.Fatal("the file in use vanished")
	}
	windows.CloseHandle(h)
	open = false
	if _, failure := New(c.cfg); failure != nil {
		t.Fatal(failure)
	}
	if exists(dir) {
		t.Fatal("the helper copy was not retried after the file was released")
	}
}
