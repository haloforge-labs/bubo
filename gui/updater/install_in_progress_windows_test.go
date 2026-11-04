//go:build windows

package updater

import "testing"

// A running helper holds the install lock; a GUI started by hand during the
// update must see that and wait. The lock belongs to an OS thread, so the helper
// side runs on its own goroutine.
func TestInstallInProgressFollowsTheInstallLock(t *testing.T) {
	dir := t.TempDir()
	if InstallInProgress(dir) {
		t.Fatal("an idle installation reported an update in progress")
	}
	held, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		unlock, err := lockInstall(dir)
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
	if !InstallInProgress(dir) {
		t.Fatal("a held install lock was not reported as an update in progress")
	}
	if InstallInProgress(t.TempDir()) {
		t.Fatal("another installation's lock must not count")
	}
	close(release)
	<-done
	if InstallInProgress(dir) {
		t.Fatal("the lock was released but the update still counts as in progress")
	}
}
