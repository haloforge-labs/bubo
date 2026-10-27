//go:build windows

package supervisor

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"bubo/gui/updater"
)

// A failed start of the updater, or a failed check, is not a failed update:
// the view says so, and the header shows no pill. Only an update the user
// started (download or install) leaves check_failed false.
func TestUpdateFailureKindSeparatesChecksFromUserStartedUpdates(t *testing.T) {
	t.Run("updater start failure", func(t *testing.T) {
		u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot})
		u.factory = func(updater.Config) (updateClient, *updater.Failure) {
			return nil, &updater.Failure{Code: updater.CodeURL, Message: "manifest must use latest or an exact stable release tag"}
		}
		defer u.Close()
		u.Start(readyWindow())
		v := waitUpdate(t, u, func(v Update) bool { return v.State == UpdateFailed })
		if !v.CheckFailed || v.Error == nil || v.Error.Code != string(updater.CodeURL) {
			t.Fatalf("start failure must be a check failure: %+v", v)
		}
		// Every can't-check state offers Check, and Check builds the updater again.
		u.wg.Wait()
		if a := lifecycleActions(ViewInput{Snapshot: saneUpdateSnapshot(), Update: v}).Check; !a.Allowed {
			t.Fatalf("a failed start must still offer Check: %+v", a)
		}
		client := &stubUpdateClient{check: updater.CheckResult{Status: updater.UpToDate}}
		u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
		if r := u.CheckUpdate(); r.Error != nil {
			t.Fatal(r.Error)
		}
		if v := u.State(); v.State != UpdateCurrent || !v.Ready || v.CheckFailed {
			t.Fatalf("the retry must leave a working updater: %+v", v)
		}
	})
	t.Run("automatic check failure", func(t *testing.T) {
		client := &stubUpdateClient{check: updater.CheckResult{Failure: &updater.Failure{Code: updater.CodeNetwork, Message: "offline"}}}
		u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot})
		u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
		defer u.Close()
		u.Start(readyWindow())
		v := waitUpdate(t, u, func(v Update) bool { return v.State == UpdateFailed })
		if !v.CheckFailed {
			t.Fatalf("a failed automatic check must be a check failure: %+v", v)
		}
		// A later good check clears the mark.
		client.check = updater.CheckResult{Status: updater.UpToDate}
		if r := u.CheckUpdate(); r.Error != nil {
			t.Fatal(r.Error)
		}
		if v := u.State(); v.CheckFailed || v.State != UpdateCurrent {
			t.Fatalf("a good check must clear the failure: %+v", v)
		}
	})
	t.Run("user started install failure", func(t *testing.T) {
		client := &stubUpdateClient{
			check:          updater.CheckResult{Status: updater.Available, Available: &updater.Release{Version: "0.1.2", Size: 10}},
			installFailure: &updater.Failure{Code: updater.CodeIO, Message: "helper blocked"},
		}
		u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot, Quiesce: func(context.Context) ([]uint32, func(), error) { return nil, func() {}, nil }})
		u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
		defer u.Close()
		u.Start(readyWindow())
		u.wg.Wait()
		if r := u.DownloadUpdate(); r.Error != nil {
			t.Fatal(r.Error)
		}
		if r := u.InstallUpdate(); r.Error == nil {
			t.Fatal("install failure reported success")
		}
		if v := u.State(); v.State != UpdateFailed || v.CheckFailed {
			t.Fatalf("a failed install is an update failure: %+v", v)
		}
	})
}

// The rollback state names the version kept in .previous, when Bubo can read it.
func TestUpdateRollbackNamesPreviousVersion(t *testing.T) {
	root := filepath.Join(t.TempDir(), "Bubo")
	previous := root + ".previous"
	if err := os.MkdirAll(previous, 0700); err != nil {
		t.Fatal(err)
	}
	manifest := `{"product":{"product_version":"0.1.0"}}`
	if err := os.WriteFile(filepath.Join(previous, "BUILD-MANIFEST.json"), []byte(manifest), 0600); err != nil {
		t.Fatal(err)
	}
	if got := previousVersion(root); got != "0.1.0" {
		t.Fatalf("previous version = %q, want 0.1.0", got)
	}
	if got := previousVersion(filepath.Join(t.TempDir(), "Missing")); got != "" {
		t.Fatalf("a missing .previous folder must give no version, got %q", got)
	}
	if got := previousVersion(""); got != "" {
		t.Fatalf("no install folder must give no version, got %q", got)
	}
}
