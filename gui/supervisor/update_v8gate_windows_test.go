//go:build windows && v8gate

package supervisor

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"bubo/gui/updater"
)

func v8RollbackApp(t *testing.T) (*AppService, *stubUpdateClient, *bool) {
	t.Helper()
	client := &stubUpdateClient{result: &updater.InstallResult{Status: "rollback_pending", Version: "0.1.2",
		Failure: &updater.Failure{Code: updater.CodeRollbackPending, Message: "candidate image still running"}}}
	exited := new(bool)
	u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot,
		Quiesce: func(context.Context) ([]uint32, func(), error) { return []uint32{42}, func() {}, nil },
		Exit:    func() { *exited = true }})
	u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
	u.Start(readyWindow())
	u.wg.Wait()
	if u.State().State != UpdateRollbackPending {
		t.Fatalf("fixture did not reach rollback_pending: %+v", u.State())
	}
	a := newTestApp(&fakeBackend{view: saneUpdateSnapshot()}, DefaultSettings(), nil, "")
	a.updates = u
	t.Cleanup(u.Close)
	return a, client, exited
}

func TestV8F16InstallPreconditionBeforeServerQuiesce(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	copyRoot, err := os.MkdirTemp(root, "supervisor-copy-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		rel, err := filepath.Rel(root, copyRoot)
		if err != nil || rel == "." || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
			t.Errorf("refusing cleanup outside A4 evidence root: %q", copyRoot)
			return
		}
		if err := os.RemoveAll(copyRoot); err != nil {
			t.Errorf("remove own copy: %v", err)
		}
	})
	install := filepath.Join(copyRoot, "Bubo")
	if err := os.MkdirAll(install, 0700); err != nil {
		t.Fatal(err)
	}
	asset, archive := updatePackage(t)
	var manifest updater.Manifest
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".zip") {
			_, _ = w.Write(archive)
			return
		}
		_ = json.NewEncoder(w).Encode(manifest)
	}))
	defer server.Close()
	manifest = updateFeed(server.URL, asset)
	pool := x509.NewCertPool()
	pool.AddCert(server.Certificate())
	cfg := updater.Config{InstalledVersion: "0.1.1", InstallDir: install,
		ManifestURL:  server.URL + updateReleasePath + "latest/download/bubo-release.json",
		AllowedHosts: []string{"127.0.0.1"}, RootCAs: pool}
	quiesced, serving := 0, true
	u := NewUpdateService(UpdateServiceConfig{Updater: cfg, Snapshot: saneUpdateSnapshot,
		Enabled: func() bool { return false },
		Quiesce: func(context.Context) ([]uint32, func(), error) {
			quiesced++
			serving = false
			return []uint32{42}, func() {}, nil
		}, Exit: func() {}})
	defer u.Close()
	u.Start(readyWindow())
	u.wg.Wait()
	if r := u.CheckUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	if r := u.DownloadUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	if u.State().State != UpdateReady || u.staged == nil {
		t.Fatalf("fixture did not stage update: %+v", u.State())
	}
	// Damage a real staged package entry. The updater's seal check is an
	// existing precondition, so no invented fake Install failure is involved.
	damaged := filepath.Join(u.staged.Directory, "bin", "bubo.exe")
	if err := os.WriteFile(damaged, []byte("damaged after verification"), 0600); err != nil {
		t.Fatal(err)
	}
	result := u.InstallUpdate()
	state := u.State()
	view := new(ViewBuilder).Build(ViewInput{Snapshot: saneUpdateSnapshot(), Saved: DefaultSettings(), Update: state})
	field := reflect.ValueOf(view.Actions).FieldByName("Install")
	installAllowed := field.IsValid() && field.Interface().(ActionAvailability).Allowed
	if result.Error == nil || result.Error.Code != string(updater.CodeHash) || quiesced != 0 || !serving || state.State != UpdateReady || !installAllowed {
		t.Fatalf("F16: rejected precondition must leave server and Install ready: error=%+v quiesced=%d serving=%v state=%s install_allowed=%v", result.Error, quiesced, serving, state.State, installAllowed)
	}
}

func TestV8F14RollbackPendingRecoverActionAndBinding(t *testing.T) {
	t.Run("View action", func(t *testing.T) {
		a, _, _ := v8RollbackApp(t)
		v := a.GetView()
		field := reflect.ValueOf(v.Actions).FieldByName("Recover")
		if !field.IsValid() {
			t.Fatal("F14: View.actions.recover is absent in rollback_pending")
		}
		recover, ok := field.Interface().(ActionAvailability)
		if !ok || !recover.Allowed || recover.ReasonCode != "" {
			t.Fatalf("F14: Recover must be allowed without a reason: %+v", recover)
		}
	})
	t.Run("bound Recover", func(t *testing.T) {
		a, client, exited := v8RollbackApp(t)
		method := reflect.ValueOf(a).MethodByName("RecoverUpdate")
		if !method.IsValid() {
			t.Fatal("F14: GUI RecoverUpdate binding is absent")
		}
		out := method.Call(nil)
		if len(out) != 1 {
			t.Fatalf("F14: RecoverUpdate returned %d values", len(out))
		}
		result, ok := out[0].Interface().(ActionResult)
		if !ok || result.Error != nil || client.recoveries != 1 || !*exited {
			t.Fatalf("F14: Recover binding did not reach updater handoff: result=%+v recoveries=%d exited=%v", result, client.recoveries, *exited)
		}
	})
}
