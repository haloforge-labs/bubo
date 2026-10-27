//go:build windows

package supervisor

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"bubo/gui/updater"
)

func TestUpdateBindingsViewProgressAndInstallFailure(t *testing.T) {
	const releasePath = "/haloforge-labs/bubo/releases/"
	var archive bytes.Buffer
	z := zip.NewWriter(&archive)
	var unpacked int64
	for _, name := range []string{"bin/bubo.exe", "bin/Bubo.Control.Web.exe", "bin/bubo-update.exe"} {
		w, err := z.Create("bubo-v0.1.2-windows-x64-gfx1151/" + name)
		if err != nil {
			t.Fatal(err)
		}
		body := []byte("fixture " + name)
		if _, err := w.Write(body); err != nil {
			t.Fatal(err)
		}
		unpacked += int64(len(body))
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(archive.Bytes())
	var feed updater.Manifest
	var checks atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".zip") {
			w.Write(archive.Bytes())
			return
		}
		checks.Add(1)
		json.NewEncoder(w).Encode(feed)
	}))
	defer server.Close()
	feed = updater.Manifest{SchemaVersion: 1, Product: "bubo", Channel: "stable", Version: "0.1.2", Tag: "v0.1.2", Commit: strings.Repeat("a", 40), Released: time.Date(2026, 10, 14, 0, 0, 0, 0, time.UTC), MinUpdatableFrom: "0.1.1", Assets: []updater.Asset{{Name: "bubo-v0.1.2-windows-x64-gfx1151.zip", Kind: "package", URL: server.URL + releasePath + "download/v0.1.2/bubo-v0.1.2-windows-x64-gfx1151.zip", Size: int64(archive.Len()), SHA256: hex.EncodeToString(hash[:]), UnpackedBytes: unpacked, TopFolder: "bubo-v0.1.2-windows-x64-gfx1151"}}, Notes: updater.Notes{Summary: "Binding fixture", URL: server.URL + releasePath + "tag/v0.1.2"}, Runtime: updater.Runtime{ID: "fixture-runtime"}, Models: updater.Models{NeedsPrepare: []string{}}, Advisories: []updater.Advisory{}}
	snapshot := actionTestInput(Stopped).Snapshot
	snapshot.Status.Capabilities = Capabilities{StatusJSON: true}
	b := &fakeBackend{view: snapshot}
	a := newTestApp(b, Settings{}, nil, "")
	installDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(installDir, "bin"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(installDir, "bin", "bubo-update.exe"), []byte("fixture helper"), 0700); err != nil {
		t.Fatal(err)
	}
	a.tray = &fakeTray{}
	progress := make(chan StreamEvent, 16)
	a.events = func(_ context.Context, event StreamEvent) { progress <- event }
	pool := x509.NewCertPool()
	pool.AddCert(server.Certificate())
	var stopped, exited bool
	a.updates = NewUpdateService(UpdateServiceConfig{Updater: updater.Config{InstalledVersion: "0.1.1", InstallDir: installDir, ManifestURL: server.URL + releasePath + "latest/download/bubo-release.json", AllowedHosts: []string{"127.0.0.1"}, RootCAs: pool}, Snapshot: b.Snapshot, Enabled: func() bool { return false }, Events: a.emitStream, Quiesce: func(context.Context) ([]uint32, func(), error) {
		stopped = true
		return nil, nil, errors.New("fixture refuses process stop")
	}, Exit: func() { exited = true }})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	defer a.Shutdown(ctx)
	if checks.Load() != 0 {
		t.Fatal("startup queried before DOM-ready")
	}
	a.DOMReady(ctx)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		result := a.CheckUpdate()
		if result.Error == nil {
			break
		}
		if result.Error.Code != "updater_starting" {
			t.Fatal(result.Error)
		}
		time.Sleep(5 * time.Millisecond)
	}
	if v := a.GetView(); v.Update.State != UpdateAvailable || v.Update.AvailableVersion != "0.1.2" || checks.Load() != 1 {
		t.Fatal("CheckUpdate did not reach View", v.Update)
	}
	if result := a.DownloadUpdate(); result.Error != nil {
		t.Fatal(result.Error)
	}
	if v := a.GetView(); v.Update.State != UpdateReady || v.Update.Progress == nil || v.Update.Progress.Phase != "ready" {
		t.Fatal("DownloadUpdate did not reach View", v.Update)
	}
	select {
	case event := <-progress:
		if event.Kind != UpdateProgressStream || event.Update == nil {
			t.Fatal("wrong Wails stream payload", event)
		}
	default:
		t.Fatal("no update progress event reached transport")
	}
	if result := a.InstallUpdate(); result.Error == nil || result.Error.Code != "process_wait" || !stopped || exited {
		t.Fatal("InstallUpdate hid stop failure", result)
	}
	if result := a.DismissUpdate(); result.Error != nil || a.GetView().Update.State != UpdateNone {
		t.Fatal("Later did not reach service", result)
	}
}

func TestUpdateQuitBypassesTrayAfterSuccessfulHandoff(t *testing.T) {
	b := &fakeBackend{}
	a := newTestApp(b, Settings{CloseToTray: true}, nil, "")
	a.ctx = context.Background()
	var quits, hides int
	a.quit = func(context.Context) { quits++ }
	a.hide = func(context.Context) { hides++ }
	a.QuitForUpdate()
	if a.BeforeClose(a.ctx) || quits != 1 || hides != 0 || !a.GetView().Window.Closing {
		t.Fatal("update quit was diverted to tray")
	}
}

func TestUpdatePackageRootKeepsCLIInShippedBin(t *testing.T) {
	for _, path := range []string{filepath.Join("C:\\", "Bubo", "Bubo.Control.Web.exe"), filepath.Join("C:\\", "Bubo", "bin", "Bubo.Control.Web.exe")} {
		if got := GUIPackageRoot(path); got != filepath.Join("C:\\", "Bubo") {
			t.Fatal(got)
		}
	}
}

func TestUpdateHealthPrecedesOptInServerAutostart(t *testing.T) {
	snapshot := actionTestInput(Stopped).Snapshot
	snapshot.Status.Capabilities = Capabilities{StatusJSON: true}
	b := &fakeBackend{view: snapshot}
	a := newTestApp(b, Settings{StartServerWithGUI: true}, nil, "")
	a.tray = &fakeTray{}
	root := t.TempDir()
	health, err := NewUpdateHealth(root, "autostart-token", time.Now().Add(5*time.Second).UTC().Format(time.RFC3339Nano))
	if err != nil {
		t.Fatal(err)
	}
	a.updates = NewUpdateService(UpdateServiceConfig{Updater: updater.Config{InstalledVersion: "0.1.1", InstallDir: root}, Snapshot: b.Snapshot, Health: health, Enabled: func() bool { return false }})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	defer a.Shutdown(ctx)
	time.Sleep(250 * time.Millisecond) // cross one real autostart poll with DOM-ready withheld
	b.mu.Lock()
	started := len(b.actions) != 0
	b.mu.Unlock()
	if started || a.updates.StartupReady() {
		t.Fatal("candidate started a server before acknowledging window health")
	}
	a.DOMReady(ctx)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		b.mu.Lock()
		started = len(b.actions) > 0 && b.actions[0] == "start"
		b.mu.Unlock()
		if started {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !started {
		t.Fatal("opt-in autostart did not resume after health")
	}
	body, err := os.ReadFile(filepath.Join(root, updater.HealthFileName))
	if err != nil || !bytes.Equal(body, updater.HealthContent("autostart-token")) {
		t.Fatalf("autostart preceded health acknowledgment: %q %v", body, err)
	}
}
