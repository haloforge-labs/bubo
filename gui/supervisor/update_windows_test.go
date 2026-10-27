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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bubo/gui/updater"
)

const updateReleasePath = "/haloforge-labs/bubo/releases/"

func updatePackage(t *testing.T) (updater.Asset, []byte) {
	t.Helper()
	var buf bytes.Buffer
	z := zip.NewWriter(&buf)
	var unpacked int64
	for _, file := range []string{"bin/bubo.exe", "bin/Bubo.Control.Web.exe", "bin/bubo-update.exe"} {
		w, err := z.Create("bubo-v0.1.2-windows-x64-gfx1151/" + file)
		if err != nil {
			t.Fatal(err)
		}
		body := []byte("verified fixture " + file)
		if _, err := w.Write(body); err != nil {
			t.Fatal(err)
		}
		unpacked += int64(len(body))
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	hash := sha256.Sum256(data)
	return updater.Asset{Name: "bubo-v0.1.2-windows-x64-gfx1151.zip", Kind: "package", Size: int64(len(data)), SHA256: hex.EncodeToString(hash[:]), UnpackedBytes: unpacked, TopFolder: "bubo-v0.1.2-windows-x64-gfx1151"}, data
}

func updateFeed(origin string, asset updater.Asset) updater.Manifest {
	asset.URL = origin + updateReleasePath + "download/v0.1.2/" + asset.Name
	return updater.Manifest{SchemaVersion: 1, Product: "bubo", Channel: "stable", Version: "0.1.2", Tag: "v0.1.2", Commit: strings.Repeat("a", 40), Released: time.Date(2026, 10, 14, 0, 0, 0, 0, time.UTC), MinUpdatableFrom: "0.1.0", Assets: []updater.Asset{asset}, Notes: updater.Notes{Summary: "Fixture release", URL: origin + updateReleasePath + "tag/v0.1.2"}, Runtime: updater.Runtime{ID: "fixture-runtime"}, Models: updater.Models{NeedsPrepare: []string{}}, Advisories: []updater.Advisory{}, RestartRequired: true}
}

func updateConfig(t *testing.T, server *httptest.Server) updater.Config {
	t.Helper()
	root := filepath.Join(t.TempDir(), "Bubo")
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0700); err != nil {
		t.Fatal(err)
	}
	pool := x509.NewCertPool()
	pool.AddCert(server.Certificate())
	return updater.Config{InstalledVersion: "0.1.1", InstallDir: root, ManifestURL: server.URL + updateReleasePath + "latest/download/bubo-release.json", AllowedHosts: []string{"127.0.0.1"}, RootCAs: pool}
}

func waitUpdate(t *testing.T, u *UpdateService, pred func(Update) bool) Update {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		v := u.State()
		if pred(v) {
			return v
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("update timeout: %+v", u.State())
	return Update{}
}

func readyWindow() <-chan struct{} { ch := make(chan struct{}); close(ch); return ch }

func saneUpdateSnapshot() Snapshot {
	return Snapshot{Initialized: true, State: NotPrepared, Status: &Status{Versions: Product{Name: "bubo", Version: "0.1.1"}, Capabilities: Capabilities{StatusJSON: true}}}
}

func TestUpdateStartupDownloadInstallWithFakeCLIAndTLS(t *testing.T) {
	b := backendFor(t, "ui")
	reply(t, b.Action("prepare", b.settings, ""))
	first := waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
	if len(first.Approval.Components) != 1 || first.Approval.Components[0] != "runtime" {
		t.Fatal("first consent must be runtime-only", first.Approval)
	}
	reply(t, b.Action("approve", b.settings, ""))
	waitBackend(t, b, func(v Snapshot) bool {
		return v.State == Stopped && v.Status != nil && v.Status.Readiness.Runtime && !v.Status.Readiness.Model
	})
	reply(t, b.s.Prepare("model"))
	second := waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
	if len(second.Approval.Components) != 1 || second.Approval.Components[0] != "model" {
		t.Fatal("second consent must be model-only", second.Approval)
	}
	reply(t, b.Action("approve", b.settings, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped && v.Status != nil && v.Status.Readiness.Model })
	reply(t, b.Action("start", b.settings, ""))
	serverIdentity := *waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready }).Instance
	serverPID := serverIdentity.PID
	asset, archive := updatePackage(t)
	var feed updater.Manifest
	var checks, downloads atomic.Int32
	entered, releaseCheck := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseCheck) }) }
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".zip") {
			downloads.Add(1)
			w.Write(archive)
			return
		}
		checks.Add(1)
		select {
		case entered <- struct{}{}:
		default:
		}
		select {
		case <-releaseCheck:
		case <-r.Context().Done():
			return
		}
		json.NewEncoder(w).Encode(feed)
	}))
	defer server.Close()
	defer release()
	feed = updateFeed(server.URL, asset)
	cfg := updateConfig(t, server)
	helper, err := os.ReadFile(updateHelperExe)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfg.InstallDir, "bin", "bubo-update.exe"), helper, 0700); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "handoff.json")
	t.Setenv("BUBO_TEST_UPDATE_HANDOFF", output)
	health, err := NewUpdateHealth(cfg.InstallDir, "fresh-token", time.Now().Add(5*time.Second).UTC().Format(time.RFC3339Nano))
	if err != nil {
		t.Fatal(err)
	}
	exited := make(chan struct{}, 1)
	progress := make(chan UpdateProgress, 16)
	u := NewUpdateService(UpdateServiceConfig{Updater: cfg, Snapshot: b.Snapshot, Health: health, Quiesce: b.QuiesceForUpdate(nil), Exit: func() { exited <- struct{}{} }, Events: func(e StreamEvent) {
		if e.Kind == UpdateProgressStream {
			progress <- *e.Update
		}
	}})
	defer u.Close()
	window := make(chan struct{})
	u.Start(window)
	u.Start(window)
	if checks.Load() != 0 {
		t.Fatal("check preceded window initialization")
	}
	close(window)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("startup check did not start")
	}
	body, err := os.ReadFile(filepath.Join(cfg.InstallDir, updater.HealthFileName))
	if err != nil || !bytes.Equal(body, updater.HealthContent("fresh-token")) {
		t.Fatalf("network began before health acknowledgment: %q %v", body, err)
	}
	if downloads.Load() != 0 {
		t.Fatal("startup downloaded automatically")
	}
	release()
	v := waitUpdate(t, u, func(v Update) bool { return v.State == UpdateAvailable })
	if checks.Load() != 1 || v.AvailableVersion != "0.1.2" || v.SizeBytes == nil || *v.SizeBytes != uint64(len(archive)) || v.Notes != feed.Notes.Summary {
		t.Fatal("available view did not preserve manifest", v)
	}
	if r := u.DownloadUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	v = u.State()
	if v.State != UpdateReady || v.Progress == nil || v.Progress.Phase != "ready" || downloads.Load() != 1 {
		t.Fatal("unverified update marked ready", v)
	}
	seenProgress := false
	for len(progress) > 0 {
		p := <-progress
		if p.BytesDone > 0 {
			seenProgress = true
		}
		if p.BytesTotal == nil {
			t.Fatal("progress total lost")
		}
	}
	if !seenProgress {
		t.Fatal("no progress stream")
	}
	if r := u.InstallUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	select {
	case <-exited:
	default:
		t.Fatal("GUI exit seam not called after handoff")
	}
	if b.Snapshot().State != Stopped || b.Snapshot().Forced || b.Snapshot().Instance != nil {
		t.Fatal("server was not stopped cleanly before handoff")
	}
	var handoff struct {
		Args []string
		CWD  string
		PID  int
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(output)
		if err == nil && json.Unmarshal(data, &handoff) == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(handoff.Args) == 0 || handoff.PID <= 0 {
		t.Fatal("real hidden helper launch was not observed")
	}
	joined := strings.Join(handoff.Args, " ")
	for _, want := range []string{"--install", cfg.InstallDir, "--wait-pids", strconv.Itoa(os.Getpid()) + ":", strconv.FormatUint(uint64(serverPID), 10) + ":" + serverIdentity.CreationTime} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing helper argument %q: %v", want, handoff.Args)
		}
	}
	if strings.HasPrefix(handoff.CWD, cfg.InstallDir+string(os.PathSeparator)) || filepath.Dir(handoff.Args[0]) != handoff.CWD {
		t.Fatal("helper image/cwd remained inside package")
	}
}

func TestUpdateStartupOffAndHTTPFailure(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(strconv.FormatBool(enabled), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				http.Error(w, "server unavailable fixture", http.StatusServiceUnavailable)
			}))
			defer server.Close()
			u := NewUpdateService(UpdateServiceConfig{Updater: updateConfig(t, server), Snapshot: saneUpdateSnapshot, Enabled: func() bool { return enabled }})
			defer u.Close()
			u.Start(readyWindow())
			if enabled {
				v := waitUpdate(t, u, func(v Update) bool { return v.State == UpdateFailed })
				if v.Error == nil || v.Error.Code != string(updater.CodeUpdateServer) || !strings.Contains(v.Error.Remedy, "503") || requests.Load() != 1 {
					t.Fatal("HTTP server failure lost its cause", v)
				}
			} else {
				waitUpdate(t, u, func(v Update) bool { return v.CurrentVersion != "" })
				// Joining startup removes scheduler timing from the opt-out proof.
				u.wg.Wait()
				if requests.Load() != 0 || u.State().State != UpdateNone {
					t.Fatal("opt-out performed a check")
				}
				if r := u.CheckUpdate(); r.Error == nil || requests.Load() != 1 {
					t.Fatal("explicit check unavailable with startup opt-out")
				}
			}
		})
	}
}

func writeUpdateResult(t *testing.T, cfg updater.Config, result updater.InstallResult) {
	t.Helper()
	root := filepath.Join(filepath.Dir(cfg.InstallDir), "."+filepath.Base(cfg.InstallDir)+".bubo-update")
	transaction := filepath.Join(root, "fixture")
	if err := os.MkdirAll(transaction, 0700); err != nil {
		t.Fatal(err)
	}
	active := struct {
		Root    string `json:"root"`
		Version string `json:"version"`
		Seal    string `json:"seal"`
		Helper  string `json:"helper"`
	}{transaction, "0.1.2", "fixture-seal", filepath.Join(transaction, "helper.exe")}
	for file, value := range map[string]interface{}{filepath.Join(root, "active.json"): active, filepath.Join(transaction, updater.ResultFileName): result} {
		body, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, body, 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestUpdateLastResultNotices(t *testing.T) {
	for _, status := range []string{"installed", "rolled_back", "failed", "rollback_pending"} {
		t.Run(status, func(t *testing.T) {
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("result notice fetched network") }))
			defer server.Close()
			cfg := updateConfig(t, server)
			result := updater.InstallResult{Status: status, Version: "0.1.2"}
			if status != "installed" {
				result.Failure = &updater.Failure{Code: updater.CodeHealth, Message: "candidate health\nfailed"}
			}
			writeUpdateResult(t, cfg, result)
			u := NewUpdateService(UpdateServiceConfig{Updater: cfg, Snapshot: saneUpdateSnapshot, Enabled: func() bool { return false }})
			defer u.Close()
			u.Start(readyWindow())
			u.wg.Wait()
			v := u.State()
			want := map[string]string{"installed": "Updated to 0.1.2.", "rolled_back": "Rolled back to 0.1.1.", "failed": "Update failed.", "rollback_pending": "Update rollback is pending."}[status]
			if !strings.HasPrefix(v.LastResultNotice, want) || strings.Contains(v.LastResultNotice, "\n") {
				t.Fatal("wrong one-line notice", v)
			}
			if status != "installed" && !strings.Contains(v.LastResultNotice, "candidate health failed") {
				t.Fatal("rollback reason lost", v)
			}
			if status == "rollback_pending" {
				if v.State != UpdateRollbackPending || u.DismissUpdate().Error == nil || u.CheckUpdate().Error == nil {
					t.Fatal("pending rollback was hidden", v)
				}
			}
		})
	}
}

func TestUpdateCancellationAndDismiss(t *testing.T) {
	asset, archive := updatePackage(t)
	var feed updater.Manifest
	entered := make(chan struct{}, 1)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".zip") {
			w.Header().Set("Content-Length", strconv.Itoa(len(archive)))
			w.Write(archive[:10])
			w.(http.Flusher).Flush()
			entered <- struct{}{}
			<-r.Context().Done()
			return
		}
		json.NewEncoder(w).Encode(feed)
	}))
	defer server.Close()
	feed = updateFeed(server.URL, asset)
	u := NewUpdateService(UpdateServiceConfig{Updater: updateConfig(t, server), Snapshot: saneUpdateSnapshot})
	defer u.Close()
	u.Start(readyWindow())
	waitUpdate(t, u, func(v Update) bool { return v.State == UpdateAvailable })
	done := make(chan ActionResult, 1)
	go func() { done <- u.DownloadUpdate() }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("download did not begin")
	}
	if r := u.DismissUpdate(); r.Error == nil || r.Error.Code != "busy" {
		t.Fatal("dismiss orphaned active transfer")
	}
	if r := u.CheckUpdate(); r.Error == nil || r.Error.Code != "busy" {
		t.Fatal("check raced transfer")
	}
	if r := u.CancelUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	select {
	case r := <-done:
		if r.Error != nil {
			t.Fatal(r.Error)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancel did not drain transfer")
	}
	if u.State().State != UpdateAvailable || u.State().Progress != nil {
		t.Fatal("canceled transfer remained busy")
	}
	if r := u.DismissUpdate(); r.Error != nil || u.State().State != UpdateNone {
		t.Fatal("Later did not dismiss")
	}
	if r := u.InstallUpdate(); r.Error == nil {
		t.Fatal("dismissed/unverified update installed")
	}
}

func TestUpdateHealthRequiresWindowAndSaneSupervisor(t *testing.T) {
	for _, kind := range []string{"window", "supervisor", "unwritable"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			health, err := NewUpdateHealth(root, "token", time.Now().Add(100*time.Millisecond).UTC().Format(time.RFC3339Nano))
			if err != nil {
				t.Fatal(err)
			}
			snapshot := saneUpdateSnapshot
			window := readyWindow()
			if kind == "window" {
				window = make(chan struct{})
			}
			if kind == "supervisor" {
				snapshot = func() Snapshot { return Snapshot{} }
			}
			if kind == "unwritable" {
				health.InstallDir = filepath.Join(root, "missing")
			}
			exit := make(chan struct{}, 1)
			u := NewUpdateService(UpdateServiceConfig{Updater: updater.Config{InstalledVersion: "0.1.1", InstallDir: root}, Health: health, Snapshot: snapshot, Exit: func() { exit <- struct{}{} }})
			defer u.Close()
			u.Start(window)
			select {
			case <-exit:
			case <-time.After(3 * time.Second):
				t.Fatal("unhealthy candidate did not request exit")
			}
			if _, err := os.Stat(filepath.Join(root, updater.HealthFileName)); !os.IsNotExist(err) {
				t.Fatal("unhealthy candidate acknowledged health", err)
			}
			if v := u.State(); v.Error == nil || v.Error.Code != string(updater.CodeHealth) {
				t.Fatal("health failure not typed", v)
			}
		})
	}
}

func TestUpdateDiagnosticsPreserveFrozenErrorFields(t *testing.T) {
	f := &updater.Failure{Code: updater.CodeHash, Message: "hash mismatch", URL: "https://fixture.invalid/archive", Path: "fixture.zip", ExpectedBytes: 10, ReceivedBytes: 9, ExpectedHash: "expected", ReceivedHash: "received", Partial: "removed", Retry: "Retry Download", NotesURL: "https://fixture.invalid/notes"}
	e := updateError(f)
	for _, want := range []string{f.URL, f.Path, "expected 10", "received 9", f.ExpectedHash, f.ReceivedHash, f.Partial, f.Retry, f.NotesURL} {
		if !strings.Contains(e.Remedy, want) {
			t.Fatalf("lost diagnostic %q: %+v", want, e)
		}
	}
	if e.Code != string(f.Code) || e.Component != "update" {
		t.Fatal(e)
	}
	if _, err := NewUpdateHealth(t.TempDir(), "bad\ntoken", time.Now().Add(time.Second).Format(time.RFC3339Nano)); err == nil {
		t.Fatal("accepted invalid health token")
	}
	if got := GUIPackageRoot(filepath.Join("C:\\", "fixture", "bin", "Bubo.Control.Web.exe")); got != filepath.Join("C:\\", "fixture") {
		t.Fatal(got)
	}
}

type stubUpdateClient struct {
	cfg                          updater.Config
	check                        updater.CheckResult
	result                       *updater.InstallResult
	installFailure               *updater.Failure
	installs, recoveries, checks int
	beforeInstall                func()
}

func (c *stubUpdateClient) Check(context.Context) updater.CheckResult { c.checks++; return c.check }
func (c *stubUpdateClient) Download(_ context.Context, progress func(updater.Progress)) (*updater.Staged, *updater.Failure) {
	progress(updater.Progress{Phase: "ready", TotalBytes: 10, ReceivedBytes: 10})
	return &updater.Staged{Directory: "fixture", Version: "0.1.2"}, nil
}
func (c *stubUpdateClient) Install(*updater.Staged) (*updater.Handoff, *updater.Failure) {
	c.installs++
	if c.beforeInstall != nil {
		c.beforeInstall()
	}
	return &updater.Handoff{HelperPID: 123}, c.installFailure
}
func (c *stubUpdateClient) Recover() (*updater.Handoff, *updater.Failure) {
	c.recoveries++
	return c.Install(nil)
}
func (c *stubUpdateClient) LastResult() (*updater.InstallResult, *updater.Failure) {
	return c.result, nil
}
func (c *stubUpdateClient) Close() {}

func TestUpdateHandoffFailureAndRecoveryKeepTruthfulState(t *testing.T) {
	for _, recovery := range []bool{false, true} {
		t.Run(strconv.FormatBool(recovery), func(t *testing.T) {
			client := &stubUpdateClient{check: updater.CheckResult{Status: updater.Available, Available: &updater.Release{Version: "0.1.2", Size: 10}}, installFailure: &updater.Failure{Code: updater.CodeIO, Message: "helper blocked"}}
			if recovery {
				client.result = &updater.InstallResult{Status: "rollback_pending"}
			}
			var exited, released bool
			u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot, Quiesce: func(context.Context) ([]uint32, func(), error) { return []uint32{42}, func() { released = true }, nil }, Exit: func() { exited = true }})
			u.factory = func(cfg updater.Config) (updateClient, *updater.Failure) { client.cfg = cfg; return client, nil }
			defer u.Close()
			u.Start(readyWindow())
			u.wg.Wait()
			if !recovery {
				if r := u.DownloadUpdate(); r.Error != nil {
					t.Fatal(r.Error)
				}
			}
			client.beforeInstall = func() {
				pids, err := client.cfg.ProcessIDs()
				if err != nil || len(pids) != 2 || pids[0] != uint32(os.Getpid()) || pids[1] != 42 {
					t.Errorf("handoff did not receive fresh owned PIDs: %v %v", pids, err)
				}
			}
			if r := u.InstallUpdate(); r.Error == nil || r.Error.Code != string(updater.CodeIO) {
				t.Fatal("handoff failure reported success", r)
			}
			if exited || !released {
				t.Fatal("failed handoff exited or held process gate")
			}
			if recovery && (u.State().State != UpdateRollbackPending || client.recoveries != 1 || client.checks != 0) {
				t.Fatal("recovery was lost or fetched updates")
			}
			client.installFailure = nil
			if r := u.InstallUpdate(); r.Error != nil || !exited {
				t.Fatal("retry did not hand off", r)
			}
		})
	}
}

func TestUpdateIntermediateRequiresAnotherExplicitCheck(t *testing.T) {
	var clients []*stubUpdateClient
	u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot})
	u.factory = func(cfg updater.Config) (updateClient, *updater.Failure) {
		client := &stubUpdateClient{cfg: cfg, check: updater.CheckResult{Status: updater.Available, Available: &updater.Release{Version: "0.1.2", Size: 10}}}
		if len(clients) == 0 {
			client.check = updater.CheckResult{Status: updater.Failed, Failure: &updater.Failure{Code: updater.CodeIntermediateRequired, Message: "Install 0.1.2 before 0.1.3", Via: "0.1.2", URL: "https://fixture.invalid/intermediate", NotesURL: "https://fixture.invalid/notes"}}
		}
		clients = append(clients, client)
		return client, nil
	}
	defer u.Close()
	u.Start(readyWindow())
	u.wg.Wait()
	if len(clients) != 1 || clients[0].checks != 1 || u.State().Error.Code != string(updater.CodeIntermediateRequired) {
		t.Fatal("startup followed an implicit intermediate request")
	}
	if r := u.CheckUpdate(); r.Error != nil {
		t.Fatal(r.Error)
	}
	if len(clients) != 2 || clients[1].cfg.ManifestURL != "https://fixture.invalid/intermediate" || clients[1].checks != 1 || u.State().AvailableVersion != "0.1.2" {
		t.Fatal("explicit intermediate route lost")
	}
}

func TestUpdateModelPauseBlocksNewQueriesAndReleasesOnFailure(t *testing.T) {
	m := NewModelService(ModelServiceConfig{})
	defer m.Close()
	_, release, err := m.PauseForUpdate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.spawn([]string{"models", "list", "--json"}, ""); err == nil {
		t.Fatal("model query started during package handoff")
	}
	release()
	m.mu.Lock()
	m.job = &ModelJob{State: "running"}
	m.mu.Unlock()
	if _, _, err := m.PauseForUpdate(context.Background()); err == nil {
		t.Fatal("running model download was silently interrupted")
	}
}
