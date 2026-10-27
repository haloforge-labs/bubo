//go:build windows

package supervisor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func fakeCLI(t *testing.T) string {
	t.Helper()
	// Reuse the shared race fake without its process-exit sleep on each call.
	t.Setenv("GORACE", "atexit_sleep_ms=0")
	return fakeExe
}

type fakeDialogs struct {
	mu           sync.Mutex
	file, folder string
	err          error
	calls        []string
}

func (d *fakeDialogs) OpenFile(ctx context.Context, title string) (string, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "file:"+title)
	return d.file, d.err
}
func (d *fakeDialogs) OpenFolder(ctx context.Context, title string) (string, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.calls = append(d.calls, "folder:"+title)
	return d.folder, d.err
}

type modelsApp struct {
	a        *AppService
	b        *fakeBackend
	dir      string
	settings string
	dialogs  *fakeDialogs
}

func newModelsApp(t *testing.T, mode string) *modelsApp {
	t.Helper()
	exe := fakeCLI(t)
	dir := t.TempDir()
	t.Setenv("BUBO_FAKE_DIR", dir)
	t.Setenv("BUBO_FAKE_MODE", mode)
	t.Setenv("BUBO_FAKE_FIXTURES", filepath.Join("..", "..", "tests", "cli", "contract"))
	if abs, err := filepath.Abs(os.Getenv("BUBO_FAKE_FIXTURES")); err == nil {
		t.Setenv("BUBO_FAKE_FIXTURES", abs)
	}
	t.Setenv("LOCALAPPDATA", filepath.Join(dir, "local"))
	t.Setenv("USERPROFILE", filepath.Join(dir, "profile"))
	b := &fakeBackend{settingsPath: filepath.Join(dir, "gui", "settings.json")}
	a := newTestApp(b, DefaultSettings(), nil, "")
	a.AttachModels(filepath.Join(dir, "gui", "settings.json"), exe, "")
	d := &fakeDialogs{}
	a.dialogs = d
	a.ctx = context.Background()
	a.events = func(context.Context, StreamEvent) {} // headless transport; no Wails runtime context
	t.Cleanup(func() { a.models.Close() })
	return &modelsApp{a: a, b: b, dir: dir, settings: filepath.Join(dir, "gui", "settings.json"), dialogs: d}
}

func TestModelsUsePackagedCLIWithSeparateRuntimeFolder(t *testing.T) {
	m := newModelsApp(t, "")
	m.a.mu.Lock()
	m.a.settings.RuntimePath = filepath.Join(t.TempDir(), "runtime without an exe")
	m.a.mu.Unlock()
	if err := actionTestError(m.a.RefreshModels()); err != nil {
		t.Fatal(err)
	}
	m.waitModels(t, func(s *ViewModelsState) bool { return s.Status == ModelsOK })
}

func (m *modelsApp) waitModels(t *testing.T, pred func(*ViewModelsState) bool) *ViewModelsState {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		f := m.a.GetView()
		if f.Models != nil && pred(f.Models) {
			return f.Models
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("frame.models never reached the state; last: %+v", m.a.GetView().Models)
	return nil
}
func listed(s *ViewModelsState) bool {
	return s.Status != ModelsLoading && !s.Refreshing
}

func (m *modelsApp) folder(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join(m.dir, name)
	if err := os.MkdirAll(p, 0700); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestFrameWithoutModelServiceHasNullModelsAndTypedErrors(t *testing.T) {
	a := newTestApp(&fakeBackend{}, DefaultSettings(), nil, "")
	data, _ := json.Marshal(a.GetView())
	if !strings.Contains(string(data), `"models":null`) {
		t.Fatalf("frame: %s", data)
	}
	if result := a.Download("x"); result.Error == nil || result.Error.Code != "internal_error" || result.Plan != nil {
		t.Fatalf("Download: %+v", result)
	}
	if result := a.AddModelFolder(`D:\x`); result.Error == nil || result.Error.Code != "internal_error" || result.Value != "" {
		t.Fatalf("AddModelFolder: %+v", result)
	}
	if result := a.BrowseModelFile(); result.Error == nil || result.Error.Code != "internal_error" || result.File != "" {
		t.Fatalf("BrowseModelFile: %+v", result)
	}
}

func TestFrameCarriesTheModelList(t *testing.T) {
	m := newModelsApp(t, "")
	if err := os.MkdirAll(filepath.Join(m.dir, "profile", ".lmstudio", "models"), 0700); err != nil {
		t.Fatal(err)
	}
	first := m.a.GetView().Models
	if first == nil || first.Status != ModelsLoading {
		t.Fatalf("first frame: %+v", first)
	}
	s := m.waitModels(t, listed)
	// The counts are the recorded list's (models-list-gui), not numbers written here.
	recorded := e21Models(t)
	if s.Status != ModelsOK || len(s.Catalogue) != len(recorded.Catalogue) || len(s.Scanned) != len(recorded.Scanned) || len(s.PreparedModels) != len(recorded.PreparedModels) || len(s.Catalogue) == 0 || s.Disk.FreeBytes == nil || len(s.SuggestedFolders) != 1 {
		t.Fatalf("models: %+v", s)
	}
	data, _ := json.Marshal(m.a.GetView())
	var frame map[string]json.RawMessage
	if err := json.Unmarshal(data, &frame); err != nil {
		t.Fatal(err)
	}
	var models map[string]json.RawMessage
	if err := json.Unmarshal(frame["models"], &models); err != nil {
		t.Fatalf("frame.models: %s", frame["models"])
	}
	for _, key := range []string{"status", "listing", "catalogue", "prepared_models", "scanned", "scan_warnings", "disk", "suggested_folders", "folders", "download"} {
		if _, ok := models[key]; !ok {
			t.Errorf("frame.models.%s is missing", key)
		}
	}
	var disk struct {
		Drive     string  `json:"drive"`
		FreeBytes *uint64 `json:"free_bytes"`
	}
	json.Unmarshal(models["disk"], &disk)
	if disk.Drive == "" || disk.FreeBytes == nil || *disk.FreeBytes == 0 {
		t.Fatalf("disk: %s", models["disk"])
	}
}

func TestRuntimeMissingFrameSaysNotReported(t *testing.T) {
	m := newModelsApp(t, "runtime-missing")
	s := m.waitModels(t, listed)
	if s.Status != ModelsNotReported || s.ReasonCode != "runtime_missing" || s.Reason == "" || len(s.Catalogue) != 0 {
		t.Fatalf("models: %+v", s)
	}
	data, _ := json.Marshal(m.a.GetView())
	// "Not reported" is the status; lists are published empty, never null, because
	// the frontend calls .find on them (owner's first-launch crash, 2026-10-15).
	if !strings.Contains(string(data), `"catalogue":[]`) || !strings.Contains(string(data), `"status":"not_reported"`) {
		t.Fatalf("frame: %.400s", data)
	}
}

func TestAddModelFolderSavesAndScansIt(t *testing.T) {
	m := newModelsApp(t, "")
	folder := m.folder(t, "my-models")
	got := m.a.AddModelFolder(folder + `\`)
	if got.Error != nil || got.Value != folder {
		t.Fatalf("%+v", got)
	}
	saved, warning := LoadSettings(m.settings)
	if warning != "" || len(saved.ModelFolders) != 1 || saved.ModelFolders[0] != folder {
		t.Fatalf("saved: %+v %q", saved, warning)
	}
	if m.a.GetView().Saved.ModelFolders[0] != folder {
		t.Fatal("the frame settings do not show the folder")
	}
	s := m.waitModels(t, func(s *ViewModelsState) bool { return listed(s) && len(s.Folders) == 1 })
	// The fake replays the recorded list, so the list cannot show the new
	// folder. What the app owes the CLI is the request: models list --scan-folder.
	if !modelsListAsked(t, m.dir, folder) {
		t.Fatalf("the CLI was not asked to scan the folder: %+v", s.Listing)
	}
	// The same folder again changes nothing; another spelling is the same folder.
	if again := m.a.AddModelFolder(strings.ToUpper(folder)); again.Error != nil || again.Value != folder {
		t.Fatalf("%+v", again)
	}
	if saved, _ := LoadSettings(m.settings); len(saved.ModelFolders) != 1 {
		t.Fatalf("duplicate saved: %v", saved.ModelFolders)
	}
	if err := actionTestError(m.a.RemoveModelFolder(folder)); err != nil {
		t.Fatal(err)
	}
	if saved, _ := LoadSettings(m.settings); len(saved.ModelFolders) != 0 {
		t.Fatalf("not removed: %v", saved.ModelFolders)
	}
	if _, err := os.Stat(folder); err != nil {
		t.Fatal("removing a folder from the list must not touch the folder")
	}
	if err := actionTestError(m.a.RemoveModelFolder(folder)); err == nil {
		t.Fatal("removing an unknown folder was accepted")
	}
}

// modelsListAsked reports whether a `models list` call of the fake
// (models-argv-N.json) carried --scan-folder FOLDER.
func modelsListAsked(t *testing.T, dir, folder string) bool {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		files, _ := filepath.Glob(filepath.Join(dir, "models-argv-*.json"))
		for _, file := range files {
			data, err := os.ReadFile(file)
			if err != nil {
				continue
			}
			var args []string
			if json.Unmarshal(data, &args) != nil {
				continue
			}
			for i := 0; i+1 < len(args); i++ {
				if args[i] == "--scan-folder" && SameFolder(args[i+1], folder) {
					return true
				}
			}
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestAddModelFolderRefusesWhatTheCLIWouldRefuse(t *testing.T) {
	m := newModelsApp(t, "")
	file := filepath.Join(m.dir, "a-file.txt")
	os.WriteFile(file, []byte("x"), 0600)
	for name, in := range map[string]string{
		"empty": "", "drive root": `C:\`, "drive": `C:`, "share root": `\\server\share`,
		"relative": `models`, "missing": filepath.Join(m.dir, "does-not-exist"), "file": file, "hyphen": `-rf`,
	} {
		if got := m.a.AddModelFolder(in); got.Error == nil || got.Value != "" {
			t.Errorf("%s accepted as %+v", name, got)
		}
	}
	saved, _ := LoadSettings(m.settings)
	if len(saved.ModelFolders) != 0 {
		t.Fatalf("a refused folder was saved: %v", saved.ModelFolders)
	}
	// No refused call ever reached the CLI as a scan folder.
	for n := 1; n < 6; n++ {
		if b, err := os.ReadFile(filepath.Join(m.dir, fmt.Sprintf("models-argv-%d.json", n))); err == nil && strings.Contains(string(b), "--scan-folder") {
			t.Fatalf("scan folder passed: %s", b)
		}
	}
}

func TestAddModelFolderStopsAtTheLimit(t *testing.T) {
	m := newModelsApp(t, "")
	for i := 0; i < MaxModelFolders; i++ {
		if result := m.a.AddModelFolder(m.folder(t, fmt.Sprintf("f%02d", i))); result.Error != nil {
			t.Fatal(result.Error)
		}
	}
	if result := m.a.AddModelFolder(m.folder(t, "one-too-many")); result.Error == nil {
		t.Fatal("the folder list grew past the limit")
	}
}

func TestPreferencePatchKeepsModelFolders(t *testing.T) {
	m := newModelsApp(t, "")
	folder := m.folder(t, "kept")
	if result := m.a.AddModelFolder(folder); result.Error != nil {
		t.Fatal(result.Error)
	}
	old := DefaultSettings() // the window's copy from before the folder was added
	old.Port = 18110
	if err := patchTestPreferences(m.a, old); err != nil {
		t.Fatal(err)
	}
	if got := m.b.settings.ModelFolders; len(got) != 1 || got[0] != folder {
		t.Fatalf("the backend saved %v", got)
	}
	if got := m.a.GetView().Saved; len(got.ModelFolders) != 1 || (got.Port == nil || *got.Port != 18110) {
		t.Fatalf("frame settings: %+v", got)
	}
}

func TestBrowseModelFolderAddsTheChosenFolder(t *testing.T) {
	m := newModelsApp(t, "")
	m.dialogs.folder = m.folder(t, "picked")
	res := m.a.BrowseModelFolder()
	if res.Error != nil || res.Cancelled || !res.Added || res.Folder != m.dialogs.folder || !strings.HasPrefix(res.Message, "Folder added: ") {
		t.Fatalf("%+v", res)
	}
	if len(m.dialogs.calls) != 1 || !strings.HasPrefix(m.dialogs.calls[0], "folder:") {
		t.Fatalf("dialog calls: %v", m.dialogs.calls)
	}
	res = m.a.BrowseModelFolder()
	if res.Error != nil || res.Added || !strings.HasPrefix(res.Message, "This folder is already in the list") {
		t.Fatalf("second pick: %+v", res)
	}
}

func TestBrowseModelFileAddsTheFolderOfTheFile(t *testing.T) {
	m := newModelsApp(t, "")
	folder := m.folder(t, "gguf-here")
	m.dialogs.file = filepath.Join(folder, "Model.GGUF")
	res := m.a.BrowseModelFile()
	if res.Error != nil || res.Cancelled || !res.Added || res.File != m.dialogs.file || res.Folder != folder {
		t.Fatalf("%+v", res)
	}
	m.dialogs.file = filepath.Join(folder, "notes.txt")
	if res := m.a.BrowseModelFile(); res.Error == nil || !strings.Contains(res.Error.Remedy, ".gguf") {
		t.Fatalf("a non-gguf file: %+v", res)
	}
	m.dialogs.file = `C:\model.gguf`
	if res := m.a.BrowseModelFile(); res.Error == nil {
		t.Fatal("a file in a drive root added the whole drive")
	}
	if saved, _ := LoadSettings(m.settings); len(saved.ModelFolders) != 1 {
		t.Fatalf("saved: %v", saved.ModelFolders)
	}
}

func TestBrowseCancelAndDialogFailure(t *testing.T) {
	m := newModelsApp(t, "")
	for name, call := range map[string]func() BrowseResult{"folder": m.a.BrowseModelFolder, "file": m.a.BrowseModelFile} {
		res := call()
		if res.Error != nil || !res.Cancelled || res.Added {
			t.Errorf("%s cancel: %+v", name, res)
		}
	}
	m.dialogs.err = errors.New("no shell")
	if res := m.a.BrowseModelFolder(); res.Error == nil || !strings.Contains(res.Error.Remedy, "dialog failed") {
		t.Fatalf("dialog failure: %+v", res)
	}
	if saved, _ := LoadSettings(m.settings); len(saved.ModelFolders) != 0 {
		t.Fatalf("saved: %v", saved.ModelFolders)
	}
	m.a.ctx = nil
	m.dialogs.err = nil
	if res := m.a.BrowseModelFile(); res.Error == nil || !strings.Contains(res.Error.Remedy, "not ready") {
		t.Fatalf("no window: %+v", res)
	}
}

func TestDownloadNeedsPlanThenConfirmThroughTheBindings(t *testing.T) {
	m := newModelsApp(t, "")
	m.waitModels(t, listed)
	result := m.a.Download(e21Flash)
	plan := result.Plan
	if result.Error != nil || plan == nil || plan.Digest == "" || !plan.EnoughSpace || len(plan.Licences) != len(e21Plan(t, e21Flash).Licences) {
		t.Fatalf("%+v", result)
	}
	f := m.a.GetView().Models
	if f.Download == nil || f.Download.State != "awaiting_confirm" || f.Download.Plan == nil {
		t.Fatalf("frame while the plan waits: %+v", f.Download)
	}
	if err := actionTestError(m.a.ConfirmDownload(e21Flash)); err != nil {
		t.Fatal(err)
	}
	s := m.waitModels(t, func(s *ViewModelsState) bool {
		return s.Download != nil && (s.Download.State == "done" || s.Download.State == "failed")
	})
	if s.Download.State != "done" || !s.Download.Prepared || s.Download.Plan != nil {
		t.Fatalf("download: %+v", s.Download)
	}
	if result := m.a.DismissModelJob(); result.Error != nil {
		t.Fatal(result.Error)
	}
	if m.a.GetView().Models.Download != nil {
		t.Fatal("job not dismissed")
	}
}

func TestCheckAndRepairThroughTheBindings(t *testing.T) {
	m := newModelsApp(t, "")
	if res := m.a.CheckModel(e21Flash, false); res.Error != nil || res.Plan == nil || res.Plan.Kind != "check" {
		t.Fatalf("%+v", res)
	}
	s := m.waitModels(t, func(s *ViewModelsState) bool { return s.Download != nil && s.Download.State == "done" })
	if s.Download.Kind != "check" {
		t.Fatalf("job: %+v", s.Download)
	}
	if result := m.a.DismissModelJob(); result.Error != nil {
		t.Fatal(result.Error)
	}
	result := m.a.CheckModel(e21Flash, true)
	if result.Error != nil || result.Plan == nil || result.Plan.Kind != "repair" || result.Plan.Digest == "" {
		t.Fatalf("%+v", result)
	}
	if err := actionTestError(m.a.CancelDownload()); err != nil {
		t.Fatal(err)
	}
	if j := m.a.GetView().Models.Download; j.State != "cancelled" {
		t.Fatalf("job: %+v", j)
	}
}

func TestEstimateBinding(t *testing.T) {
	m := newModelsApp(t, "")
	result := m.a.Estimate(e21Folder(t, e21Flash), 65536, 1)
	if result.Error != nil || result.Estimate == nil || result.Estimate.State != "estimated" {
		t.Fatalf("%+v", result)
	}
	if result := m.a.Estimate("a b", 0, 0); result.Error == nil || result.Estimate != nil {
		t.Fatal("bad subject accepted")
	}
}

func TestShutdownEndsARunningDownload(t *testing.T) {
	m := newModelsApp(t, "model-stall")
	if result := m.a.Download(e21Flash); result.Error != nil {
		t.Fatal(result.Error)
	}
	if err := actionTestError(m.a.ConfirmDownload(e21Flash)); err != nil {
		t.Fatal(err)
	}
	m.waitModels(t, func(s *ViewModelsState) bool { return s.Download != nil && s.Download.Phase == "download" })
	done := make(chan struct{})
	go func() { m.a.Shutdown(nil); close(done) }()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("shutdown did not end the download")
	}
}
