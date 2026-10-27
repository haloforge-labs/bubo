//go:build windows

package supervisor

// E19: the download folder decides only where NEW downloads go. Every prepared
// package stays known by its manifest path, in every download folder used and
// in the Import folders, and the CLI lists them through --scan-folder.
//
// The tests below were written before the code (see the failure list in the
// E19 report). They run the supervisor against the fake CLI in package mode
// (BUBO_FAKE_PACKAGES=1), which keeps packages on a real disk and follows
// --output, --models-root and --scan-folder the way the real CLI does.

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

func packageApp(t *testing.T) *modelsApp {
	t.Helper()
	m := newModelsApp(t, "")
	t.Setenv("BUBO_FAKE_PACKAGES", "1")
	return m
}

func patchDataFolder(a *AppService, path string) ActionResult {
	return a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{DataFolder: &path}})
}
func useDefaultDataFolder(a *AppService) ActionResult {
	return a.PatchPreferences(PreferencesPatch{Clear: []string{"data_folder"}})
}

func (m *modelsApp) savedSettings() Settings {
	m.a.mu.RLock()
	defer m.a.mu.RUnlock()
	return m.a.settings
}

func (m *modelsApp) argvs(t *testing.T, prefix, verb string) [][]string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(m.dir, prefix+"-argv-*.json"))
	if err != nil {
		t.Fatal(err)
	}
	type numbered struct {
		n    int
		args []string
	}
	var rows []numbered
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var args []string
		if err := json.Unmarshal(data, &args); err != nil {
			t.Fatal(err)
		}
		base := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(path), prefix+"-argv-"), ".json")
		n, err := strconv.Atoi(base)
		if err != nil {
			t.Fatal(err)
		}
		if verb == "" || (len(args) > 1 && args[1] == verb) || (len(args) > 0 && args[0] == verb) {
			rows = append(rows, numbered{n, args})
		}
	}
	for i := range rows {
		for j := i + 1; j < len(rows); j++ {
			if rows[j].n < rows[i].n {
				rows[i], rows[j] = rows[j], rows[i]
			}
		}
	}
	var out [][]string
	for _, row := range rows {
		out = append(out, row.args)
	}
	return out
}

func hasPair(args []string, name, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name && SameFolder(args[i+1], value) {
			return true
		}
	}
	return false
}

func (m *modelsApp) defaultID(t *testing.T) string {
	t.Helper()
	s := m.waitModels(t, listed)
	for _, entry := range s.Catalogue {
		if entry.Role == "recommended" && entry.DefaultStatus == "default" {
			return entry.ID
		}
	}
	t.Fatal("the fake catalogue has no default entry")
	return ""
}

func (m *modelsApp) refreshed(t *testing.T) *ViewModelsState {
	t.Helper()
	if err := actionTestError(m.a.RefreshModels()); err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	return m.waitModels(t, listed)
}

// download plans and confirms a catalogue download and waits for the result.
func (m *modelsApp) download(t *testing.T, id string) ModelPlan {
	t.Helper()
	result := m.a.Download(id)
	if result.Error != nil {
		t.Fatalf("plan: %+v", result.Error)
	}
	if err := actionTestError(m.a.ConfirmDownload(id)); err != nil {
		t.Fatal(err)
	}
	m.waitModels(t, func(s *ViewModelsState) bool {
		return s.Download != nil && (s.Download.State == "done" || s.Download.State == "failed")
	})
	if job := m.a.GetView().Models.Download; job == nil || job.State != "done" {
		t.Fatalf("download did not finish: %+v", job)
	}
	m.a.DismissModelJob()
	return *result.Plan
}

func preparedFor(s *ViewModelsState, id string) []PreparedModel {
	var rows []PreparedModel
	for _, row := range s.PreparedModels {
		if row.ID == id {
			rows = append(rows, row)
		}
	}
	return rows
}

func manifestOf(row PreparedModel) string {
	if row.ManifestPath == nil {
		return ""
	}
	return *row.ManifestPath
}

func setFreeSpace(t *testing.T, dir string, volume string, bytes uint64) {
	t.Helper()
	b, _ := json.Marshal(map[string]uint64{strings.ToUpper(volume): bytes})
	if err := os.WriteFile(filepath.Join(dir, "free.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// Failure list 1-5: choosing the folder

func TestDownloadFolderPickerCancelledChangesNothing(t *testing.T) {
	m := packageApp(t)
	m.dialogs.folder = ""
	result := m.a.BrowseDownloadFolder()
	if !result.Cancelled || result.Error != nil || result.Folder != "" {
		t.Fatalf("a cancelled picker: %+v", result)
	}
	if s := m.savedSettings(); s.DataFolder != "" || len(s.DownloadFolders) != 0 {
		t.Fatalf("a cancelled picker saved: %+v", s)
	}
	if len(m.dialogs.calls) != 1 || !strings.HasPrefix(m.dialogs.calls[0], "folder:") {
		t.Fatalf("dialog calls: %v", m.dialogs.calls)
	}
}

func TestDownloadFolderRefusesAnUnusablePath(t *testing.T) {
	m := packageApp(t)
	file := filepath.Join(m.dir, "a-file")
	if err := os.WriteFile(file, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{
		"a drive root":    `D:\`,
		"relative":        `models`,
		"leading hyphen":  `-D:\x`,
		"missing":         filepath.Join(m.dir, "missing"),
		"a file":          file,
		"a UNC share":     `\\server\share`,
		"control letters": "C:\\a\nb",
	} {
		t.Run(name, func(t *testing.T) {
			m.dialogs.folder = path
			result := m.a.BrowseDownloadFolder()
			if result.Error == nil || result.Error.Code != "download_folder_invalid" {
				t.Fatalf("%s: %+v", path, result)
			}
			if got := m.savedSettings().DataFolder; got != "" {
				t.Fatalf("saved %q", got)
			}
			// The same refusal on the preference route, so no caller bypasses it.
			if r := patchDataFolder(m.a, path); r.Error == nil || r.Error.Code != "download_folder_invalid" {
				t.Fatalf("PatchPreferences accepted %q: %+v", path, r)
			}
		})
	}
}

func TestDownloadFolderMustBeWritable(t *testing.T) {
	m := packageApp(t)
	folder := m.folder(t, "readonly")
	previous := folderWritable
	folderWritable = func(string) error { return errors.New("access is denied") }
	defer func() { folderWritable = previous }()
	m.dialogs.folder = folder
	result := m.a.BrowseDownloadFolder()
	if result.Error == nil || result.Error.Code != "download_folder_unwritable" {
		t.Fatalf("an unwritable folder: %+v", result)
	}
	if got := m.savedSettings().DataFolder; got != "" {
		t.Fatalf("saved %q", got)
	}
}

func TestDownloadFolderWritableProbeLeavesNoFile(t *testing.T) {
	folder := t.TempDir()
	if err := folderWritable(folder); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(folder)
	if len(entries) != 0 {
		t.Fatalf("the probe left %d entries", len(entries))
	}
}

func TestDownloadFolderIsLockedWhileServingOrDownloading(t *testing.T) {
	for name, arrange := range map[string]func(m *modelsApp){
		"serving":         func(m *modelsApp) { m.b.view.State = Ready },
		"starting":        func(m *modelsApp) { m.b.view.State = Starting },
		"attached":        func(m *modelsApp) { m.b.view.State = AttachedExternal },
		"preparing":       func(m *modelsApp) { m.b.view.State = Preparing },
		"download":        func(m *modelsApp) { m.a.models.mu.Lock(); m.a.models.job = &ModelJob{ID: "x", State: "running"}; m.a.models.mu.Unlock() },
		"download paused": func(m *modelsApp) { m.a.models.mu.Lock(); m.a.models.job = &ModelJob{ID: "x", State: "cancelling"}; m.a.models.mu.Unlock() },
	} {
		t.Run(name, func(t *testing.T) {
			m := packageApp(t)
			folder := m.folder(t, "target")
			current := m.folder(t, "current")
			m.a.mu.Lock()
			m.a.settings.DataFolder = current
			m.a.mu.Unlock()
			arrange(m)
			view := m.a.GetView()
			change := view.DownloadFolder.Change
			if change.Allowed || change.ReasonCode != "download_folder_locked" || change.Reason == "" {
				t.Fatalf("availability: %+v", change)
			}
			for _, result := range []ActionResult{patchDataFolder(m.a, folder), useDefaultDataFolder(m.a)} {
				if result.Error == nil || result.Error.Code != "download_folder_locked" {
					t.Fatalf("a change while %s: %+v", name, result)
				}
			}
			m.dialogs.folder = folder
			if r := m.a.BrowseDownloadFolder(); r.Error == nil || r.Error.Code != "download_folder_locked" {
				t.Fatalf("the picker opened while %s: %+v", name, r)
			}
			if got := m.savedSettings().DataFolder; got != current {
				t.Fatalf("saved %q, want %q", got, current)
			}
		})
	}
}

func TestDownloadFolderIsOpenWhenIdleOrWaitingForConsent(t *testing.T) {
	m := packageApp(t)
	m.b.view.State = Stopped
	m.a.models.mu.Lock()
	m.a.models.job = &ModelJob{ID: "x", State: "awaiting_confirm"}
	m.a.models.mu.Unlock()
	if change := m.a.GetView().DownloadFolder.Change; !change.Allowed {
		t.Fatalf("the consent dialog changes the folder and re-plans: %+v", change)
	}
}

// ---------------------------------------------------------------------------
// Settings and the View

func TestDownloadFolderSavedRememberedAndShownInTheView(t *testing.T) {
	m := packageApp(t)
	m.waitModels(t, listed)
	view := m.a.GetView()
	// The default folder is the one the CLI names; the GUI keeps no copy of it.
	if !view.DownloadFolder.IsDefault || !SameFolder(view.DownloadFolder.Path, filepath.Join(os.Getenv("LOCALAPPDATA"), "Bubo", "models")) || view.DownloadFolder.FreeBytes == nil {
		t.Fatalf("the default folder: %+v", view.DownloadFolder)
	}
	d1, d2 := m.folder(t, "d1"), m.folder(t, "d2")
	m.dialogs.folder = d1
	if r := m.a.BrowseDownloadFolder(); r.Error != nil || r.Cancelled || !SameFolder(r.Folder, d1) {
		t.Fatalf("picked: %+v", r)
	}
	if r := patchDataFolder(m.a, d2+`\`); r.Error != nil {
		t.Fatal(r.Error)
	}
	saved := m.savedSettings()
	if !SameFolder(saved.DataFolder, d2) || len(saved.DownloadFolders) != 2 || !SameFolder(saved.DownloadFolders[0], d1) || !SameFolder(saved.DownloadFolders[1], d2) {
		t.Fatalf("saved: %+v", saved)
	}
	view = m.a.GetView()
	if view.DownloadFolder.IsDefault || !SameFolder(view.DownloadFolder.Path, d2) || !view.DownloadFolder.Change.Allowed {
		t.Fatalf("view: %+v", view.DownloadFolder)
	}
	// Choosing the same folder twice remembers it once.
	if r := patchDataFolder(m.a, d2); r.Error != nil {
		t.Fatal(r.Error)
	}
	if got := len(m.savedSettings().DownloadFolders); got != 2 {
		t.Fatalf("%d remembered folders", got)
	}
	// Use default keeps the list: models downloaded there stay known.
	if r := useDefaultDataFolder(m.a); r.Error != nil {
		t.Fatal(r.Error)
	}
	saved = m.savedSettings()
	view = m.a.GetView()
	if saved.DataFolder != "" || len(saved.DownloadFolders) != 2 || !view.DownloadFolder.IsDefault {
		t.Fatalf("after Use default: %+v %+v", saved, view.DownloadFolder)
	}
	// The setting survives a restart of the app.
	loaded, warning := LoadSettings(m.settings)
	if warning != "" || len(loaded.DownloadFolders) != 2 {
		t.Fatalf("reloaded: %+v %q", loaded, warning)
	}
}

func TestSettingsFileRejectsDamagedDownloadFolders(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"download_folders":["relative\\path"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, warning := LoadSettings(path); warning == "" {
		t.Fatal("a relative download folder loaded")
	}
	s := DefaultSettings()
	s.DownloadFolders = []string{`D:\`}
	if err := SaveSettings(path, s); err == nil {
		t.Fatal("a drive root saved")
	}
}

func TestDownloadFolderDiskFollowsTheTargetAtOnce(t *testing.T) {
	m := packageApp(t)
	m.waitModels(t, listed)
	d1 := m.folder(t, "disk-target")
	if r := patchDataFolder(m.a, d1); r.Error != nil {
		t.Fatal(r.Error)
	}
	// Before the next list arrives the old drive's free space must not be shown
	// under the new folder.
	disk := m.a.models.State().Disk
	if !SameFolder(disk.Path, d1) {
		t.Fatalf("disk: %+v", disk)
	}
}

// ---------------------------------------------------------------------------
// What the CLI is asked to scan

func TestModelListScansEveryDownloadFolderAndImportFolder(t *testing.T) {
	m := packageApp(t)
	d1, d2, imported := m.folder(t, "d1"), m.folder(t, "d2"), m.folder(t, "imported")
	if _, _, err := m.a.addModelFolder(imported); err != nil {
		t.Fatal(err)
	}
	patchDataFolder(m.a, d1)
	patchDataFolder(m.a, d2)
	useDefaultDataFolder(m.a)
	m.refreshed(t)
	lists := m.argvs(t, "models", "list")
	args := lists[len(lists)-1]
	for _, want := range []string{imported, d1, d2} {
		if !hasPair(args, "--scan-folder", want) {
			t.Fatalf("the list did not scan %s: %v", want, args)
		}
	}
	// The GUI no longer depends on one models root.
	for _, a := range args {
		if a == "--models-root" {
			t.Fatalf("--models-root is still passed: %v", args)
		}
	}
}

func TestRememberedFolderTheUserDeletedIsNotScannedAndNotAWarning(t *testing.T) {
	m := packageApp(t)
	gone := filepath.Join(m.dir, "deleted download folder") // its drive is there
	m.a.mu.Lock()
	m.a.settings.DownloadFolders = []string{gone}
	m.a.mu.Unlock()
	s := m.refreshed(t)
	lists := m.argvs(t, "models", "list")
	if hasPair(lists[len(lists)-1], "--scan-folder", gone) {
		t.Fatalf("a deleted folder was scanned: %v", lists[len(lists)-1])
	}
	if len(s.ScanWarnings) != 0 {
		t.Fatalf("warnings: %+v", s.ScanWarnings)
	}
}

func TestRememberedFolderOnAnUnplugedDriveStillWarns(t *testing.T) {
	// A model on a USB drive must not vanish silently: the folder stays in the
	// scan so the CLI reports it, and the Models page shows the warning.
	m := packageApp(t)
	gone := `Q:\Bubo downloads`
	if _, err := os.Stat(`Q:\`); err == nil {
		t.Skip("drive Q: exists on this machine")
	}
	m.a.mu.Lock()
	m.a.settings.DownloadFolders = []string{gone}
	m.a.mu.Unlock()
	s := m.refreshed(t)
	lists := m.argvs(t, "models", "list")
	if !hasPair(lists[len(lists)-1], "--scan-folder", gone) {
		t.Fatalf("an absent drive's folder was dropped: %v", lists[len(lists)-1])
	}
	if len(s.ScanWarnings) != 1 {
		t.Fatalf("warnings: %+v", s.ScanWarnings)
	}
}

// ---------------------------------------------------------------------------
// Failure list 8: the first download into a chosen folder, end to end

func TestFirstDownloadGoesToTheChosenFolderOnly(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1 := m.folder(t, "d1")
	patchDataFolder(m.a, d1)
	plan := m.download(t, id)
	want := filepath.Join(d1, "models", id)
	if got := plan.Paths["model"]; !SameFolder(got, want) {
		t.Fatalf("the plan names %q, want %q", got, want)
	}
	if plan.Drive != strings.ToUpper(filepath.VolumeName(d1)) || !plan.EnoughSpace || plan.RequiredBytes < 110*(1<<30) {
		t.Fatalf("plan: %+v", plan)
	}
	if _, err := os.Stat(filepath.Join(want, "bubo-package.json")); err != nil {
		t.Fatal("the package was not written to the chosen folder", err)
	}
	if _, err := os.Stat(filepath.Join(os.Getenv("LOCALAPPDATA"), "Bubo", "models", id)); err == nil {
		t.Fatal("a download wrote to the default folder as well")
	}
	if !hasPair(m.argvs(t, "prepare", "")[0], "--output", d1) {
		t.Fatalf("prepare arguments: %v", m.argvs(t, "prepare", ""))
	}
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 1 || !SameFolder(manifestOf(rows[0]), filepath.Join(want, "bubo-package.json")) || rows[0].PreparedState != "prepared" {
		t.Fatalf("rows: %+v", rows)
	}
}

func TestPlanOnAFullDriveIsShortAndNothingIsWritten(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1 := m.folder(t, "full")
	patchDataFolder(m.a, d1)
	setFreeSpace(t, m.dir, filepath.VolumeName(d1), 10<<30)
	result := m.a.Download(id)
	if result.Error != nil || result.Plan == nil {
		t.Fatalf("plan: %+v", result)
	}
	if result.Plan.EnoughSpace || result.Plan.FreeBytes != 10<<30 || result.Plan.RequiredBytes <= result.Plan.FreeBytes {
		t.Fatalf("plan: %+v", result.Plan)
	}
	if err := m.a.ConfirmDownload(id); err.Error == nil || err.Error.Code != "insufficient_space" {
		t.Fatalf("a short plan was confirmed: %+v", err)
	}
	if _, statErr := os.Stat(filepath.Join(d1, "models")); statErr == nil {
		t.Fatal("files were written to a full drive")
	}
	// Changing the folder re-plans on the other drive state: the shortfall goes.
	setFreeSpace(t, m.dir, filepath.VolumeName(d1), 500<<30)
	if r := m.a.Download(id); r.Plan == nil || !r.Plan.EnoughSpace {
		t.Fatalf("re-plan: %+v", r)
	}
}

// ---------------------------------------------------------------------------
// Nothing disappears when the folder changes

func TestModelDownloadedToD1StaysListedAndStartableAfterTheFolderChanges(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1 := m.folder(t, "d1")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	manifest := filepath.Join(d1, "models", id, "bubo-package.json")

	// Change the folder back to the default.
	if r := useDefaultDataFolder(m.a); r.Error != nil {
		t.Fatal(r.Error)
	}
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 1 || !SameFolder(manifestOf(rows[0]), manifest) {
		t.Fatalf("the model downloaded to D1 disappeared: %+v", rows)
	}
	// Selecting it starts the server with its manifest.
	if r := m.a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{ManifestPath: &manifest}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	args := startArgs(m.savedSettings())
	if !hasPair(args, "--manifest", manifest) {
		t.Fatalf("serve arguments: %v", args)
	}
	// A new download goes to the default folder, not to D1.
	plan := m.a.Download(id).Plan
	if plan == nil || !SameFolder(plan.Paths["model"], filepath.Join(os.Getenv("LOCALAPPDATA"), "Bubo", "models", id)) {
		t.Fatalf("plan: %+v", plan)
	}
}

func TestSamePackageInTwoFoldersGivesTwoRowsEachWithItsFolder(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1, d2 := m.folder(t, "d1"), m.folder(t, "d2")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	patchDataFolder(m.a, d2)
	m.download(t, id)
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 2 {
		t.Fatalf("rows: %+v", rows)
	}
	got := []string{manifestOf(rows[0]), manifestOf(rows[1])}
	want := []string{filepath.Join(d1, "models", id, "bubo-package.json"), filepath.Join(d2, "models", id, "bubo-package.json")}
	match := func(a, b string) bool { return SameFolder(a, b) }
	if !(match(got[0], want[0]) && match(got[1], want[1]) || match(got[0], want[1]) && match(got[1], want[0])) {
		t.Fatalf("manifests %v, want %v", got, want)
	}
}

func TestImportedPackageFolderIsOnePackageRowNotShards(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	imported := m.folder(t, filepath.Join("imports", id))
	marker, _ := json.Marshal(map[string]any{"package_id": id, "size_bytes": 7, "files": []string{"model.gguf"}})
	if err := os.WriteFile(filepath.Join(imported, "bubo-package.json"), marker, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(imported, "model.gguf"), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.a.addModelFolder(filepath.Dir(imported)); err != nil {
		t.Fatal(err)
	}
	s := m.refreshed(t)
	rows := preparedFor(s, id)
	if len(rows) != 1 || !SameFolder(manifestOf(rows[0]), filepath.Join(imported, "bubo-package.json")) {
		t.Fatalf("rows: %+v", rows)
	}
	if len(s.Scanned) != 0 {
		t.Fatalf("the package's files were listed as loose models: %+v", s.Scanned)
	}
}

// ---------------------------------------------------------------------------
// Partial packages and repair

func TestPartialPackageInAnotherFolderIsListedPartialAndNotRunnable(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1, d2 := m.folder(t, "d1"), m.folder(t, "d2")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	if err := os.Remove(filepath.Join(d1, "models", id, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	patchDataFolder(m.a, d2)
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 1 || rows[0].PreparedState != "partial" || !SameFolder(manifestOf(rows[0]), filepath.Join(d1, "models", id, "bubo-package.json")) {
		t.Fatalf("a package with a missing file: %+v", rows)
	}
	// The CLI itself refuses to serve it, so the GUI never offers Start for it.
	out, err := exec.Command(fakeExe, "serve", "--manifest", manifestOf(rows[0]), "--jsonl").CombinedOutput()
	if err == nil || !strings.Contains(string(out), "model_missing") {
		t.Fatalf("serve --manifest of a partial package: %v %s", err, out)
	}
}

// The CLI names the --output that repairs a package (repair_output). The GUI
// passes it back unchanged and does no folder arithmetic.
func TestRepairOfAPackageUsesTheOutputItsRowNames(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1, d2 := m.folder(t, "d1"), m.folder(t, "d2")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	if err := os.Remove(filepath.Join(d1, "models", id, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	patchDataFolder(m.a, d2) // the download folder is D2 now; the broken package is in D1
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 1 || rows[0].RepairOutput == nil || !SameFolder(*rows[0].RepairOutput, d1) {
		t.Fatalf("the row does not carry the repair output: %+v", rows)
	}
	result := m.a.RepairPackage(id, *rows[0].RepairOutput)
	if result.Error != nil || result.Plan == nil || !SameFolder(result.Plan.Paths["model"], filepath.Join(d1, "models", id)) {
		t.Fatalf("repair plan: %+v", result)
	}
	prepares := m.argvs(t, "prepare", "")
	if last := prepares[len(prepares)-1]; !hasPair(last, "--output", d1) || hasPair(last, "--output", d2) {
		t.Fatalf("repair arguments: %v", last)
	}
	if err := actionTestError(m.a.ConfirmDownload(id)); err != nil {
		t.Fatal(err)
	}
	m.waitModels(t, func(s *ViewModelsState) bool { return s.Download != nil && s.Download.State == "done" })
	prepares = m.argvs(t, "prepare", "")
	if run := prepares[len(prepares)-1]; !hasPair(run, "--output", d1) {
		t.Fatalf("repair run arguments: %v", run)
	}
	if rows := preparedFor(m.refreshed(t), id); len(rows) != 1 || rows[0].PreparedState != "prepared" {
		t.Fatalf("after repair: %+v", rows)
	}
}

func TestPackageWithoutARepairOutputCannotBeRepairedAndAnyOtherOutputIsRefused(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	imported := m.folder(t, filepath.Join("imports", "somewhere else"))
	marker, _ := json.Marshal(map[string]any{"package_id": id, "size_bytes": 7, "files": []string{"model.gguf"}})
	if err := os.WriteFile(filepath.Join(imported, "bubo-package.json"), marker, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(imported, "model.gguf"), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.a.addModelFolder(filepath.Dir(imported)); err != nil {
		t.Fatal(err)
	}
	rows := preparedFor(m.refreshed(t), id)
	if len(rows) != 1 || rows[0].RepairOutput != nil {
		t.Fatalf("a package outside prepare's layout must carry no repair output: %+v", rows)
	}
	for _, output := range []string{m.dir, filepath.Dir(imported), `C:\elsewhere`} {
		if r := m.a.RepairPackage(id, output); r.Error == nil || r.Error.Code != "model_folder_not_allowed" {
			t.Fatalf("output %q: %+v", output, r)
		}
	}
}

// The catalogue rows are read by the CLI at its default folder, so a repair or
// a check of a catalogue row acts there. It never follows the download folder:
// the GUI does not know another package's root (see the report).
func TestRepairActsOnTheDefaultFolderWhateverTheDownloadFolderIs(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	m.download(t, id) // the default folder
	defaultPackage := filepath.Join(os.Getenv("LOCALAPPDATA"), "Bubo", "models", id)
	if err := os.Remove(filepath.Join(defaultPackage, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	d2 := m.folder(t, "d2")
	patchDataFolder(m.a, d2)
	s := m.refreshed(t)
	state := ""
	for _, entry := range s.Catalogue {
		if entry.ID == id {
			state = entry.PreparedState
		}
	}
	if state != "partial" {
		t.Fatalf("catalogue state %q", state)
	}
	result := m.a.CheckModel(id, true)
	if result.Error != nil || result.Plan == nil || !SameFolder(result.Plan.Paths["model"], defaultPackage) {
		t.Fatalf("repair plan: %+v", result)
	}
	prepares := m.argvs(t, "prepare", "")
	for _, a := range prepares[len(prepares)-1] {
		if a == "--output" {
			t.Fatalf("repair passed --output: %v", prepares[len(prepares)-1])
		}
	}
}

// ---------------------------------------------------------------------------
// Estimate and Start use the package path

func TestSavedEstimateOfASelectedPackageUsesItsFolder(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1 := m.folder(t, "d1")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	useDefaultDataFolder(m.a)
	folder := filepath.Join(d1, "models", id)
	manifest := filepath.Join(folder, "bubo-package.json")
	st := actionTestInput(Stopped).Snapshot
	st.Status.MemoryEstimate = nil
	other := filepath.Join(m.dir, "default", "bubo-package.json")
	st.Status.PreparedDefault = &other
	m.b.view = st
	m.a.mu.Lock()
	m.a.settings.ManifestPath = manifest
	m.a.mu.Unlock()
	v := waitSavedEstimate(t, m.a)
	if v.SavedEstimate.Error != nil && v.SavedEstimate.Error.Code == "estimate_needs_restart" {
		t.Fatalf("a package in another folder cannot be estimated: %+v", v.SavedEstimate.Error)
	}
	estimates := recordedEstimates(t, m.dir)
	if len(estimates) == 0 || !reflect.DeepEqual(estimates[len(estimates)-1][:4], []string{"models", "estimate", "--model", filepath.Clean(folder)}) {
		t.Fatalf("estimate arguments: %v", estimates)
	}
	// Flash Next keeps its measured table, so the verdict is a real "Fits."
	if v.SavedEstimate.MemoryEstimate == nil || v.SavedEstimate.MemoryEstimate.Fits == nil {
		t.Fatalf("no verdict: %+v", v.SavedEstimate)
	}
}

func TestEstimateByFolderGivesTheMeasuredTable(t *testing.T) {
	m := packageApp(t)
	id := m.defaultID(t)
	d1 := m.folder(t, "d1")
	patchDataFolder(m.a, d1)
	m.download(t, id)
	est, err := m.a.EstimateModel(filepath.Join(d1, "models", id), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if est.MemoryEstimate == nil || est.MemoryEstimate.fitVerdict() == nil {
		t.Fatalf("a Flash Next package folder has no verdict: %+v", est)
	}
}
