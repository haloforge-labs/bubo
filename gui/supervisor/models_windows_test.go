//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type modelRig struct {
	svc     *ModelService
	dir     string
	profile string
	mu      sync.Mutex
	inputs  ModelInputs
	stall   time.Duration
	ended   chan ModelJob
}

func newModelRig(t *testing.T, mode string) *modelRig {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("GORACE", "atexit_sleep_ms=0")
	t.Setenv("BUBO_FAKE_DIR", dir)
	t.Setenv("BUBO_FAKE_MODE", mode)
	t.Setenv("BUBO_FAKE_FIXTURES", contractDir())
	r := &modelRig{dir: dir, profile: filepath.Join(dir, "profile"), stall: 10 * time.Second, ended: make(chan ModelJob, 8)}
	r.svc = NewModelService(ModelServiceConfig{
		Exe: func() string { return fakeExe },
		Inputs: func() ModelInputs {
			r.mu.Lock()
			defer r.mu.Unlock()
			return ModelInputs{Folders: append([]string(nil), r.inputs.Folders...), DataFolder: r.inputs.DataFolder, DownloadFolders: append([]string(nil), r.inputs.DownloadFolders...), RuntimeDir: r.inputs.RuntimeDir}
		},
		StallTimeout: func() time.Duration { r.mu.Lock(); defer r.mu.Unlock(); return r.stall },
		ListTimeout:  10 * time.Second, CallTimeout: 10 * time.Second, StopTimeout: 3 * time.Second,
		DrainTimeout: 200 * time.Millisecond, Tick: 20 * time.Millisecond,
		LocalAppData: filepath.Join(dir, "local"), UserProfile: r.profile,
		DiskFree: func(path string) (string, uint64, error) { return "C:", 123456789, nil },
		OnJobEnd: func(j ModelJob) {
			select {
			case r.ended <- j:
			default:
			}
		},
	})
	t.Cleanup(r.svc.Close)
	return r
}
func (r *modelRig) setInputs(in ModelInputs) { r.mu.Lock(); r.inputs = in; r.mu.Unlock() }
func (r *modelRig) setStall(d time.Duration) { r.mu.Lock(); r.stall = d; r.mu.Unlock() }

func (r *modelRig) argv(t *testing.T, prefix string, n int) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(r.dir, prefix+"-argv-"+string(rune('0'+n))+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var a []string
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	return a
}

func (r *modelRig) refresh(t *testing.T) ModelsState {
	t.Helper()
	r.svc.Refresh()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		s := r.svc.State()
		if !s.Refreshing && s.Status != ModelsLoading {
			return s
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("the model list did not arrive")
	return ModelsState{}
}

func (r *modelRig) waitJob(t *testing.T, pred func(ModelJob) bool) ModelJob {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if j := r.svc.State().Download; j != nil && pred(*j) {
			return *j
		}
		time.Sleep(10 * time.Millisecond)
	}
	j := r.svc.State().Download
	t.Fatalf("job never reached the state; last: %+v", j)
	return ModelJob{}
}
func ended(j ModelJob) bool {
	return j.State == "done" || j.State == "failed" || j.State == "cancelled"
}

// The recorded list (E21) has the catalogue, five packages, a loose file and no scan warning.
func TestModelListFillsTheFrameState(t *testing.T) {
	r := newModelRig(t, "")
	if err := os.MkdirAll(filepath.Join(r.profile, ".lmstudio", "models"), 0700); err != nil {
		t.Fatal(err)
	}
	first := r.svc.State()
	if first.Status != ModelsLoading || first.Catalogue != nil || first.Download != nil {
		t.Fatalf("before the first read: %+v", first)
	}
	r.setInputs(ModelInputs{Folders: []string{`D:\my-models`}})
	s := r.refresh(t)
	want := e21Models(t)
	if s.Status != ModelsOK || s.Reason != "" || len(s.Catalogue) != len(want.Catalogue) || len(s.Scanned) != len(want.Scanned) || len(s.PreparedModels) != len(want.PreparedModels) || s.Listing == nil || s.UpdatedAt == "" {
		t.Fatalf("state: %+v", s)
	}
	if len(s.ScanWarnings) != len(want.ScanWarnings) || s.Catalogue[0].ID != e21Flash {
		t.Fatalf("rows: %+v", s)
	}
	if s.Disk.Drive != "C:" || s.Disk.FreeBytes == nil || *s.Disk.FreeBytes != 123456789 || s.Disk.Error != "" {
		t.Fatalf("disk: %+v", s.Disk)
	}
	if len(s.SuggestedFolders) != 1 || s.SuggestedFolders[0].Label != "LM Studio's model folder" || !s.SuggestedFolders[0].Exists || s.SuggestedFolders[0].Added {
		t.Fatalf("suggested: %+v", s.SuggestedFolders)
	}
	if len(s.Folders) != 1 || s.Folders[0] != `D:\my-models` {
		t.Fatalf("folders: %v", s.Folders)
	}
	args := r.argv(t, "models", 1)
	if strings.Join(args, " ") != `models list --json --scan-folder D:\my-models` {
		t.Fatalf("list arguments: %v", args)
	}
	// Adding the suggested folder marks it added without another read.
	r.setInputs(ModelInputs{Folders: []string{`D:\my-models`, filepath.Join(r.profile, ".lmstudio", "models")}})
	if got := r.svc.State(); !got.SuggestedFolders[0].Added || len(got.Folders) != 2 {
		t.Fatalf("added flag: %+v", got.SuggestedFolders)
	}
	// The JSON the window reads carries the CLI rows and the null of a missing list.
	b, _ := json.Marshal(s)
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"status", "listing", "catalogue", "prepared_models", "scanned", "scan_warnings", "disk", "suggested_folders", "folders", "download"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("frame key models.%s is missing", key)
		}
	}
}

func TestFirstStartModelListMatchesRuntimeMissing(t *testing.T) {
	r := newModelRig(t, "ui")
	s := r.refresh(t)
	if s.Status != ModelsNotReported || s.ReasonCode != "runtime_missing" || s.ExitCode != 11 ||
		s.Reason != "Run bubo prepare runtime. Reinstall the Bubo package." || len(s.Catalogue) != 0 {
		t.Fatalf("first-start model list: %+v", s)
	}
}

func TestModelListWithoutLMStudioFolderOffersNothing(t *testing.T) {
	r := newModelRig(t, "")
	s := r.refresh(t)
	if s.SuggestedFolders == nil || len(s.SuggestedFolders) != 0 {
		t.Fatalf("suggested: %#v", s.SuggestedFolders)
	}
}

// E19: the download folder is scanned like any other folder; the list no longer
// depends on one --models-root.
func TestModelListScansTheDownloadFolderAndNeverPassesAModelsRoot(t *testing.T) {
	r := newModelRig(t, "")
	folder := t.TempDir()
	r.setInputs(ModelInputs{DataFolder: folder})
	r.refresh(t)
	args := strings.Join(r.argv(t, "models", 1), " ")
	if !strings.Contains(args, "--scan-folder "+folder) || strings.Contains(args, "--models-root") {
		t.Fatalf("arguments: %s", args)
	}
}

func TestRuntimeMissingIsReportedWithItsReason(t *testing.T) {
	r := newModelRig(t, "runtime-missing")
	s := r.refresh(t)
	if s.Status != ModelsNotReported || s.ReasonCode != "runtime_missing" || s.ExitCode != 11 || s.Reason != "Run bubo prepare runtime. Reinstall the Bubo package." {
		t.Fatalf("state: %+v", s)
	}
	if s.Catalogue != nil || s.Scanned != nil || s.Listing != nil {
		t.Fatalf("a list is shown without a CLI answer: %+v", s)
	}
	b, _ := json.Marshal(s)
	if !strings.Contains(string(b), `"catalogue":null`) {
		t.Fatalf("the window needs a null catalogue: %s", b)
	}
	// Disk and the suggested folders are plain OS reads and still arrive.
	if s.Disk.FreeBytes == nil {
		t.Fatal("free disk space should not depend on the runtime")
	}
}

func TestBrokenModelListIsNotShown(t *testing.T) {
	r := newModelRig(t, "models-garbage")
	s := r.refresh(t)
	if s.Status != ModelsNotReported || s.ReasonCode != "protocol_mismatch" || s.Catalogue != nil {
		t.Fatalf("state: %+v", s)
	}
}

func TestMissingCLIIsReported(t *testing.T) {
	r := newModelRig(t, "")
	r.svc.cfg.Exe = func() string { return filepath.Join(r.dir, "missing", "bubo.exe") }
	s := r.refresh(t)
	if s.Status != ModelsNotReported || s.ReasonCode != "cli_unavailable" || s.Reason == "" {
		t.Fatalf("state: %+v", s)
	}
}

func TestRefreshJoinsCallsAndStateNeverBlocks(t *testing.T) {
	r := newModelRig(t, "")
	for i := 0; i < 5; i++ {
		r.svc.Refresh()
	}
	start := time.Now()
	for i := 0; i < 2000; i++ {
		_ = r.svc.State()
	}
	if time.Since(start) > time.Second {
		t.Fatalf("State blocked: %v", time.Since(start))
	}
	r.refresh(t)
	// Five calls make at most a read and the one more read they asked for.
	if _, err := os.Stat(filepath.Join(r.dir, "models-argv-3.json")); err == nil {
		t.Fatal("five Refresh calls started more than two reads")
	}
}

func TestEstimateUsesTheCLIAndValidatesTheSubject(t *testing.T) {
	r := newModelRig(t, "")
	flash := e21Folder(t, e21Flash)
	est, err := r.svc.Estimate(flash, 65536, 1)
	if err != nil || est.State != "estimated" || est.Subject != flash {
		t.Fatalf("%+v %v", est, err)
	}
	args := strings.Join(r.argv(t, "models", 1), " ")
	if args != "models estimate --model "+flash+" --json --context 65536 --sessions 1" {
		t.Fatalf("arguments: %s", args)
	}
	none, err := r.svc.Estimate(e21Folder(t, e21Hauhau), 0, 0)
	if err != nil || none.State != "none_yet" || none.Reason == nil {
		t.Fatalf("%+v %v", none, err)
	}
	for _, bad := range []string{"", "-x", "relative\\path", "a b"} {
		if _, err := r.svc.Estimate(bad, 0, 0); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if _, err := r.svc.Estimate(`S:\bubo-walk\E21\Import\Qwen3.8-27B-UD-Q8_K_XL.gguf`, 0, 0); err != nil {
		t.Fatalf("an absolute path is allowed: %v", err)
	}
}

func TestEstimateRuntimeMissingIsATypedError(t *testing.T) {
	r := newModelRig(t, "runtime-missing")
	_, err := r.svc.Estimate(e21Flash, 0, 0)
	var me *ModelError
	if !errors.As(err, &me) || me.Code != "runtime_missing" || me.ExitCode != 11 {
		t.Fatalf("%#v", err)
	}
}

func TestPlanThenConfirmDownloadsAfterTheUserAgrees(t *testing.T) {
	r := newModelRig(t, "")
	r.refresh(t)
	r.setInputs(ModelInputs{DataFolder: `E:\Bubo data`})
	plan, err := r.svc.StartDownload(e21Flash, "download")
	if err != nil {
		t.Fatal(err)
	}
	recorded := e21Plan(t, e21Flash)
	if plan.ID != e21Flash || plan.Kind != "download" || len(plan.Files) != len(recorded.Files) || !plan.EnoughSpace || len(plan.Licences) != len(recorded.Licences) || plan.Licences[0].Text != recorded.Licences[0].Text || plan.Licence == nil || plan.Licence.PromptLine != e21PromptLine(t, e21Flash) {
		t.Fatalf("plan: %+v", plan)
	}
	job := r.svc.State().Download
	if job == nil || job.State != "awaiting_confirm" || job.Plan == nil || job.Plan.Digest != plan.Digest {
		t.Fatalf("job after the plan: %+v", job)
	}
	// The plan call is plan-only: no download was started.
	if args := strings.Join(r.argv(t, "prepare", 1), " "); args != "prepare model --catalogue-id "+e21Flash+` --jsonl --plan --output E:\Bubo data` {
		t.Fatalf("plan arguments: %s", args)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "prepare-argv-2.json")); err == nil {
		t.Fatal("the download started before the confirm call")
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	done := r.waitJob(t, ended)
	if done.State != "done" || !done.Prepared || done.ManifestPath == nil || done.Error != nil || done.BytesDone != 4 || done.BytesTotal == nil || *done.BytesTotal != 4 || done.Phase != "download" {
		t.Fatalf("job: %+v", done)
	}
	if args := strings.Join(r.argv(t, "prepare", 2), " "); args != "prepare model --catalogue-id "+e21Flash+` --jsonl --output E:\Bubo data` {
		t.Fatalf("download arguments: %s", args)
	}
	select {
	case j := <-r.ended:
		if j.State != "done" {
			t.Fatalf("OnJobEnd: %+v", j)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("OnJobEnd did not run")
	}
	// A finished job can be dismissed; a new plan can start.
	r.svc.DismissJob()
	if r.svc.State().Download != nil {
		t.Fatal("job not dismissed")
	}
}

func TestModelPlanKeepsPlansAfterManyVerificationRecords(t *testing.T) {
	r := newModelRig(t, "model-plan-verification")
	plan, err := r.svc.StartDownload(e21Flash, "download")
	if err != nil || plan.Digest == "" || len(plan.Licences) != len(e21Plan(t, e21Flash).Licences) {
		t.Fatalf("plan after verification progress: %+v, %v", plan, err)
	}
}

func TestConfirmNeedsAPlanForThatModel(t *testing.T) {
	r := newModelRig(t, "")
	if err := r.svc.ConfirmDownload(e21Flash); err == nil {
		t.Fatal("confirm without a plan started a download")
	}
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload("another-model"); err == nil {
		t.Fatal("confirm for another model started a download")
	}
	if err := r.svc.CancelJob(); err != nil {
		t.Fatal(err)
	}
	j := r.svc.State().Download
	if j == nil || j.State != "cancelled" || j.Plan != nil {
		t.Fatalf("job: %+v", j)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err == nil {
		t.Fatal("a cancelled plan was confirmed")
	}
	if _, err := os.Stat(filepath.Join(r.dir, "prepare-argv-2.json")); err == nil {
		t.Fatal("a download ran")
	}
}

func TestConfirmRefusesWhenTheDriveIsTooSmall(t *testing.T) {
	r := newModelRig(t, "")
	os.WriteFile(filepath.Join(r.dir, "volume-short"), nil, 0600)
	plan, err := r.svc.StartDownload(e21Flash, "download")
	// The fake lowers free_bytes of the recorded volume to one byte under its
	// required_bytes; the drive name and the required bytes are the recorded ones.
	recorded := e21Plan(t, e21Flash)
	if err != nil || plan.EnoughSpace || plan.FreeBytes != recorded.RequiredBytes-1 || plan.RequiredBytes != recorded.RequiredBytes || plan.Drive != recorded.Drive {
		t.Fatalf("%+v %v", plan, err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err == nil || !strings.Contains(err.Error(), "not enough free space") {
		t.Fatalf("confirm: %v", err)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "prepare-argv-2.json")); err == nil {
		t.Fatal("a download ran on a full drive")
	}
}

func TestChangedPlanIsDeclined(t *testing.T) {
	r := newModelRig(t, "model-plan-changed")
	// The plan call itself shows digest c; the run offers d.
	t.Setenv("BUBO_FAKE_MODE", "")
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_MODE", "model-plan-changed")
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	if j.State != "failed" || j.Error == nil || j.Error.Code != "plan_changed" || j.Prepared {
		t.Fatalf("job: %+v", j)
	}
}

func TestDownloadFailureKeepsTheCLIWords(t *testing.T) {
	r := newModelRig(t, "model-error")
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	if j.State != "failed" || j.Error == nil || j.Error.Code != "storage_failed" || j.Error.Remedy != "The drive is full. Free space and start again." || j.Error.ExitCode != 7 || j.Prepared {
		t.Fatalf("job: %+v", j)
	}
}

func TestRemovedEntryFailsAtThePlanWithTheCLICode(t *testing.T) {
	r := newModelRig(t, "")
	_, err := r.svc.StartDownload(e21Disabled, "download")
	var me *ModelError
	if !errors.As(err, &me) || me.Code != "invalid_arguments" || me.ExitCode != 2 {
		t.Fatalf("%#v", err)
	}
	if j := r.svc.State().Download; j == nil || j.State != "failed" || j.Error == nil {
		t.Fatalf("job: %+v", j)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "prepare-argv-2.json")); err == nil {
		t.Fatal("a download ran")
	}
}

func TestStartRejectsBadIDsAndKinds(t *testing.T) {
	r := newModelRig(t, "")
	for _, id := range []string{"", "-x", "a b", `a\b`} {
		if _, err := r.svc.StartDownload(id, "download"); err == nil {
			t.Errorf("%q accepted", id)
		}
		if err := r.svc.Check(id); err == nil {
			t.Errorf("check %q accepted", id)
		}
	}
	if _, err := r.svc.StartDownload(e21Flash, "delete"); err == nil {
		t.Error("unknown kind accepted")
	}
	if _, err := os.Stat(filepath.Join(r.dir, "prepare-argv-1.json")); err == nil {
		t.Fatal("the CLI ran for a bad id")
	}
}

func TestCheckRunsOfflineAndNeedsNoConfirm(t *testing.T) {
	r := newModelRig(t, "")
	if err := r.svc.Check(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	if j.State != "done" || j.Kind != "check" || j.Phase != "verify" || !strings.Contains(j.Message, "checks passed") {
		t.Fatalf("job: %+v", j)
	}
	args := strings.Join(r.argv(t, "prepare", 1), " ")
	if !strings.Contains(args, "--offline") || !strings.Contains(args, "--verify") || strings.Contains(args, "--plan") {
		t.Fatalf("arguments: %s", args)
	}
}

func TestRepairIsPlannedThenVerifiedWithFullChecks(t *testing.T) {
	r := newModelRig(t, "")
	plan, err := r.svc.StartDownload(e21Flash, "repair")
	if err != nil || plan.Kind != "repair" {
		t.Fatalf("%+v %v", plan, err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	args := strings.Join(r.argv(t, "prepare", 2), " ")
	if j.State != "done" || !strings.Contains(args, "--verify") || strings.Contains(args, "--offline") {
		t.Fatalf("%+v / %s", j, args)
	}
}

func TestOnlyOneModelActionAtATime(t *testing.T) {
	r := newModelRig(t, "model-stall")
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	r.waitJob(t, func(j ModelJob) bool { return j.State == "running" && j.Phase == "download" })
	if _, err := r.svc.StartDownload("other", "download"); err == nil {
		t.Fatal("a second plan started while a download runs")
	}
	if err := r.svc.Check(e21Flash); err == nil {
		t.Fatal("a check started while a download runs")
	}
	if err := r.svc.CancelJob(); err != nil {
		t.Fatal(err)
	}
	if j := r.waitJob(t, ended); j.State != "cancelled" {
		t.Fatalf("job: %+v", j)
	}
}

func TestProgressClearsTheStallWarning(t *testing.T) {
	r := newModelRig(t, "")
	os.WriteFile(filepath.Join(r.dir, "model-steps"), []byte("12"), 0600)
	r.setStall(5 * time.Second)
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	if j := r.waitJob(t, ended); j.State != "done" || j.Stalled {
		t.Fatalf("job: %+v", j)
	}
}

func TestCloseEndsARunningDownload(t *testing.T) {
	r := newModelRig(t, "model-stall")
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	r.waitJob(t, func(j ModelJob) bool { return j.Phase == "download" })
	done := make(chan struct{})
	go func() { r.svc.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not return")
	}
	if _, err := r.svc.StartDownload(e21Flash, "download"); err == nil {
		t.Fatal("a closed service started a plan")
	}
}

func TestDiskFreeReadsTheVolumeOfAFolderThatDoesNotExistYet(t *testing.T) {
	base := t.TempDir()
	drive, free, err := diskFreeWindows(filepath.Join(base, "not", "yet", "there"))
	if err != nil || free == 0 || drive != filepath.VolumeName(base) {
		t.Fatalf("%q %d %v", drive, free, err)
	}
	if _, err := os.Stat(filepath.Join(base, "not")); err == nil {
		t.Fatal("the read created a folder")
	}
}

func TestSettingsKeepModelFolders(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	s := DefaultSettings()
	s.ModelFolders = []string{`D:\Models`, `E:\more\models`}
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	got, warning := LoadSettings(path)
	if warning != "" || len(got.ModelFolders) != 2 || got.ModelFolders[1] != `E:\more\models` {
		t.Fatalf("%+v %q", got, warning)
	}
	for _, bad := range [][]string{{`C:\`}, {`relative`}, {`D:\Models\`}, make([]string, MaxModelFolders+1)} {
		s.ModelFolders = bad
		if err := SaveSettings(path, s); err == nil {
			t.Errorf("saved %v", bad)
		}
	}
	// A damaged folder list is reported as damaged settings, never half used.
	os.WriteFile(path, []byte(`{"version":1,"port":8080,"profile":"default","context":262144,"sessions":1,"model_folders":["C:\\"]}`), 0600)
	if got, warning := LoadSettings(path); warning == "" || len(got.ModelFolders) != 0 {
		t.Fatalf("%+v %q", got, warning)
	}
}
