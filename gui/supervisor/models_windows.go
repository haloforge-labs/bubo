//go:build windows

package supervisor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/windows"
)

// ModelService runs the model calls of the Models page: `models list`,
// `models estimate` and `prepare model --catalogue-id` (plan, download,
// repair, check). It keeps the latest answer for the frame key `models`.
//
// Rules:
//   - Every CLI call runs in an owned process (Job Object) with a deadline.
//   - State() only copies cached data under a lock. It never starts a process,
//     reads the disk or waits, so the polling loop of the window stays fast.
//   - The service never scans a drive. It passes only the folders the user
//     added. The CLI applies the scan rules.
//   - A download needs two calls: StartDownload shows the plan, ConfirmDownload
//     starts it. The approval line sent to the CLI is bound to the digest of
//     the plan the user saw.
//   - A silent download is stopped after the deadline so Retry can start a
//     fresh plan. A server token stall has a separate warning-only watchdog.

// (The old 25 s cap is gone: it was shorter than the CLI's own silent steps.)
// downloadStallFloor is the least silence that counts as a stall. The CLI has
// silent steps of its own: connect 30 s, low speed 60 s, Retry-After up to 60 s.
// A variable only so tests can shorten it.
var downloadStallFloor = 90 * time.Second

// stallLimit is the silence that stops a download: the user's stall setting,
// never below the floor. Zero or less leaves the watchdog off.
func stallLimit(user time.Duration) time.Duration {
	if user <= 0 {
		return 0
	}
	return max(user, downloadStallFloor)
}

type ModelService struct {
	cfg ModelServiceConfig

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu         sync.RWMutex
	state      ModelsState // lists and disk; Folders, Download and Added are filled in State()
	job        *ModelJob
	jobProc    *OwnedProcess
	jobOp      string
	pendingRun *pendingRun
	refreshing bool
	again      bool
	closed     bool
	updating   bool
	processes  map[*OwnedProcess]struct{}
	// diskFor is the download folder (data folder) that state.Disk was read for.
	diskFor string
}

// ModelInputs is what the service needs from the saved settings. The window
// changes it when the user adds or removes a folder or changes the data folder.
type ModelInputs struct {
	// Folders are the Import folders the user added.
	Folders []string
	// DataFolder is where new downloads go ("" = the CLI's default folder).
	DataFolder string
	// DownloadFolders are the download folders used so far. Every one is
	// scanned, so a package downloaded before a folder change stays listed.
	DownloadFolders []string
	RuntimeDir      string
}

type ModelServiceConfig struct {
	Events func(StreamEvent)
	// Exe returns the absolute path of the packaged bubo.exe.
	Exe func() string
	// Inputs returns the saved folders and the data folder.
	Inputs func() ModelInputs
	// Keys scrubs a stored API key from CLI text. May be nil.
	Keys *KeyManager
	// StallTimeout is the no-progress warning time; default 120 s.
	StallTimeout func() time.Duration
	// ListTimeout bounds `models list` (default 90 s); CallTimeout bounds plan
	// and estimate (default 60 s); StopTimeout is the wait after Cancel before
	// the owned process is ended (default 45 s).
	ListTimeout, CallTimeout, StopTimeout, DrainTimeout time.Duration
	// Tick is the stall watchdog period (default 1 s).
	Tick time.Duration
	// LocalAppData and UserProfile locate the default model folder and the
	// folders of other programs. Default: the environment.
	LocalAppData, UserProfile string
	// DiskFree reads the free space for a path (default: Windows).
	DiskFree func(path string) (drive string, free uint64, err error)
	// OnJobEnd runs once when a job ends (the window uses it to refresh).
	OnJobEnd func(ModelJob)
}

type pendingRun struct {
	id, kind, digest, dataFolder, runtimeDir string
	plan                                     ModelPlan
}

func NewModelService(cfg ModelServiceConfig) *ModelService {
	if cfg.ListTimeout <= 0 {
		cfg.ListTimeout = 90 * time.Second
	}
	if cfg.CallTimeout <= 0 {
		cfg.CallTimeout = 60 * time.Second
	}
	if cfg.StopTimeout <= 0 {
		cfg.StopTimeout = 45 * time.Second
	}
	if cfg.DrainTimeout <= 0 {
		cfg.DrainTimeout = 2 * time.Second
	}
	if cfg.Tick <= 0 {
		cfg.Tick = time.Second
	}
	if cfg.LocalAppData == "" {
		cfg.LocalAppData = os.Getenv("LOCALAPPDATA")
	}
	if cfg.UserProfile == "" {
		cfg.UserProfile = os.Getenv("USERPROFILE")
	}
	if cfg.DiskFree == nil {
		cfg.DiskFree = diskFreeWindows
	}
	if cfg.Inputs == nil {
		cfg.Inputs = func() ModelInputs { return ModelInputs{} }
	}
	if cfg.StallTimeout == nil {
		cfg.StallTimeout = func() time.Duration { return DefaultStallTimeout }
	}
	m := &ModelService{cfg: cfg}
	m.ctx, m.cancel = context.WithCancel(context.Background())
	m.state = ModelsState{Status: ModelsLoading, Reason: "The model list is being read.", Disk: ModelsDisk{}}
	return m
}

// Close ends the service. A running download is asked to cancel and then ended.
func (m *ModelService) Close() {
	m.mu.Lock()
	m.closed = true
	p := m.jobProc
	op := m.jobOp
	m.mu.Unlock()
	m.cancel()
	if p != nil {
		_ = p.Input(CancelLine(op))
		p.CloseInput()
		_ = p.Force()
	}
	m.wg.Wait()
}

func (m *ModelService) scrub(s string) string {
	if m.cfg.Keys == nil {
		return s
	}
	return m.cfg.Keys.ScrubString(s)
}
func (m *ModelService) scrubBytes(b []byte) []byte {
	if m.cfg.Keys == nil {
		return b
	}
	return m.cfg.Keys.Scrub(b)
}
func modelsNow() string { return time.Now().UTC().Format("2006-01-02T15:04:05Z") }

// ---------------------------------------------------------------------------
// State for the frame

// State returns the cached model state. It never blocks on a process or a disk.
func (m *ModelService) State() ModelsState {
	in := m.cfg.Inputs()
	m.mu.RLock()
	s := m.state
	diskFor := m.diskFor
	s.Refreshing = m.refreshing
	if m.job != nil {
		j := *m.job
		s.Download = &j
	}
	m.mu.RUnlock()
	s.Folders = append([]string{}, in.Folders...)
	// The space shown belongs to the folder new downloads go to. After a change
	// the old drive's number must not stay under the new folder.
	if !strings.EqualFold(filepath.Clean(diskFor), filepath.Clean(in.DataFolder)) {
		s.Disk = ModelsDisk{Path: in.DataFolder}
	}
	if s.Disk.Path == "" {
		// Until the CLI names its default folder, show the one the free-space
		// read starts from, so the Settings row always has a path.
		s.Disk.Path = m.modelsRoot()
	}
	suggested := make([]SuggestedFolder, 0, len(s.SuggestedFolders))
	for _, f := range s.SuggestedFolders {
		f.Added = false
		for _, a := range in.Folders {
			if SameFolder(a, f.Path) {
				f.Added = true
			}
		}
		suggested = append(suggested, f)
	}
	s.SuggestedFolders = suggested
	return copyModelsState(s)
}

// Refresh starts one background read of the model list. Calls while a read is
// running are joined into one more read. It returns at once.
func (m *ModelService) Refresh() {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	if m.refreshing {
		m.again = true
		m.mu.Unlock()
		return
	}
	m.refreshing = true
	m.wg.Add(1)
	m.mu.Unlock()
	RunProtected(func() {
		defer m.wg.Done()
		for {
			m.refreshOnce()
			m.mu.Lock()
			if !m.again || m.closed {
				m.refreshing = false
				m.mu.Unlock()
				return
			}
			m.again = false
			m.mu.Unlock()
		}
	})
}

// Started is true after the first Refresh call.
func (m *ModelService) Started() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.refreshing || m.state.UpdatedAt != ""
}

// modelsRoot is the CLI's default models folder, used for the free-space read
// until the first list names it. Which packages exist never comes from this
// path: they come from the CLI's list of the folders passed to scanFolders.
func (m *ModelService) modelsRoot() string {
	return filepath.Join(m.cfg.LocalAppData, "Bubo", "models")
}

// scanFolders are the folders the CLI is asked to search for model packages:
// the Import folders, the current download folder and every download folder
// used before. The GUI keeps no list of models; it passes the folders and
// shows what the CLI returns, so a new location needs no GUI work.
func (m *ModelService) scanFolders(in ModelInputs) []string {
	var out []string
	add := func(folder string, remembered bool) {
		if folder == "" {
			return
		}
		for _, have := range out {
			if SameFolder(have, folder) {
				return
			}
		}
		// A remembered folder the user deleted is not a problem to report. A
		// folder on a drive that is not there (unplugged) is: the CLI warns.
		if remembered {
			if _, err := os.Stat(folder); os.IsNotExist(err) {
				if _, rootErr := os.Stat(filepath.VolumeName(folder) + `\`); rootErr == nil {
					return
				}
			}
		}
		out = append(out, folder)
	}
	for _, f := range in.Folders {
		add(f, false)
	}
	add(in.DataFolder, true)
	for _, f := range in.DownloadFolders {
		add(f, true)
	}
	return out
}

func (m *ModelService) refreshOnce() {
	in := m.cfg.Inputs()
	root := in.DataFolder
	if root == "" {
		root = m.modelsRoot()
	}
	args := []string{"models", "list", "--json"}
	for _, f := range m.scanFolders(in) {
		args = append(args, "--scan-folder", f)
	}
	disk := make(chan ModelsDisk, 1)
	RunProtected(func() { disk <- m.readDisk(root) })
	events, err := m.capture(m.ctx, args, 1, m.cfg.ListTimeout)
	next := ModelsState{Status: ModelsOK, UpdatedAt: modelsNow()}
	if err == nil {
		models, ok := events[0].Value.(*Models)
		if !ok || events[0].Type != "models" {
			err = protocol("models list answered with another event")
		} else {
			next.Listing = &models.Listing
			next.Catalogue, next.PreparedModels = models.Catalogue, models.PreparedModels
			next.Scanned, next.ScanWarnings = models.Scanned, models.ScanWarnings
			if next.Catalogue == nil {
				next.Catalogue = []CatalogueEntry{}
			}
			if next.Scanned == nil {
				next.Scanned = []ScannedModel{}
			}
			if next.ScanWarnings == nil {
				next.ScanWarnings = []ScanWarning{}
			}
			if next.PreparedModels == nil {
				next.PreparedModels = []PreparedModel{}
			}
		}
	}
	if err != nil {
		next.Status = ModelsNotReported
		next.Reason, next.ReasonCode, next.ExitCode = m.reasonText(err)
	}
	select {
	case next.Disk = <-disk:
	case <-time.After(10 * time.Second):
		next.Disk = ModelsDisk{Path: root, Error: "The free space of the drive could not be read in time."}
	}
	if in.DataFolder == "" && next.Listing != nil && next.Listing.ModelsRoot != "" && !SameFolder(next.Listing.ModelsRoot, root) {
		// The CLI names the folder it really uses; read the space of that one.
		next.Disk = m.readDisk(next.Listing.ModelsRoot)
	}
	next.SuggestedFolders = m.suggestedFolders()
	m.mu.Lock()
	m.state = next
	m.diskFor = in.DataFolder
	m.mu.Unlock()
}

// reasonText turns a failed read into the plain reason of the frame.
func (m *ModelService) reasonText(err error) (reason, code string, exit int) {
	var coded interface{ ErrorCode() string }
	if errors.As(err, &coded) {
		return m.scrub(err.Error()), coded.ErrorCode(), 0
	}
	var ce *CommandError
	if errors.As(err, &ce) {
		c := ce.Event.Value.(*CLIError)
		code := c.Code
		if code == "unsupported" {
			code = "protocol_mismatch"
		}
		text := m.scrub(strings.TrimSpace(c.Remedy))
		switch code {
		case "runtime_missing":
			if text == "" {
				text = "The Bubo runtime is not installed. Prepare it first."
			}
		case "":
			code = "protocol_mismatch"
		}
		if text == "" {
			text = "The CLI could not read the model list (" + code + ")."
		}
		return text, code, int(ce.ExitCode)
	}
	var pe *ProtocolError
	if errors.As(err, &pe) {
		return "The CLI answer for the model list did not match the contract. " + m.scrub(pe.Detail), "protocol_mismatch", 0
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "Reading the model list took too long. Try again, or add fewer folders.", "timeout", 0
	}
	if errors.Is(err, context.Canceled) {
		return "Reading the model list was cancelled.", "cancelled", 0
	}
	return "The CLI could not be run: " + m.scrub(err.Error()), "cli_unavailable", 0
}

func (m *ModelService) readDisk(path string) ModelsDisk {
	d := ModelsDisk{Path: path}
	drive, free, err := m.cfg.DiskFree(path)
	d.Drive = drive
	if err != nil {
		d.Error = "The free space of the drive could not be read: " + err.Error()
		return d
	}
	d.FreeBytes = &free
	return d
}

// suggestedFolders lists folders of other programs that exist. It does plain
// existence checks on two fixed paths. It never reads inside them.
func (m *ModelService) suggestedFolders() []SuggestedFolder {
	out := []SuggestedFolder{}
	if m.cfg.UserProfile == "" {
		return out
	}
	candidates := []string{
		filepath.Join(m.cfg.UserProfile, ".lmstudio", "models"),
		filepath.Join(m.cfg.UserProfile, ".cache", "lm-studio", "models"),
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && fi.IsDir() {
			out = append(out, SuggestedFolder{Label: "LM Studio's model folder", Path: c, Exists: true})
			break
		}
	}
	return out
}

// diskFreeWindows reads the free space the user can use on the volume of the
// nearest existing parent of path. The folder may not exist yet.
func diskFreeWindows(path string) (string, uint64, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", 0, err
	}
	probe := abs
	for {
		if _, err := os.Stat(probe); err == nil {
			break
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return filepath.VolumeName(abs), 0, fmt.Errorf("no parent folder exists")
		}
		probe = parent
	}
	p, err := windows.UTF16PtrFromString(probe)
	if err != nil {
		return "", 0, err
	}
	var avail, total, totalFree uint64
	if err := windows.GetDiskFreeSpaceEx(p, &avail, &total, &totalFree); err != nil {
		return filepath.VolumeName(abs), 0, err
	}
	return filepath.VolumeName(abs), avail, nil
}

// ---------------------------------------------------------------------------
// CLI calls

type modelProc struct {
	p      *OwnedProcess
	events chan Event
	bad    chan error
	quit   chan struct{}
}

func (m *ModelService) exe() (string, error) {
	if m.cfg.Exe == nil {
		return "", errors.New("the CLI path is not set")
	}
	exe := m.cfg.Exe()
	if !filepath.IsAbs(exe) {
		return "", errors.New("the CLI path is not absolute")
	}
	return exe, nil
}

// spawn starts one owned CLI process. Events arrive in order on proc.events.
func (m *ModelService) spawn(args []string, runtimeDir string) (*modelProc, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, appError("closing", "Model work was requested while the GUI is closing.")
	}
	if m.updating {
		return nil, appError("update_installing", "Model work was requested during an update installation.")
	}
	exe, err := m.exe()
	if err != nil {
		return nil, err
	}
	mp := &modelProc{events: make(chan Event, 256), bad: make(chan error, 1), quit: make(chan struct{})}
	p, err := launchOwnedEnv(exe, args, childEnvironment(os.Environ(), "", runtimeDir), m.cfg.DrainTimeout, func(b []byte, complete bool) {
		e, err := ParseEvent(m.scrubBytes(b))
		if !complete {
			err = protocol("truncated stdout record")
		}
		if err != nil {
			select {
			case mp.bad <- err:
			default:
			}
			return
		}
		select {
		case mp.events <- e:
		case <-mp.quit:
		}
	}, func(b []byte, complete bool) {}, nil)
	if err != nil {
		return nil, err
	}
	mp.p = p
	if m.processes == nil {
		m.processes = make(map[*OwnedProcess]struct{})
	}
	m.processes[p] = struct{}{}
	RunProtected(func() {
		<-p.finished
		m.mu.Lock()
		delete(m.processes, p)
		m.mu.Unlock()
	})
	return mp, nil
}

// capture runs one CLI call and returns its records. At most max records are
// accepted. An error record becomes a CommandError that holds the exit code.
func (m *ModelService) capture(ctx context.Context, args []string, max int, limit time.Duration) ([]Event, error) {
	return m.captureRuntime(ctx, args, max, limit, m.cfg.Inputs().RuntimeDir, false)
}

func (m *ModelService) captureRuntime(ctx context.Context, args []string, max int, limit time.Duration, runtimeDir string, discardProgress bool) ([]Event, error) {
	mp, err := m.spawn(args, runtimeDir)
	if err != nil {
		return nil, err
	}
	defer close(mp.quit)
	ctx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	var events []Event
	appendEvent := func(e Event) {
		// A plan verifies existing files before returning its bounded plan parts.
		// Verification progress can be numerous and has no plan data to retain.
		if !discardProgress || e.Type != "progress" {
			events = append(events, e)
		}
	}
	collect := func() {
		for {
			select {
			case e := <-mp.events:
				appendEvent(e)
			default:
				return
			}
		}
	}
	for {
		select {
		case e := <-mp.events:
			appendEvent(e)
			if len(events) > max {
				mp.p.Force()
				<-mp.p.Done
				return nil, protocol("the CLI sent more records than the contract allows")
			}
		case result := <-mp.p.Done:
			collect()
			select {
			case err := <-mp.bad:
				return nil, err
			default:
			}
			if len(events) > 0 && events[len(events)-1].Type == "error" {
				return events, &CommandError{events[len(events)-1], result.ExitCode}
			}
			if result.Err != nil {
				return nil, result.Err
			}
			if result.DrainTimedOut {
				return nil, protocol("helper pipes did not reach EOF")
			}
			if len(events) == 0 {
				if result.ExitCode != 0 {
					return nil, cliExitError(result.ExitCode)
				}
				return nil, protocol("missing one-shot record")
			}
			if result.ExitCode != 0 {
				return nil, cliExitError(result.ExitCode)
			}
			if len(events) > max {
				return nil, protocol("the CLI sent more records than the contract allows")
			}
			return events, nil
		case err := <-mp.bad:
			mp.p.Force()
			<-mp.p.Done
			return nil, err
		case <-ctx.Done():
			mp.p.Force()
			<-mp.p.Done
			return nil, ctx.Err()
		}
	}
}

func checkProduct(e Event) error {
	if e.Product.Name != "bubo" {
		return protocol("the answer does not come from bubo")
	}
	return nil
}

// Estimate runs `models estimate` for a catalogue id or an absolute path.
// context 0 and sessions 0 leave the CLI defaults.
func (m *ModelService) Estimate(subject string, contextTokens uint64, sessions uint32) (ModelEstimate, error) {
	return m.estimateRuntime(subject, contextTokens, sessions, m.cfg.Inputs().RuntimeDir, "")
}

// A saved setup estimate must retain the runtime captured with that request.
func (m *ModelService) estimateRuntime(subject string, contextTokens uint64, sessions uint32, runtimeDir, profile string) (ModelEstimate, error) {
	if !ValidCatalogueID(subject) {
		// A full path: a model file or a package folder. Pass it clean, without
		// a trailing separator.
		clean, err := NormalizeModelFolderPath(subject)
		if err != nil {
			return ModelEstimate{}, appError("invalid_arguments", "Choose a catalogue id or a full path.")
		}
		subject = clean
	}
	args := []string{"models", "estimate", "--model", subject, "--json"}
	if contextTokens > 0 {
		args = append(args, "--context", strconv.FormatUint(contextTokens, 10))
	}
	if sessions > 0 {
		args = append(args, "--sessions", strconv.FormatUint(uint64(sessions), 10))
	}
	if profile != "" {
		args = append(args, "--profile", profile)
	}
	events, err := m.captureRuntime(m.ctx, args, 1, m.cfg.CallTimeout, runtimeDir, false)
	if err != nil {
		return ModelEstimate{}, m.plainError(err)
	}
	est, ok := events[0].Value.(*ModelEstimate)
	if !ok || checkProduct(events[0]) != nil {
		return ModelEstimate{}, protocol("estimate answered with another event")
	}
	return *est, nil
}

// ModelError is a CLI failure in the words the CLI gave.
type ModelError struct {
	Code     string
	Remedy   string
	ExitCode int
	Detail   string
}

func (e *ModelError) Error() string { return e.Remedy }

func (m *ModelService) plainError(err error) error {
	var ce *CommandError
	if errors.As(err, &ce) {
		c := ce.Event.Value.(*CLIError)
		code := c.Code
		if code == "unsupported" {
			code = "protocol_mismatch"
		}
		r := m.scrub(strings.TrimSpace(c.Remedy))
		if r == "" {
			r = "The CLI stopped with " + code + "."
		}
		return &ModelError{Code: code, Remedy: r, ExitCode: int(ce.ExitCode), Detail: m.scrub(strings.TrimSpace(c.Detail))}
	}
	reason, code, _ := m.reasonText(err)
	return &ModelError{Code: code, Remedy: reason}
}

// ---------------------------------------------------------------------------
// Plan, download, repair, check

func (m *ModelService) jobArgs0(id, kind string, dataFolder string, planOnly bool) []string {
	args := []string{"prepare", "model", "--catalogue-id", id, "--jsonl"}
	if planOnly {
		args = append(args, "--plan")
	} else {
		switch kind {
		case "repair":
			args = append(args, "--verify")
		case "check":
			// Offline: the CLI cannot start a transfer. It verifies the files.
			args = append(args, "--offline", "--verify")
		}
	}
	// The caller decides the folder: new downloads use the download folder; a
	// check or repair uses the --output of the row it was started from.
	if dataFolder != "" {
		args = append(args, "--output", dataFolder)
	}
	return args
}

// repairOutputOf is the --output a retry of a repair or check must reuse.
func repairOutputOf(kind, output string) string {
	if kind == "download" {
		return ""
	}
	return output
}

func (m *ModelService) busy() bool {
	if m.job == nil {
		return false
	}
	switch m.job.State {
	case "planning", "running", "cancelling":
		return true
	}
	return false
}

func (m *ModelService) setJob(j *ModelJob) {
	j.UpdatedAt = modelsNow()
	// The job's own plan is dropped from the frame once the user agreed, so
	// the whole-job numbers are added here, for the frame and the event alike.
	j.Overall = nil
	if j.Plan != nil {
		j.Overall = jobProgress(planJobFiles(*j.Plan), j.Plan.Kind, j.FileID, j.Phase, j.BytesDone)
	}
	m.mu.Lock()
	cp := *j
	m.job = &cp
	m.mu.Unlock()
	if m.cfg.Events != nil {
		job := copyModelJob(cp)
		m.cfg.Events(StreamEvent{Kind: DownloadProgress, Job: &job})
	}
}

// StartDownload runs `prepare model --catalogue-id ID --plan` and returns the
// plan: files, sizes, free space and the licence text. Nothing is written and
// nothing is downloaded. The window shows the plan and calls ConfirmDownload
// when the user agrees. kind is "download" or "repair".
func (m *ModelService) StartDownload(id, kind string) (ModelPlan, error) {
	return m.startAt(id, kind, "", false)
}

// StartRepair plans the repair of a package whose row carries the --output the
// CLI named for it (the row's repair_output). The GUI does no folder arithmetic.
func (m *ModelService) StartRepair(id, output string) (ModelPlan, error) {
	return m.startAt(id, "repair", output, true)
}

// startAt: a download goes to the download folder. A repair or check acts where
// the CLI reads the row it was started from: the folder the row names, else the
// CLI default (explicit false).
func (m *ModelService) startAt(id, kind, output string, explicit bool) (ModelPlan, error) {
	if kind != "download" && kind != "repair" {
		return ModelPlan{}, appError("internal_error", "Unknown model action.")
	}
	if !ValidCatalogueID(id) {
		return ModelPlan{}, appError("invalid_arguments", "The model id is not valid.")
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ModelPlan{}, appError("closing", "The GUI is exiting.")
	}
	if m.busy() {
		m.mu.Unlock()
		return ModelPlan{}, appError("busy", "A model action is already running. Wait for it or cancel it.")
	}
	m.mu.Unlock()
	in := m.cfg.Inputs()
	if kind == "download" {
		output = in.DataFolder
	} else if !explicit {
		output = ""
	}
	j := &ModelJob{ID: id, Kind: kind, State: "planning", Message: "Reading the plan.", RepairOutput: repairOutputOf(kind, output)}
	m.setJob(j)
	events, err := m.captureRuntime(m.ctx, m.jobArgs0(id, kind, output, true), 400, m.cfg.CallTimeout, in.RuntimeDir, true)
	fail := func(err error) (ModelPlan, error) {
		me := m.plainError(err)
		var e *ModelError
		errors.As(me, &e)
		j.State, j.Message, j.Error = "failed", e.Remedy, &ModelJobError{Code: e.Code, Remedy: e.Remedy, ExitCode: e.ExitCode, Detail: e.Detail}
		m.setJob(j)
		return ModelPlan{}, me
	}
	if err != nil {
		return fail(err)
	}
	var parts []Plan
	var result *Result
	for _, e := range events {
		if err := checkProduct(e); err != nil {
			return fail(err)
		}
		switch v := e.Value.(type) {
		case *Plan:
			parts = append(parts, *v)
		case *Result:
			result = v
		}
	}
	if result == nil || !result.PlanOnly || result.ExitCode != 0 {
		return fail(protocol("the plan call did not end with a plan_only result"))
	}
	plan, err := assemblePlan(parts, "model")
	if err != nil {
		return fail(err)
	}
	plan.ID, plan.Kind = id, kind
	plan.Licence = m.catalogueLicence(id)
	j.State, j.Message, j.Plan = "awaiting_confirm", "The plan is ready. Nothing is downloaded until you confirm.", &plan
	m.mu.Lock()
	m.pendingRun = &pendingRun{id: id, kind: kind, digest: plan.Digest, dataFolder: output, runtimeDir: in.RuntimeDir, plan: plan}
	m.mu.Unlock()
	m.setJob(j)
	return plan, nil
}

func (m *ModelService) catalogueLicence(id string) *ModelLicence {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, c := range m.state.Catalogue {
		if c.ID == id {
			l := c.Licence
			return &l
		}
	}
	return nil
}

// ConfirmDownload starts the download of the plan the user saw. It refuses
// when no plan is waiting for this id, when a volume is too small, or when the
// CLI later asks to approve a plan with another digest.
func (m *ModelService) ConfirmDownload(id string) error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return appError("closing", "The GUI is exiting.")
	}
	pr := m.pendingRun
	if pr == nil || pr.id != id || m.job == nil || m.job.State != "awaiting_confirm" || m.job.ID != id {
		m.mu.Unlock()
		return appError("plan_changed", "There is no plan for this model. Start the download again to see the plan.")
	}
	if !pr.plan.EnoughSpace {
		m.mu.Unlock()
		return appError("insufficient_space", "There is not enough free space on the drive. Free space, then start again.")
	}
	m.pendingRun = nil
	m.mu.Unlock()
	return m.run(pr.id, pr.kind, pr.digest, pr.dataFolder, pr.runtimeDir, &pr.plan)
}

// Check verifies the files of a prepared model without a network. The CLI
// runs with --offline --verify.
func (m *ModelService) Check(id string) error { return m.CheckAt(id, "") }

// CheckAt is Check for a package whose row names its own --output.
func (m *ModelService) CheckAt(id, output string) error {
	if !ValidCatalogueID(id) {
		return appError("invalid_arguments", "The model id is not valid.")
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return appError("closing", "The GUI is exiting.")
	}
	if m.busy() {
		m.mu.Unlock()
		return appError("busy", "A model action is already running. Wait for it or cancel it.")
	}
	m.mu.Unlock()
	in := m.cfg.Inputs()
	return m.run(id, "check", "", output, in.RuntimeDir, nil)
}

// CancelJob discards a waiting plan or asks a running job to cancel. After
// StopTimeout the owned process is ended.
func (m *ModelService) CancelJob() error {
	m.mu.Lock()
	if m.job == nil {
		m.mu.Unlock()
		return nil
	}
	switch m.job.State {
	case "awaiting_confirm":
		m.pendingRun = nil
		j := *m.job
		m.mu.Unlock()
		j.State, j.Message, j.Plan = "cancelled", "Cancelled. Nothing was downloaded.", nil
		m.setJob(&j)
		return nil
	case "running":
		j := *m.job
		m.mu.Unlock()
		// The run loop sends the cancel line as soon as it knows the operation id.
		j.State, j.Message = "cancelling", "Cancelling. The CLI is asked to stop."
		m.setJob(&j)
		return nil
	case "cancelling", "planning":
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()
	return nil
}

// DismissJob clears a finished job from the frame.
func (m *ModelService) DismissJob() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.job != nil && !m.busy() && m.job.State != "awaiting_confirm" {
		m.job = nil
	}
}

func (m *ModelService) run(id, kind, digest, dataFolder, runtimeDir string, plan *ModelPlan) error {
	args := m.jobArgs0(id, kind, dataFolder, false)
	j := &ModelJob{ID: id, Kind: kind, State: "running", Plan: plan, Message: "Starting.", BytesTotal: nil, RepairOutput: repairOutputOf(kind, dataFolder)}
	mp, err := m.spawn(args, runtimeDir)
	if err != nil {
		me := m.plainError(err)
		var e *ModelError
		errors.As(me, &e)
		j.State, j.Message, j.Error = "failed", e.Remedy, &ModelJobError{Code: e.Code, Remedy: e.Remedy, ExitCode: e.ExitCode, Detail: e.Detail}
		m.setJob(j)
		return me
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		mp.p.Force()
		close(mp.quit)
		return appError("closing", "The GUI is exiting.")
	}
	m.jobProc = mp.p
	m.wg.Add(1)
	m.mu.Unlock()
	m.setJob(j)
	RunProtected(func() { m.follow(mp, j, digest, kind == "check") })
	return nil
}

// follow reads the events of one prepare run until the process ends.
func (m *ModelService) follow(mp *modelProc, j *ModelJob, digest string, offline bool) {
	defer m.wg.Done()
	defer close(mp.quit)
	var parts []Plan
	var wireOp string
	var seq uint64
	var cancelling, cancelConfirmed, cancelSent bool
	var terminal *ModelJobError
	var result *Result
	lastProgress := time.Now()
	var lastDone uint64
	var lastPhase, lastFile string
	var forceTimer <-chan time.Time
	tick := time.NewTicker(m.cfg.Tick)
	defer tick.Stop()
	set := func() {
		// Cancel is set from another goroutine; a progress update must not undo it.
		m.mu.RLock()
		if m.job != nil && m.job.State == "cancelling" && j.State == "running" {
			j.State, j.Message = "cancelling", m.job.Message
		}
		m.mu.RUnlock()
		m.setJob(j)
	}
	fail := func(code, remedy string) {
		if terminal == nil {
			terminal = &ModelJobError{Code: code, Remedy: remedy}
		}
	}
	handle := func(e Event) {
		if wireOp == "" {
			wireOp = e.OperationID
			m.mu.Lock()
			m.jobOp = wireOp
			m.mu.Unlock()
		}
		if e.OperationID != wireOp || e.Seq <= seq || e.Product.Name != "bubo" {
			fail("protocol_mismatch", "The CLI answer broke the contract (operation or sequence). The download was stopped.")
			RunProtected(func() { _ = mp.p.Input(CancelLine(wireOp)); mp.p.CloseInput() })
			return
		}
		seq = e.Seq
		switch v := e.Value.(type) {
		case *Plan:
			parts = append(parts, *v)
		case *Approval:
			p, err := assemblePlan(parts, "model")
			ok := err == nil && len(v.Components) == 1 && v.Components[0] == "model" &&
				v.Digest != "" && (digest == "" || p.Digest == digest)
			// A check runs with --offline, so an approval cannot start a transfer.
			if !ok && !(offline && err == nil && len(v.Components) == 1 && v.Components[0] == "model") {
				fail("plan_changed", "The plan changed after you saw it. Start again to see the new plan.")
				_ = mp.p.Input(ApprovalLine("decline", wireOp, v.Digest))
				mp.p.CloseInput()
				return
			}
			if err := mp.p.Input(ApprovalLine("approve", wireOp, v.Digest)); err != nil {
				fail("cli_unavailable", "The approval could not be sent to the CLI.")
			}
			lastProgress = time.Now()
			j.Message = "Approved. Waiting for the first file."
			set()
		case *Progress:
			if v.Done != lastDone || v.Phase != lastPhase || v.FileID != lastFile {
				lastProgress, lastDone, lastPhase, lastFile = time.Now(), v.Done, v.Phase, v.FileID
				j.Stalled, j.StallSeconds = false, 0
			}
			j.Phase, j.FileID, j.BytesDone, j.BytesTotal = v.Phase, v.FileID, v.Done, v.Total
			j.Message = ""
			set()
		case *Result:
			r := *v
			result = &r
		case *CLIError:
			if v.Code == "cancelled" && cancelling {
				cancelConfirmed = true
			} else {
				code := v.Code
				if code == "unsupported" || code == "" {
					code = "protocol_mismatch"
				}
				fail(code, m.scrub(strings.TrimSpace(v.Remedy)))
				if terminal != nil && terminal.Detail == "" {
					terminal.Detail = m.scrub(strings.TrimSpace(v.Detail))
				}
			}
		}
	}
	finish := func(res ProcessResult) {
		for len(mp.events) > 0 {
			handle(<-mp.events)
		}
		select {
		case err := <-mp.bad:
			fail("protocol_mismatch", m.scrub(err.Error()))
		default:
		}
		m.mu.Lock()
		m.jobProc, m.jobOp = nil, ""
		m.mu.Unlock()
		j.Stalled, j.StallSeconds = false, 0
		switch {
		case cancelConfirmed && terminal == nil:
			j.State, j.Message = "cancelled", "Cancelled. Files that were complete stay; partial files can be resumed."
		case terminal != nil:
			terminal.ExitCode = int(res.ExitCode)
			if terminal.Remedy == "" {
				terminal.Remedy = "The model action stopped with " + terminal.Code + "."
			}
			j.State, j.Error, j.Message = "failed", terminal, terminal.Remedy
		case cancelling && res.ExitCode != 0:
			j.State, j.Message = "cancelled", "Cancelled. Files that were complete stay; partial files can be resumed."
		case res.Err != nil || res.DrainTimedOut || res.ExitCode != 0 || result == nil:
			code, why := "protocol_mismatch", fmt.Sprintf("The CLI ended with exit %d and no result.", res.ExitCode)
			if result != nil && res.ExitCode == 0 {
				why = "The CLI process did not end cleanly."
			}
			j.State, j.Error, j.Message = "failed", &ModelJobError{Code: code, Remedy: why, ExitCode: int(res.ExitCode)}, why
		case result.ExitCode != 0 || !result.Prepared:
			why := fmt.Sprintf("The CLI did not report the model as prepared (exit %d).", result.ExitCode)
			j.State, j.Error, j.Message = "failed", &ModelJobError{Code: "model_unusable", Remedy: why, ExitCode: int(result.ExitCode)}, why
		default:
			j.State, j.Prepared, j.ManifestPath = "done", true, result.ManifestPath
			if j.Kind == "check" {
				j.Message = "The files are complete and the checks passed."
			} else {
				j.Message = "The model is ready."
			}
		}
		set()
		if m.cfg.OnJobEnd != nil {
			m.cfg.OnJobEnd(*j)
		}
		m.Refresh()
	}
	for {
		select {
		case e := <-mp.events:
			handle(e)
		case err := <-mp.bad:
			fail("protocol_mismatch", "The CLI answer broke the contract: "+m.scrub(err.Error()))
			RunProtected(func() { _ = mp.p.Input(CancelLine(wireOp)); mp.p.CloseInput() })
		case res := <-mp.p.Done:
			finish(res)
			return
		case <-tick.C:
			m.mu.RLock()
			state := ""
			if m.job != nil {
				state = m.job.State
			}
			m.mu.RUnlock()
			if state == "cancelling" && !cancelling {
				cancelling = true
				forceTimer = time.After(m.cfg.StopTimeout)
			}
			if cancelling && !cancelSent && wireOp != "" {
				cancelSent = true
				op := wireOp
				RunProtected(func() {
					_ = mp.p.Input(CancelLine(op))
					mp.p.CloseInput()
				})
			}
			if state == "running" {
				limit := stallLimit(m.cfg.StallTimeout())
				if limit > 0 && time.Since(lastProgress) >= limit && !j.Stalled {
					j.Stalled = true
					j.StallSeconds = int(time.Since(lastProgress).Seconds())
					fail("download_stalled", fmt.Sprintf("No download progress for %d s.", j.StallSeconds))
					// Ask the CLI to cancel; only StopTimeout forces it.
					if !cancelSent && wireOp != "" {
						cancelSent = true
						op := wireOp
						RunProtected(func() {
							_ = mp.p.Input(CancelLine(op))
							mp.p.CloseInput()
						})
					}
					if forceTimer == nil {
						forceTimer = time.After(m.cfg.StopTimeout)
					}
					set()
				}
			}
		case <-forceTimer:
			forceTimer = nil
			fail("stop_timeout", "The CLI did not stop in time and was ended.")
			_ = mp.p.Force()
		}
	}
}
