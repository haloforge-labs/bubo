//go:build windows

package supervisor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"bubo/gui/updater"
)

type updateClient interface {
	Check(context.Context) updater.CheckResult
	Download(context.Context, func(updater.Progress)) (*updater.Staged, *updater.Failure)
	Install(*updater.Staged) (*updater.Handoff, *updater.Failure)
	Recover() (*updater.Handoff, *updater.Failure)
	LastResult() (*updater.InstallResult, *updater.Failure)
	Close()
}

type UpdateServiceConfig struct {
	Updater  updater.Config
	Snapshot func() Snapshot
	Enabled  func() bool
	// Quiesce blocks new CLI work, stops normally, and returns retained owned
	// PIDs plus a release function. Release runs on a failed handoff as well.
	Quiesce        func(context.Context) ([]uint32, func(), error)
	Events         func(StreamEvent)
	Exit           func()
	Health         *UpdateHealth
	LogDir         string
	PrepareRuntime func() ActionResult
}

// UpdateService owns app update state. Startup runs once on a worker; only
// explicit actions reach Download, Install or Recover. No window code owns it.
type UpdateService struct {
	mu          sync.Mutex
	cfg         UpdateServiceConfig
	view        Update
	client      updateClient
	staged      *updater.Staged
	busy        bool
	cancel      context.CancelFunc
	ctx         context.Context
	close       context.CancelFunc
	startOnce   sync.Once
	wg          sync.WaitGroup
	closed      bool
	initialized bool
	healthy     bool
	recovery    bool
	via         string
	pids        []uint32
	factory     func(updater.Config) (updateClient, *updater.Failure)
	logMu       sync.Mutex
	logPath     string
	logError    error
}

func NewUpdateService(cfg UpdateServiceConfig) *UpdateService {
	ctx, cancel := context.WithCancel(context.Background())
	u := &UpdateService{cfg: cfg, ctx: ctx, close: cancel, view: Update{State: UpdateNone}, factory: func(c updater.Config) (updateClient, *updater.Failure) { return updater.New(c) }}
	if cfg.LogDir != "" {
		u.logPath = filepath.Join(cfg.LogDir, "updater.log")
		u.view.LogPath = u.logPath
		if err := os.MkdirAll(cfg.LogDir, 0700); err != nil {
			u.logError = err
		} else if f, err := os.OpenFile(u.logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600); err != nil {
			u.logError = err
		} else {
			_ = f.Close()
		}
	}
	return u
}

func (u *UpdateService) State() Update {
	u.mu.Lock()
	view := copyUpdate(u.view)
	u.mu.Unlock()
	if view.LogPath != "" && u.logError == nil {
		if data, err := os.ReadFile(view.LogPath); err == nil {
			lines := strings.Split(strings.TrimSpace(string(data)), "\n")
			if len(lines) > 200 {
				lines = lines[len(lines)-200:]
			}
			view.LogLines = lines
		}
	}
	return view
}

func (u *UpdateService) logEvent(event string) {
	if u.logPath == "" || u.logError != nil {
		return
	}
	u.logMu.Lock()
	defer u.logMu.Unlock()
	f, err := os.OpenFile(u.logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(f, "%s %s\n", time.Now().UTC().Format(time.RFC3339), strings.Join(strings.Fields(event), " "))
	_ = f.Close()
}

// Optional server autostart must not load an engine in a candidate that has
// yet to acknowledge the helper's window/supervisor health handshake.
func (u *UpdateService) StartupReady() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.cfg.Health == nil || u.healthy
}

// Start is called on window startup, with Wails' DOM-ready notification. The
// health deadline includes the wait for both window and CLI initialization.
func (u *UpdateService) Start(windowReady <-chan struct{}) {
	u.startOnce.Do(func() {
		u.mu.Lock()
		defer u.mu.Unlock()
		if u.closed {
			return
		}
		u.wg.Add(1)
		RunProtected(func() { defer u.wg.Done(); u.start(windowReady) })
	})
}

func (u *UpdateService) start(windowReady <-chan struct{}) {
	if u.logError != nil {
		u.failCheck(&updater.Failure{Code: updater.CodeIO, Message: "cannot open updater log: " + u.logError.Error()})
		return
	}
	u.logEvent("Updater starting")
	ctx := u.ctx
	if u.cfg.Health != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, u.cfg.Health.Deadline)
		defer cancel()
	} else {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, updater.DefaultHealthTimeout)
		defer cancel()
	}
	select {
	case <-windowReady:
	case <-ctx.Done():
		u.startFailure(ctx.Err())
		return
	}
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	var snap Snapshot
	for {
		snap = u.cfg.Snapshot()
		if snap.Initialized {
			break
		}
		select {
		case <-ctx.Done():
			u.startFailure(ctx.Err())
			return
		case <-tick.C:
		}
	}
	u.mu.Lock()
	version := u.cfg.Updater.InstalledVersion
	installDir := u.cfg.Updater.InstallDir
	u.mu.Unlock()
	if version == "" && installDir != "" {
		var f *updater.Failure
		version, f = updater.InstalledVersion(installDir)
		if f != nil {
			u.initializationFailure(f)
			return
		}
	} else if version == "" && snap.Status != nil {
		// Headless service fixtures have no installed package tree.
		version = snap.Status.Versions.Version
	}
	u.mu.Lock()
	u.cfg.Updater.InstalledVersion = version
	u.view.CurrentVersion = version
	if snap.Status != nil {
		if path, ok := switchScalar(snap.Status.Paths["runtime"]); ok {
			u.cfg.Updater.OldRuntimePath = path
		}
	}
	u.cfg.Updater.ProcessIDs = func() ([]uint32, error) {
		u.mu.Lock()
		defer u.mu.Unlock()
		return append([]uint32{uint32(os.Getpid())}, u.pids...), nil
	}
	client, f := u.factory(u.cfg.Updater)
	if f == nil {
		u.client = client
	}
	u.mu.Unlock()
	if f != nil {
		u.initializationFailure(f)
		return
	}
	// Package metadata and updater construction must succeed before the helper
	// commits the candidate. Engine and model readiness are independent.
	if u.cfg.Health != nil {
		if err := u.cfg.Health.Acknowledge(snap); err != nil {
			u.startFailure(err)
			return
		}
	}
	u.mu.Lock()
	u.healthy = true
	u.mu.Unlock()
	following := false
	if result, f := client.LastResult(); f != nil {
		u.failCheck(f)
		return
	} else if result != nil {
		u.lastResult(result)
	} else if pending, ok := client.(interface{ PendingResult() bool }); ok && pending.PendingResult() {
		following = true
	}
	if u.State().State == UpdateRollbackPending {
		u.mu.Lock()
		u.initialized = true
		u.view.Ready = true
		u.mu.Unlock()
		return
	}
	u.mu.Lock()
	u.initialized = true
	u.view.Ready = true
	u.mu.Unlock()
	u.logEvent("Updater ready with installed version " + version)
	if !following {
		u.startupCheck()
		return
	}
	// The client runs one operation at a time, so the startup check waits until
	// the helper's result is read; beside it, the check would answer busy.
	u.wg.Add(1)
	RunProtected(func() {
		defer u.wg.Done()
		u.followResult(client)
		u.startupCheck()
	})
}

// startupCheck is the one automatic check after start, unless Settings turned
// automatic checks off or an interrupted update waits for recovery.
func (u *UpdateService) startupCheck() {
	if u.ctx.Err() != nil || u.State().State == UpdateRollbackPending {
		return
	}
	if u.cfg.Enabled == nil || u.cfg.Enabled() {
		_ = u.CheckUpdate()
	}
}

func (u *UpdateService) followResult(client updateClient) {
	ctx, cancel := context.WithTimeout(u.ctx, 2*updater.DefaultHealthTimeout)
	defer cancel()
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	for {
		result, f := client.LastResult()
		if f == nil && result != nil {
			u.lastResult(result)
			return
		}
		if f != nil && f.Code != updater.CodeBusy {
			u.fail(f)
			return
		}
		select {
		case <-ctx.Done():
			u.logEvent("Update helper result did not arrive before deadline")
			return
		case <-tick.C:
		}
	}
}

// previousVersion reads the kept .previous package. It is empty when the
// folder is absent (a rollback that already restored it) or unreadable.
func previousVersion(installDir string) string {
	if installDir == "" {
		return ""
	}
	version, f := updater.InstalledVersion(installDir + ".previous")
	if f != nil {
		return ""
	}
	return version
}

func (u *UpdateService) startFailure(err error) {
	if u.ctx.Err() != nil {
		return
	}
	u.failCheck(&updater.Failure{Code: updater.CodeHealth, Message: err.Error(), Retry: "Restart the GUI and try again."})
	if u.cfg.Health != nil && u.cfg.Exit != nil {
		u.cfg.Exit()
	}
}

func (u *UpdateService) initializationFailure(f *updater.Failure) {
	u.failCheck(f)
	if u.cfg.Health != nil && u.cfg.Exit != nil {
		u.cfg.Exit()
	}
}

func updateError(f *updater.Failure) *CLIError {
	if f == nil {
		return nil
	}
	parts := []string{f.Message}
	if f.Via != "" {
		parts = append(parts, "Check again to check intermediate version "+f.Via+" first.")
	}
	if f.URL != "" {
		parts = append(parts, "URL: "+f.URL)
	}
	if f.Path != "" {
		parts = append(parts, "Path: "+f.Path)
	}
	if f.ExpectedBytes != 0 || f.ReceivedBytes != 0 {
		parts = append(parts, fmt.Sprintf("Bytes: expected %d, received %d.", f.ExpectedBytes, f.ReceivedBytes))
	}
	if f.ExpectedHash != "" || f.ReceivedHash != "" {
		parts = append(parts, "SHA-256: expected "+f.ExpectedHash+", received "+f.ReceivedHash+".")
	}
	if f.Partial != "" {
		parts = append(parts, "Partial download: "+f.Partial+".")
	}
	if f.NotesURL != "" {
		parts = append(parts, "Release page: "+f.NotesURL)
	}
	if f.Retry != "" && f.Retry != "retry the same action" {
		parts = append(parts, f.Retry)
	}
	code := string(f.Code)
	if f.Code == updater.CodeUnsupported {
		code = "internal_error"
		if f.Message == "manifest schema is newer than this updater" {
			code = "updater_too_old"
		}
	}
	return &CLIError{Code: code, Component: "update", Remedy: strings.Join(parts, " "), Retryable: f.Code != updater.CodeUnsupported && f.Code != updater.CodeDowngrade}
}

// fail records the failure of an update the user started (download or install).
func (u *UpdateService) fail(f *updater.Failure) ActionResult { return u.failAs(f, false) }

// failCheck records a failure of the updater's start or of a check. The user
// started no update, so the view marks it and the header shows no pill.
func (u *UpdateService) failCheck(f *updater.Failure) ActionResult { return u.failAs(f, true) }

func (u *UpdateService) failAs(f *updater.Failure, check bool) ActionResult {
	u.logEvent("Update failed: " + string(f.Code) + ": " + f.Message)
	u.mu.Lock()
	rollback := f.Code == updater.CodeRollbackPending || u.recovery
	installDir := u.cfg.Updater.InstallDir
	u.mu.Unlock()
	previous := ""
	if rollback {
		previous = previousVersion(installDir)
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	u.view.State, u.view.Error = UpdateFailed, updateError(f)
	u.view.CheckFailed = check
	if rollback {
		u.view.State = UpdateRollbackPending
		u.view.CheckFailed = false
		u.view.PreviousVersion = previous
		u.recovery = true
	}
	u.view.Progress = nil
	if f.Code == updater.CodeIntermediateRequired && f.Via != "" {
		u.via = f.URL
	}
	err := copyCLIError(*u.view.Error)
	return ActionResult{Error: &err}
}

func (u *UpdateService) begin() (updateClient, context.Context, ActionResult) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.closed || u.client == nil || !u.initialized {
		if !u.view.Ready && u.view.Error != nil {
			err := copyCLIError(*u.view.Error)
			return nil, nil, ActionResult{Error: &err}
		}
		return nil, nil, codedAction("updater_starting", "Updater is not initialized.")
	}
	if u.busy {
		return nil, nil, ActionResult{Error: updateError(&updater.Failure{Code: updater.CodeBusy, Message: "An update action is active."})}
	}
	u.busy = true
	ctx, cancel := context.WithCancel(u.ctx)
	u.cancel = cancel
	u.wg.Add(1)
	return u.client, ctx, ActionResult{}
}

func (u *UpdateService) end() {
	u.mu.Lock()
	u.cancel()
	u.cancel, u.busy = nil, false
	u.mu.Unlock()
	u.wg.Done()
}

// retryStart builds the updater client again after a failed start. Only a
// client that could not be built is retried: the version and folders were
// read before that step. A candidate's health launch never retries.
func (u *UpdateService) retryStart() {
	u.mu.Lock()
	need := !u.closed && u.client == nil && !u.initialized && u.view.CheckFailed && u.cfg.Health == nil && u.cfg.Updater.InstalledVersion != ""
	cfg := u.cfg.Updater
	u.mu.Unlock()
	if !need {
		return
	}
	client, f := u.factory(cfg)
	if f != nil {
		u.failCheck(f)
		return
	}
	u.mu.Lock()
	u.client, u.initialized, u.healthy = client, true, true
	u.view.Ready, u.view.State, u.view.Error, u.view.CheckFailed = true, UpdateNone, nil, false
	u.mu.Unlock()
	u.logEvent("Updater ready after a retry")
}

func (u *UpdateService) CheckUpdate() ActionResult {
	u.retryStart()
	client, ctx, r := u.begin()
	if r.Error != nil {
		return r
	}
	defer u.end()
	u.mu.Lock()
	if u.view.State == UpdateRollbackPending {
		u.mu.Unlock()
		return codedAction("rollback_pending", "Recover the pending update before checking again.")
	}
	via := u.via
	u.via, u.staged = "", nil
	if via != "" {
		cfg := u.cfg.Updater
		cfg.ManifestURL = via
		var f *updater.Failure
		client, f = u.factory(cfg)
		if f != nil {
			u.mu.Unlock()
			return u.failCheck(f)
		}
		u.client.Close()
		u.client = client
	}
	u.mu.Unlock()
	result := client.Check(ctx)
	if result.Failure != nil {
		return u.failCheck(result.Failure)
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	notice, current, runtimeRequired := u.view.LastResultNotice, u.view.CurrentVersion, u.view.RuntimeFollowupRequired
	u.view = Update{State: UpdateCurrent, Ready: true, CurrentVersion: current, LastResultNotice: notice, LogPath: u.logPath, RuntimeFollowupRequired: runtimeRequired}
	if release := result.Available; release != nil {
		size := uint64(release.Size)
		u.view.State, u.view.AvailableVersion, u.view.Notes, u.view.SizeBytes = UpdateAvailable, release.Version, release.Notes.Summary, &size
		u.view.RuntimeDownloadBytes, u.view.ModelsDownloadBytes = ReleaseReuseFacts(release.Manifest)
		for _, a := range release.Advisories {
			u.view.Notes += "\n" + a.Text
		}
	}
	u.logEvent("Check completed: " + string(result.Status))
	return ActionResult{}
}

// ReleaseReuseFacts projects a validated release's explicit payload decisions.
// A changed component never acquires an inferred zero-byte reuse promise.
func ReleaseReuseFacts(manifest updater.Manifest) (runtime, models *uint64) {
	if !manifest.Runtime.Changed {
		value := uint64(manifest.Runtime.DownloadBytes)
		runtime = &value
	}
	if !manifest.Models.Changed && len(manifest.Models.NeedsPrepare) == 0 {
		value := uint64(0)
		models = &value
	}
	return
}

func (u *UpdateService) DownloadUpdate() ActionResult {
	client, ctx, r := u.begin()
	if r.Error != nil {
		return r
	}
	defer u.end()
	u.mu.Lock()
	if u.view.AvailableVersion == "" || u.view.State == UpdateRollbackPending {
		u.mu.Unlock()
		return u.fail(&updater.Failure{Code: updater.CodeNotChecked, Message: "Check for an available release first."})
	}
	u.view.State, u.view.Error, u.staged = UpdateInProgress, nil, nil
	u.mu.Unlock()
	staged, f := client.Download(ctx, func(p updater.Progress) {
		total := uint64(p.TotalBytes)
		progress := UpdateProgress{Phase: p.Phase, BytesDone: uint64(p.ReceivedBytes), BytesTotal: &total}
		u.mu.Lock()
		u.view.Progress = &progress
		u.mu.Unlock()
		if u.cfg.Events != nil {
			cp := copyUpdateProgress(progress)
			u.cfg.Events(StreamEvent{Kind: UpdateProgressStream, Update: &cp})
		}
	})
	if f != nil {
		if f.Code == updater.CodeCanceled {
			u.mu.Lock()
			u.view.State, u.view.Progress, u.view.Error = UpdateAvailable, nil, nil
			u.mu.Unlock()
			return ActionResult{}
		}
		return u.fail(f)
	}
	u.logEvent("Update package verified and staged")
	u.mu.Lock()
	u.staged, u.view.State = staged, UpdateReady
	u.mu.Unlock()
	return ActionResult{}
}

func (u *UpdateService) CancelUpdate() ActionResult {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.view.State != UpdateInProgress || u.cancel == nil {
		return ActionResult{}
	}
	u.cancel()
	return ActionResult{}
}

func (u *UpdateService) ClearReleaseHistory() ActionResult {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.busy || u.client == nil || !u.initialized {
		return codedAction("updater_starting", "Updater is not ready to clear its release history.")
	}
	if u.view.State == UpdateRollbackPending {
		return codedAction("rollback_pending", "Recover the previous version first.")
	}
	c, ok := u.client.(interface{ ClearReleaseHistory() *updater.Failure })
	if !ok {
		return codedAction("internal_error", "Updater cannot clear release history.")
	}
	if f := c.ClearReleaseHistory(); f != nil {
		return ActionResult{Error: updateError(f)}
	}
	u.view.Error = nil
	u.view.State = UpdateNone
	return ActionResult{}
}

func (u *UpdateService) DismissUpdate() ActionResult {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.view.State == UpdateRollbackPending {
		return codedAction("rollback_pending", "Resolve the pending rollback before dismissing it.")
	}
	if u.view.LastResultNotice != "" && u.client != nil {
		c, ok := u.client.(interface{ ClearLastResult() *updater.Failure })
		if !ok {
			return codedAction("internal_error", "Updater cannot clear its saved result.")
		}
		if f := c.ClearLastResult(); f != nil {
			return ActionResult{Error: updateError(f)}
		}
	}
	if u.view.State == UpdateNone || u.view.State == UpdateCurrent {
		u.view.LastResultNotice = ""
		return ActionResult{}
	}
	if u.closed || u.client == nil {
		u.view = Update{State: UpdateNone, CurrentVersion: u.view.CurrentVersion, Ready: u.view.Ready, LogPath: u.logPath, RuntimeFollowupRequired: u.view.RuntimeFollowupRequired}
		return ActionResult{}
	}
	if u.busy {
		return ActionResult{Error: updateError(&updater.Failure{Code: updater.CodeBusy, Message: "Cancel or finish the active update action first."})}
	}
	u.staged = nil
	u.view = Update{State: UpdateNone, CurrentVersion: u.view.CurrentVersion, Ready: u.view.Ready, LogPath: u.logPath, RuntimeFollowupRequired: u.view.RuntimeFollowupRequired}
	return ActionResult{}
}

func (u *UpdateService) InstallUpdate() ActionResult {
	client, ctx, r := u.begin()
	if r.Error != nil {
		return r
	}
	defer u.end()
	u.mu.Lock()
	staged, recovery := u.staged, u.view.State == UpdateRollbackPending
	u.mu.Unlock()
	if staged == nil && !recovery {
		return u.fail(&updater.Failure{Code: updater.CodeNotChecked, Message: "Download and verify the update before installing."})
	}
	if u.cfg.Quiesce == nil || u.cfg.Exit == nil {
		return u.fail(&updater.Failure{Code: updater.CodeUnsupported, Message: "Supervisor update shutdown is unavailable."})
	}
	if !recovery {
		if checker, ok := client.(interface {
			Preflight(*updater.Staged) *updater.Failure
		}); ok {
			if f := checker.Preflight(staged); f != nil {
				u.logEvent("Install preflight failed: " + string(f.Code) + ": " + f.Message)
				return ActionResult{Error: updateError(f)}
			}
		}
	}
	pids, release, err := u.cfg.Quiesce(ctx)
	if release != nil {
		defer release()
	}
	if err != nil {
		if coded := ActionResultFromError(err).Error; coded != nil && coded.Code == "update_external_server" {
			return ActionResult{Error: coded}
		}
		return u.fail(&updater.Failure{Code: updater.CodeProcesses, Message: err.Error(), Retry: "Stop active CLI work and retry."})
	}
	// The callback is evaluated by lane C at handoff. Keep the stopped PIDs
	// along with our GUI PID; the helper validates creation times itself.
	u.mu.Lock()
	u.pids = append([]uint32(nil), pids...)
	u.mu.Unlock()
	// The real client's callback closes over this service, not a stale slice.
	var handoff *updater.Handoff
	var f *updater.Failure
	if recovery {
		handoff, f = client.Recover()
	} else {
		handoff, f = client.Install(staged)
	}
	if f != nil {
		if recovery && f.Code == updater.CodeUnsupported && f.Message == "there is no interrupted update to recover" {
			u.mu.Lock()
			u.recovery = false
			u.view = Update{State: UpdateNone, CurrentVersion: u.view.CurrentVersion, Ready: u.view.Ready, LogPath: u.logPath, RuntimeFollowupRequired: u.view.RuntimeFollowupRequired}
			u.mu.Unlock()
			return ActionResult{}
		}
		return u.fail(f)
	}
	if handoff == nil || handoff.HelperPID <= 0 {
		return u.fail(&updater.Failure{Code: updater.CodeProcesses, Message: "Updater returned no helper process."})
	}
	u.logEvent("Update helper started")
	if u.cfg.Exit != nil {
		u.cfg.Exit()
	}
	return ActionResult{}
}

func (u *UpdateService) lastResult(result *updater.InstallResult) {
	u.logEvent("Update result: " + result.Status)
	u.mu.Lock()
	u.view.RuntimeFollowupRequired = result.Status == "installed" && result.RuntimeChanged
	switch result.Status {
	case "installed":
		if !result.ResultDismissed {
			u.view.LastResultNotice = "Updated to " + result.Version + "."
		}
	case "rolled_back":
		u.view.LastResultNotice = "Rolled back to " + u.view.CurrentVersion + "."
	case "rollback_pending":
		u.recovery = true
		u.view.State = UpdateRollbackPending
		u.view.PreviousVersion = previousVersion(u.cfg.Updater.InstallDir)
		u.view.LastResultNotice = "Update rollback is pending. Install retries recovery after stopping CLI work."
	case "failed":
		u.view.State = UpdateFailed
		u.view.LastResultNotice = "Update failed."
		if result.Failure != nil && result.Failure.Code == updater.CodeUpdateStopped {
			// The updater's sentence is already plain and complete.
			u.view.LastResultNotice = ""
		}
	}
	if result.Failure != nil {
		u.view.LastResultNotice = strings.TrimSpace(u.view.LastResultNotice + " " + strings.Join(strings.Fields(result.Failure.Message), " "))
		u.view.Error = updateError(result.Failure)
	}
	u.mu.Unlock()
	if result.Status == "installed" && result.RuntimeChanged {
		u.wg.Add(1)
		RunProtected(func() { defer u.wg.Done(); u.waitForRuntimePreparation() })
		if !result.RuntimeFollowupStarted && u.cfg.PrepareRuntime != nil {
			if marker, ok := u.client.(interface{ MarkRuntimeFollowupStarted() *updater.Failure }); ok {
				if f := marker.MarkRuntimeFollowupStarted(); f != nil {
					u.logEvent("Cannot record runtime plan start: " + f.Message)
				}
			}
			u.logEvent("New runtime required; opening the preparation plan")
			RunProtected(func() {
				if action := u.cfg.PrepareRuntime(); action.Error != nil {
					u.logEvent("Cannot open runtime preparation plan: " + action.Error.Code)
				}
			})
		}
		if result.OldRuntimePath != "" {
			u.wg.Add(1)
			RunProtected(func() { defer u.wg.Done(); u.waitForRuntimeHealth(result.OldRuntimePath) })
		}
	}
}

func (u *UpdateService) waitForRuntimePreparation() {
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	sawPlan := false
	for {
		snap := u.cfg.Snapshot()
		if snap.State == Preparing || snap.Approval != nil {
			sawPlan = true
		} else if sawPlan && snap.State == Stopped && snap.Prepared && snap.LastError == nil && snap.Status != nil && snap.Status.Readiness.Runtime {
			u.RuntimePrepared()
			return
		} else if snap.State == Failed || snap.State == NotPrepared {
			sawPlan = false
		}
		select {
		case <-u.ctx.Done():
			return
		case <-tick.C:
		}
	}
}

// RuntimePrepared is called after the new executable's plan completes with
// approval and a successful CLI result. Decline leaves the requirement set.
func (u *UpdateService) RuntimePrepared() {
	u.mu.Lock()
	defer u.mu.Unlock()
	if !u.view.RuntimeFollowupRequired || u.client == nil {
		return
	}
	marker, ok := u.client.(interface{ MarkRuntimePrepared() *updater.Failure })
	if !ok {
		u.logEvent("Updater cannot persist runtime preparation result")
		return
	}
	if f := marker.MarkRuntimePrepared(); f != nil {
		u.logEvent("Cannot persist runtime preparation result: " + f.Message)
		return
	}
	u.view.RuntimeFollowupRequired = false
	u.logEvent("New runtime preparation completed")
}

func (u *UpdateService) waitForRuntimeHealth(oldPath string) {
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		snap := u.cfg.Snapshot()
		// A fresh CLI status response saying its own instance is ready is the
		// server health request. Never remove an old runtime before that reply.
		if snap.State == Ready && snap.Instance != nil && snap.Instance.Ready && snap.Status != nil {
			if newPath, ok := switchScalar(snap.Status.Paths["runtime"]); ok && !strings.EqualFold(oldPath, newPath) {
				if f := updater.RemoveOldRuntime(oldPath, newPath); f != nil {
					u.logEvent("Old runtime cleanup failed: " + f.Message)
				} else {
					u.logEvent("Old runtime removed after new server health")
				}
				return
			}
		}
		select {
		case <-u.ctx.Done():
			return
		case <-tick.C:
		}
	}
}

func (u *UpdateService) Close() {
	u.mu.Lock()
	u.closed = true
	u.close()
	u.mu.Unlock()
	u.wg.Wait()
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.client != nil {
		u.client.Close()
	}
}
