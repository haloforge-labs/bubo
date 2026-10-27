//go:build windows

package supervisor

import (
	"context"
	"strings"
	"sync"
	"time"
)

// LanState is everything a window may know about local network access and the
// API key. It never holds the key: only whether one exists and a mask. The key
// itself comes only from RevealKey, called by an explicit Show or Copy action.
type LanState struct {
	// LanAccess is the saved setting; a restart applies it.
	LanAccess bool `json:"lan_access"`
	// LanActive is true while the running server is bound to the network.
	LanActive bool `json:"lan_active"`
	// APIKeyPresent is true when the credential store holds a key.
	APIKeyPresent bool `json:"api_key_present"`
	// APIKeyMask is shown in place of the key ("" when no key exists).
	APIKeyMask string `json:"api_key_mask"`
	// CopyWithKey says whether "Copy address and key" is shown: only with LAN access on and a key.
	CopyWithKey bool `json:"copy_with_key"`
	// KeyRestartNeeded is true after Reset key while a network server still runs the old key.
	KeyRestartNeeded bool `json:"key_restart_needed"`
	// Unavailable is a message when the key store failed (LAN access then stays off).
	Unavailable string `json:"unavailable,omitempty"`
	// Addresses lists where other devices reach the server. Empty unless LAN access is on.
	// It never holds the API key.
	Addresses []LanAddress `json:"addresses"`
	// AddressesUnavailable is set when LAN access is on and no address was found.
	AddressesUnavailable string `json:"addresses_unavailable,omitempty"`
}

// AppService owns app policy and state. Hooks perform the OS operations.
type AppService struct {
	view              ViewBuilder
	estimate          savedEstimateState
	appliedSettings   *Settings
	appliedGeneration uint64
	mu                sync.RWMutex
	backend           Backend
	settings          Settings
	lease             *SingleInstance
	done              chan struct{}
	stopOnce          sync.Once
	warning           string
	warningCode       string
	closing, exited   bool
	quit              func(context.Context)
	show              func(context.Context)
	hide              func(context.Context)
	ctx               context.Context
	events            func(context.Context, StreamEvent)
	streamMu          sync.Mutex
	lastStreamAt      time.Time
	trayAction        string
	trayAvailable     bool
	tray              TrayPlatform
	trayStop          func()
	keys              *KeyManager
	lanNote           string
	keyStaleGen       uint64
	keyStale          bool
	adapters          adapterSource
	models            *ModelService
	dialogs           PathDialogs
	folders           FolderOpener
	updates           *UpdateService
	windowReady       chan struct{}
	readyOnce         sync.Once
	// folderMu serialises changes of the model folder list and Save.
	folderMu sync.Mutex
}

func NewAppService(b Backend, settings Settings, lease *SingleInstance, warning string, hooks AppHooks) *AppService {
	return &AppService{backend: b, settings: settings, lease: lease, done: make(chan struct{}), windowReady: make(chan struct{}), warning: warning, warningCode: "internal_error", quit: hooks.Quit, show: hooks.Show, hide: hooks.Hide, trayAvailable: true, tray: hooks.Tray, keys: NewKeyManager(nil), adapters: winAdapters{}, events: hooks.Events, dialogs: hooks.Dialogs, folders: hooks.Folders}
}

func (a *AppService) SetStartupNotice(code, detail string) {
	a.mu.Lock()
	a.warning, a.warningCode = detail, code
	a.mu.Unlock()
}

// Preference writes share tray and credential checks and side effects. folderMu
// also serializes read/merge/write patches so concurrent partial saves compose.
func (a *AppService) saveSettingsLocked(settings Settings) error {
	a.mu.RLock()
	trayAvailable := a.trayAvailable
	if tightSetupChanged(a.settings, settings) {
		settings.AllowTightSetup = false
		settings.TightSetupFingerprint = ""
	}
	settings.ModelFolders = append([]string(nil), a.settings.ModelFolders...)
	a.mu.RUnlock()
	var removed []string
	if provider, ok := a.backend.(registryProvider); ok {
		settings.SwitchOverrides, removed = dropStaleOverrides(settings.SwitchOverrides, provider.Registry())
	}
	trayDropped := settings.CloseToTray && !trayAvailable
	if trayDropped {
		settings.CloseToTray = false
	}
	// Local network access needs a stored key. If the store cannot hold it,
	// the setting stays off: there is no plaintext fallback.
	var lanErr error
	if settings.LanAccess {
		if _, err := a.keys.Ensure(); err != nil {
			settings.LanAccess = false
			lanErr = err
		}
	}
	if err := a.backend.Save(settings); err != nil {
		return err
	}
	a.mu.Lock()
	a.settings = settings
	a.warning, a.warningCode = "", ""
	if len(removed) > 0 {
		a.warning = "Dropped stale overrides: " + strings.Join(removed, ", ")
		a.warningCode = "setting_removed"
	}
	if trayDropped {
		a.warning = "The saved close-to-tray choice was turned off because the tray is unavailable."
		a.warningCode = "tray_unavailable"
	}
	if lanErr != nil {
		a.lanNote = lanErr.Error()
	} else {
		a.lanNote = ""
	}
	a.mu.Unlock()
	return lanErr
}

// RevealKey returns the API key only for an explicit Show or Copy-with-key
// action. The View and ordinary snippet copies contain no credential.
func (a *AppService) RevealKey() StringResult { return StringResultFrom(a.keys.Reveal()) }

// ResetKey replaces the stored key with a new one and drops the old one.
// A server already running with the old key needs a restart.
func (a *AppService) ResetKey() ActionResult { return ActionResultFromError(a.resetKey()) }

func (a *AppService) resetKey() error {
	if err := a.keys.Reset(); err != nil {
		a.mu.Lock()
		a.lanNote = err.Error()
		a.mu.Unlock()
		return err
	}
	snap := a.backend.Snapshot()
	a.mu.Lock()
	a.lanNote = ""
	a.keyStale, a.keyStaleGen = true, snap.Generation
	a.mu.Unlock()
	return nil
}

// runAction checks the same saved-setup availability published to both clients.
func (a *AppService) runAction(action, path string) error {
	if action == "exit" {
		return appError("internal_error", "Close the window to stop GUI-owned work and exit.")
	}
	a.folderMu.Lock()
	a.mu.RLock()
	settings, closing := a.settings, a.closing
	a.mu.RUnlock()
	if closing {
		a.folderMu.Unlock()
		return appError("closing", "The GUI is exiting.")
	}
	if action == "start" || action == "restart" {
		if provider, ok := a.backend.(registryProvider); ok {
			_, removed := dropStaleOverrides(settings.SwitchOverrides, provider.Registry())
			if len(removed) > 0 {
				if err := a.saveSettingsLocked(settings); err != nil {
					a.folderMu.Unlock()
					return err
				}
				a.mu.RLock()
				settings = a.settings
				a.mu.RUnlock()
			}
		}
	}
	if action == "start" || action == "stop" || action == "restart" {
		// A just-saved setup starts its estimate asynchronously for the View.
		// A launch waits for that same check instead of rejecting the click
		// while the new saved values are still being checked.
		if (action == "start" || action == "restart") && a.models != nil {
			deadline := time.Now().Add(a.models.cfg.CallTimeout + time.Second)
			for {
				snap := a.backend.Snapshot()
				estimate, estimateErr := a.savedEstimate(settings, snap.Status, snap.Generation)
				if estimate != nil || estimateErr != nil || snap.Status == nil {
					break
				}
				if time.Now().After(deadline) {
					a.folderMu.Unlock()
					return appError("estimate_unavailable", "The memory check did not finish. Try again.")
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
		v := a.GetView()
		available := v.Actions.Start
		if action == "stop" {
			available = v.Actions.Stop
		}
		if action == "restart" {
			available = v.Actions.Restart
		}
		if !available.Allowed {
			a.folderMu.Unlock()
			return &actionUnavailableError{available}
		}
	}
	reply := a.backend.Action(action, settings, path)
	a.folderMu.Unlock()
	err := (<-reply).Err
	if err == nil && (action == "start" || action == "restart") {
		generation := a.backend.Snapshot().Generation
		a.mu.Lock()
		a.appliedSettings, a.appliedGeneration = &settings, generation
		a.mu.Unlock()
	}
	return err
}

func (a *AppService) clearedAppliedPreference(saved Settings, generation uint64) bool {
	a.mu.RLock()
	applied, appliedGeneration := a.appliedSettings, a.appliedGeneration
	a.mu.RUnlock()
	if applied == nil || generation != appliedGeneration {
		return false
	}
	previous := serveArguments(*applied, requestedHost(*applied), applied.Port)
	current := serveArguments(saved, requestedHost(saved), saved.Port)
	for i, old := range previous {
		if old.preference != "" && old.value != "" && current[i].value == "" {
			return true
		}
	}
	return false
}

type actionUnavailableError struct{ ActionAvailability }

func (e *actionUnavailableError) Error() string     { return e.Reason }
func (e *actionUnavailableError) ErrorCode() string { return e.ReasonCode }

func (a *AppService) SetVisible(visible bool) ActionResult {
	a.mu.RLock()
	ctx := a.ctx
	a.mu.RUnlock()
	if visible {
		if a.show == nil {
			return codedAction("internal_error", "The native window is not attached.")
		}
		a.show(ctx)
	} else {
		if a.hide == nil {
			return codedAction("internal_error", "The native window is not attached.")
		}
		a.hide(ctx)
	}
	a.backend.SetVisible(visible)
	return ActionResult{}
}

func (a *AppService) Startup(ctx context.Context) {
	a.mu.Lock()
	a.ctx = ctx
	a.mu.Unlock()
	if a.updates != nil {
		a.updates.Start(a.windowReady)
	}
	if p, ok := a.backend.(streamProvider); ok {
		p.SetEventSink(a.emitStream)
	}
	stop, err := a.tray.Start(a.fromTray, func() TrayMenu { return a.TrayMenu() })
	a.mu.Lock()
	a.trayStop = stop
	if err != nil {
		a.trayAvailable = false
		a.warning = "System tray failed: " + err.Error() + ". Keep this window open while the server runs."
		a.warningCode = "tray_unavailable"
	}
	a.mu.Unlock()
	a.mu.RLock()
	autoStart := a.settings.StartServerWithGUI
	a.mu.RUnlock()
	a.backend.SetVisible(true)
	go func() {
		defer ReportPanic()
		tick := time.NewTicker(200 * time.Millisecond)
		defer tick.Stop()
		var show <-chan struct{}
		if a.lease != nil {
			show = a.lease.ShowWindow
		}
		for {
			select {
			case <-ctx.Done():
				return
			case <-a.done:
				return
			case _, ok := <-show:
				if !ok {
					show = nil
					continue
				}
				a.show(ctx)
				a.backend.SetVisible(true)
			case <-tick.C:
				if autoStart && a.backend.Snapshot().Initialized && (a.updates == nil || a.updates.StartupReady()) {
					v := a.GetView()
					a.mu.RLock()
					estimating := a.estimate.busy
					a.mu.RUnlock()
					if !v.Actions.Start.Allowed && v.Actions.Start.ReasonCode == "estimate_unavailable" && estimating {
						continue
					}
					autoStart = false
					RunProtected(func() { a.fromTray("start") })
				}
			}
		}
	}()
}

func (a *AppService) DOMReady(context.Context) {
	a.readyOnce.Do(func() { close(a.windowReady) })
}

// The supervisor stopped owned work and handed off to the helper. Bypass the
// close-to-tray choice so the helper can wait for this GUI to exit.
func (a *AppService) QuitForUpdate() {
	a.mu.Lock()
	a.closing, a.exited = true, true
	ctx := a.ctx
	a.mu.Unlock()
	a.quit(ctx)
}

// Keep the GUI and the mutex alive until Exit has finished stopping/draining
// the supervisor's owned Job Objects. An attached external server is untouched.
func (a *AppService) BeforeClose(ctx context.Context) bool {
	a.mu.RLock()
	hideToTray := a.settings.CloseToTray && a.trayAvailable && !a.closing && !a.exited
	a.mu.RUnlock()
	if hideToTray {
		// Close-to-tray (off by default): the window hides, the process, the
		// tray and any server keep running. Quit in the tray menu or the
		// close dialog still follow the explicit exit rule below.
		a.hide(ctx)
		a.backend.SetVisible(false)
		return true
	}
	return a.closeFlow(ctx)
}

// A native close cannot wait for a page script to draw or answer a dialog.
// Closing stops owned work; the explicit close-to-tray preference above is the
// keep-running choice. An attached external server remains untouched.
func (a *AppService) closeFlow(ctx context.Context) bool {
	return a.exit(ctx)
}

// Tray Stop and Restart use the same panel confirmations as the visible buttons.
func (a *AppService) fromTray(action string) {
	a.mu.RLock()
	ctx := a.ctx
	a.mu.RUnlock()
	switch action {
	case "show":
		a.show(ctx)
		a.backend.SetVisible(true)
	case "quit":
		a.exit(ctx)
	case "stop", "restart":
		a.mu.Lock()
		a.trayAction = action
		a.mu.Unlock()
		a.show(ctx)
		a.backend.SetVisible(true)
	case "start":
		if err := a.runAction(action, ""); err != nil {
			a.mu.Lock()
			a.warning = "Start failed: " + err.Error()
			a.warningCode = ActionResultFromError(err).Error.Code
			a.mu.Unlock()
		}
	}
}

func (a *AppService) TakeTrayAction() StringResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	action := a.trayAction
	a.trayAction = ""
	return StringResult{Value: action}
}

func (a *AppService) exit(ctx context.Context) bool {
	a.mu.Lock()
	if a.exited {
		a.mu.Unlock()
		return false
	}
	if a.closing {
		a.mu.Unlock()
		return true
	}
	a.closing = true
	settings := a.settings
	a.mu.Unlock()
	go func() {
		defer ReportPanic()
		err := (<-a.backend.Action("exit", settings, "")).Err
		a.mu.Lock()
		if err != nil {
			a.closing = false
			a.warning = "Exit failed: " + err.Error()
			a.warningCode = "internal_error"
			a.mu.Unlock()
			return
		}
		a.exited = true
		a.mu.Unlock()
		a.quit(ctx)
	}()
	return true
}

func (a *AppService) Shutdown(context.Context) {
	a.stopOnce.Do(func() {
		close(a.done)
		if a.updates != nil {
			a.updates.Close()
		}
		a.mu.RLock()
		stop := a.trayStop
		a.mu.RUnlock()
		if stop != nil {
			stop()
		}
		if a.models != nil {
			a.models.Close() // a running download is cancelled; partial files can be resumed
		}
		a.mu.RLock()
		settings := a.settings
		a.mu.RUnlock()
		_ = (<-a.backend.Action("exit", settings, "")).Err
	})
}

// AppHooks executes OS operations. AppService owns all decisions and state;
// Wails supplies the window, tray, dialog and event adapters.
type AppHooks struct {
	Quit, Show, Hide func(context.Context)
	Tray             TrayPlatform
	Dialogs          PathDialogs
	Folders          FolderOpener
	Events           func(context.Context, StreamEvent)
}

func (a *AppService) UseKeys(keys *KeyManager)          { a.keys = keys }
func (a *AppService) UseUpdates(updates *UpdateService) { a.updates = updates }
func (a *AppService) Models() *ModelService             { return a.models }
func (a *AppService) UpdatesEnabled() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.settings.UpdatesEnabled()
}
