package supervisor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"bubo/gui/updater"
)

type Backend interface {
	Snapshot() Snapshot
	Logs() []string
	Action(string, Settings, string) <-chan ResultReply
	Save(Settings) error
	Notice() string
	Diagnostic() string
	BundlePreview() string
	SetVisible(bool)
}

// NativeBackend delegates every CLI process to the supervisor. It persists
// requested command settings only; all effective values come from the CLI.
type NativeBackend struct {
	mu                                      sync.RWMutex
	s                                       *Supervisor
	config                                  Config
	settings                                Settings
	guiExe                                  string
	notice, noticeCode, diagnostic, preview string
	busy                                    bool
	updating                                bool
	exiting                                 bool
	exited                                  bool
	busyDone                                chan struct{}
	eventSink                               func(StreamEvent)
}

func NewBackend(c Config, settings Settings, guiExe string) (*NativeBackend, error) {
	c.ProductVersion, _ = updater.InstalledVersion(GUIPackageRoot(guiExe))
	c.RuntimeDir = settings.RuntimePath
	s, err := New(c)
	if err != nil {
		return nil, err
	}
	s.SetModelSelection(settings.ManifestPath, settings.ModelPath)
	return &NativeBackend{s: s, config: c, settings: settings, guiExe: guiExe}, nil
}
func (b *NativeBackend) Snapshot() Snapshot {
	b.mu.RLock()
	s := b.s
	b.mu.RUnlock()
	return s.Snapshot()
}
func (b *NativeBackend) Logs() []string { b.mu.RLock(); s := b.s; b.mu.RUnlock(); return s.Logs() }
func (b *NativeBackend) Notice() string { b.mu.RLock(); defer b.mu.RUnlock(); return b.notice }
func (b *NativeBackend) CodedNotice() Notice {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.notice == "" {
		return Notice{}
	}
	return Notice{Code: b.noticeCode, Detail: b.notice}
}
func (b *NativeBackend) Diagnostic() string    { b.mu.RLock(); defer b.mu.RUnlock(); return b.diagnostic }
func (b *NativeBackend) BundlePreview() string { b.mu.RLock(); defer b.mu.RUnlock(); return b.preview }
func (b *NativeBackend) SetVisible(v bool)     { b.mu.RLock(); defer b.mu.RUnlock(); b.s.SetVisible(v) }

// Registry is the engine switch registry the CLI listed, or nil until it arrives.
func (b *NativeBackend) Registry() *Switches {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.s.Registry()
}

func (b *NativeBackend) SetEventSink(sink func(StreamEvent)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.eventSink = sink
	b.s.SetEventSink(sink)
}

// A model job completes independently of the lifecycle action gate. Deliver
// its success to the supervisor owner even while a user action is pending.
func (b *NativeBackend) modelReady() {
	b.mu.RLock()
	s := b.s
	b.mu.RUnlock()
	s.emit(update{s.Snapshot().Generation, "model-ready", nil})
}

func (b *NativeBackend) Save(settings Settings) error {
	if settings.RuntimePath != "" && !filepath.IsAbs(settings.RuntimePath) {
		return appError("invalid_arguments", "Select an absolute runtime folder.")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if tightSetupChanged(b.settings, settings) {
		settings.AllowTightSetup = false
		settings.TightSetupFingerprint = ""
	}
	if b.exiting {
		return appError("closing", "The GUI is exiting.")
	}
	if b.updating {
		return appError("update_installing", "An update installation is active.")
	}
	var removed []string
	registry := b.s.Registry()
	settings.SwitchOverrides, removed = dropStaleOverrides(settings.SwitchOverrides, registry)
	if registry != nil {
		if _, _, err := switchCommand(settings.SwitchOverrides, registry); err != nil {
			return err
		}
	}
	settings.LastError = b.s.Snapshot().LastError
	settings.lastErrorVersion = b.config.ProductVersion
	if settings.StartGUIAtLogin != b.settings.StartGUIAtLogin {
		if err := SetLoginStart(settings.StartGUIAtLogin, b.guiExe); err != nil {
			return appError("login_start_failed", err.Error())
		}
	}
	if err := SaveSettings(b.config.SettingsPath, settings); err != nil {
		if settings.StartGUIAtLogin != b.settings.StartGUIAtLogin {
			_ = SetLoginStart(b.settings.StartGUIAtLogin, b.guiExe)
		}
		return appError("filesystem", err.Error())
	}
	previous := b.settings
	b.settings = settings
	b.s.SetStallTimeout(settings.StallTimeout())
	// Status reports readiness for the model the user selected.
	b.s.SetModelSelection(settings.ManifestPath, settings.ModelPath)
	if len(removed) > 0 {
		b.notice, b.noticeCode = "Dropped stale overrides: "+strings.Join(removed, ", "), "setting_removed"
	} else {
		b.notice, b.noticeCode = settingsNotice(previous, settings, b.s.Snapshot(), registry)
	}
	return nil
}

// settingsNotice is "Restart to apply these settings." only when a running
// server would start differently from how it runs now: a setting that the next
// start passes to the server differs from what the running server has. The
// model download folder, the start-at-login choice and every save with no
// running server give no notice. Without the CLI's switch registry the app
// cannot compare values, so it says so only when a serve setting changed.
func settingsNotice(previous, next Settings, snap Snapshot, registry *Switches) (detail, code string) {
	const restart, restartCode = "Restart to apply these settings.", "settings_saved"
	if snap.Instance == nil {
		return "", ""
	}
	if registry == nil {
		if serveRequest(previous) != serveRequest(next) {
			return restart, restartCode
		}
		return "", ""
	}
	changed := serveRequest(previous) != serveRequest(next)
	for _, view := range switchOverrideViews(ViewInput{Saved: next, Registry: registry, Snapshot: snap}) {
		switch {
		case view.RestartState == SwitchPending:
			return restart, restartCode
		case view.RestartState == SwitchUnknownUntilRestart && (view.Saved == nil || changed):
			// A cleared override: the next start uses the CLI default. Or the
			// running value is not reported and this save changed a serve setting.
			return restart, restartCode
		}
	}
	return "", ""
}

// serveRequest is what the next start passes to the server, as one comparable
// text: the serve arguments and the saved switch overrides.
func serveRequest(s Settings) string {
	return fmt.Sprintf("%+v|%v", serveArguments(s, requestedHost(s), s.Port), s.SwitchOverrides)
}

func prepareArgs(s Settings) []string {
	var args []string
	if s.DataFolder != "" {
		args = append(args, "--output", s.DataFolder)
	}
	return args
}
func (b *NativeBackend) Action(action string, settings Settings, path string) <-chan ResultReply {
	reply := make(chan ResultReply, 1)
	b.mu.Lock()
	if b.updating && action != "exit" {
		b.mu.Unlock()
		reply <- ResultReply{Err: appError("update_installing", "An update installation is active.")}
		return reply
	}
	if action == "exit" && b.exited {
		b.mu.Unlock()
		reply <- ResultReply{}
		return reply
	}
	if b.exiting && action != "exit" {
		b.mu.Unlock()
		reply <- ResultReply{Err: appError("closing", "The GUI is exiting.")}
		return reply
	}
	if action == "exit" {
		b.exiting = true
	}
	if b.busy && action != "stop" && action != "cancel" && action != "exit" {
		b.mu.Unlock()
		reply <- ResultReply{Err: appError("busy", "An action is active.")}
		return reply
	}
	tracked := action != "stop" && action != "cancel" && action != "exit"
	if tracked {
		b.busy = true
		b.busyDone = make(chan struct{})
	}
	pending := b.busyDone
	s := b.s
	config := b.config
	b.mu.Unlock()
	RunProtected(func() {
		var err error
		if action == "exit" && pending != nil {
			<-pending
			b.mu.RLock()
			s = b.s
			b.mu.RUnlock()
		}
		switch action {
		case "start", "restart", "prepare":
			c := config
			c.RuntimeDir = settings.RuntimePath
			c.Port = settings.Port
			if c.RuntimeDir != config.RuntimeDir || c.Port != config.Port {
				if v := s.Snapshot(); v.State == AttachedExternal {
					err = appError("external_read_only", "Detach the external server before changing the runtime folder or port.")
					break
				}
				if err = (<-s.Exit()).Err; err != nil {
					break
				}
				b.mu.Lock()
				if b.exiting {
					b.mu.Unlock()
					err = appError("closing", "The GUI is exiting.")
					break
				}
				var next *Supervisor
				c.StallTimeout = settings.StallTimeout()
				next, err = New(c)
				if err != nil {
					b.mu.Unlock()
					break
				}
				b.s = next
				next.SetModelSelection(settings.ManifestPath, settings.ModelPath)
				next.SetEventSink(b.eventSink)
				b.config = c
				b.mu.Unlock()
				s = next
			}
			if action == "prepare" {
				err = (<-s.Prepare(prepareArgs(settings)...)).Err
			} else if action == "restart" && s.Snapshot().Instance != nil {
				err = (<-s.Restart(startArgs(settings)...)).Err
			} else {
				err = (<-s.Start(startArgs(settings)...)).Err
			}
		case "stop", "cancel":
			err = (<-s.Stop()).Err
		case "exit":
			err = (<-s.Exit()).Err
		case "approve":
			v := s.Snapshot()
			if v.Approval == nil || v.Approval.Digest == "" {
				err = appError("protocol_mismatch", "The CLI did not supply the approval digest.")
				break
			}
			if !completePlan(v.PlanParts, v.Approval) {
				err = &ProtocolError{Detail: "The preparation plan is incomplete or duplicated."}
				break
			}
			err = (<-s.Consent(true)).Err
		case "decline":
			err = (<-s.Consent(false)).Err
		case "dismiss":
			err = (<-s.DismissError()).Err
		case "manage":
			err = (<-s.Manage()).Err
		case "select":
			err = (<-s.SelectInstance(path)).Err
		case "diagnose", "preview", "bundle":
			var args []string
			if action == "preview" {
				args = []string{"--preview"}
			}
			if action == "bundle" {
				b.mu.RLock()
				hasPreview := b.preview != ""
				b.mu.RUnlock()
				if !hasPreview {
					err = appError("diagnostic_failed", "Read the CLI bundle preview before saving locally.")
					break
				}
				if !filepath.IsAbs(path) {
					err = appError("invalid_arguments", "Select an absolute local ZIP path.")
					break
				}
				args = []string{"--bundle", path}
			}
			r := (<-s.Diagnose(args...))
			err = r.Err
			if err == nil && action != "diagnose" {
				var payload map[string]json.RawMessage
				_ = json.Unmarshal(r.Event.Payload, &payload)
				var files []string
				var preview map[string]string
				_ = json.Unmarshal(payload["files"], &files)
				_ = json.Unmarshal(payload["text_preview"], &preview)
				if (r.Event.Type != "support_bundle" && r.Event.Type != "diagnose_artifact") || len(files) == 0 || len(preview) == 0 {
					err = appError("diagnostic_failed", "The CLI did not supply a support bundle file list and text preview.")
				}
			}
			if err == nil {
				text := string(r.Event.Payload)
				var pretty bytes.Buffer
				if json.Indent(&pretty, r.Event.Payload, "", "  ") == nil {
					text = pretty.String()
				}
				b.mu.Lock()
				if action == "diagnose" {
					b.diagnostic = text
				} else if action == "preview" {
					b.preview = text
				} else {
					b.notice, b.noticeCode = path, "support_bundle_saved"
				}
				b.mu.Unlock()
			}
		default:
			err = appError("internal_error", "Unknown backend action: "+action)
		}
		b.mu.Lock()
		if action == "exit" && err == nil {
			b.exited = true
		}
		if action == "exit" && err != nil {
			b.exiting = false
		}
		if tracked {
			b.busy = false
			close(b.busyDone)
			b.busyDone = nil
		}
		if err != nil {
			b.notice = err.Error()
			b.noticeCode = ActionResultFromError(err).Error.Code
		}
		b.mu.Unlock()
		reply <- ResultReply{Err: err}
	})
	return reply
}

func DefaultCLIPath() string {
	exe, _ := os.Executable()
	return filepath.Join(GUIPackageRoot(exe), "bin", "bubo.exe")
}

// GUIPackageRoot supports the shipped bin/ layout and root-level developer
// builds. The runtime override is independent of the installed package.
func GUIPackageRoot(exe string) string {
	dir := filepath.Dir(exe)
	if strings.EqualFold(filepath.Base(dir), "bin") {
		return filepath.Dir(dir)
	}
	return dir
}

func completePlan(parts []Plan, approval *Approval) bool {
	if approval == nil || len(approval.Components) == 0 {
		return false
	}
	for _, component := range approval.Components {
		var total uint64
		digest := ""
		seen := map[uint64]bool{}
		for _, part := range parts {
			if part.Component != component {
				continue
			}
			if total == 0 {
				total = part.Parts
				digest = part.Digest
			}
			if part.Parts != total || part.Digest != digest || part.Part >= total || seen[part.Part] {
				return false
			}
			seen[part.Part] = true
		}
		if total == 0 || uint64(len(seen)) != total {
			return false
		}
	}
	return true
}
