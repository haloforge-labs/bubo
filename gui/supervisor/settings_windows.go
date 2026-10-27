//go:build windows

package supervisor

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/sys/windows"
)

type Settings struct {
	Version               int    `json:"version"`
	StartGUIAtLogin       bool   `json:"start_gui_at_login"`
	StartServerWithGUI    bool   `json:"start_server_with_gui"`
	CloseToTray           bool   `json:"close_to_tray"`
	InferenceExpanded     bool   `json:"inference_expanded"`
	AllowTightSetup       bool   `json:"allow_tight_setup"`
	TightSetupFingerprint string `json:"tight_setup_fingerprint,omitempty"`
	// Nil preserves default-on behavior for existing settings files. Only an
	// explicit user choice is persisted.
	CheckForUpdates *bool             `json:"check_for_updates,omitempty"`
	Port            int               `json:"port,omitempty"`
	Profile         string            `json:"profile,omitempty"`
	Context         uint64            `json:"context,omitempty"`
	Sessions        uint32            `json:"sessions,omitempty"`
	ModelPath       string            `json:"model_path,omitempty"`
	RuntimePath     string            `json:"runtime_path,omitempty"`
	DataFolder      string            `json:"data_folder,omitempty"`
	ManifestPath    string            `json:"manifest_path,omitempty"`
	NPUPrefill      string            `json:"npu_prefill,omitempty"`
	Effort          string            `json:"effort,omitempty"`
	SwitchOverrides map[string]string `json:"switch_overrides,omitempty"`
	// LanAccess serves the local network (host 0.0.0.0 with an API key).
	// Off by default; a restart applies it. The key is never stored here.
	LanAccess bool `json:"lan_access"`
	// StallSeconds is the no-token-progress warning time (default 120).
	StallSeconds int `json:"stall_seconds,omitempty"`
	// ModelFolders are the folders the user added on the Models page. The CLI scans
	// them (depth 3, .gguf only). Nothing else is scanned.
	ModelFolders []string `json:"model_folders,omitempty"`
	// DownloadFolders are the download folders (data_folder values) this app has
	// used. The download folder decides only where new downloads go; every
	// folder here is still scanned for prepared packages, so a model downloaded
	// before a change stays listed and starts by its manifest path.
	DownloadFolders []string `json:"download_folders,omitempty"`
	LastError       *Failure `json:"last_error,omitempty"`
	// Persistence metadata only, never a frontend preference or CLI fact.
	lastErrorVersion string
}

type persistedSettings struct {
	Settings
	LastErrorVersion string `json:"last_error_version,omitempty"`
}

func DefaultSettings() Settings {
	return Settings{Version: 1, StallSeconds: 120}
}

func (s Settings) UpdatesEnabled() bool {
	return s.CheckForUpdates == nil || *s.CheckForUpdates
}

func tightSetupChanged(before, after Settings) bool {
	return before.ModelPath != after.ModelPath || before.ManifestPath != after.ManifestPath || before.Context != after.Context || before.Sessions != after.Sessions || before.Profile != after.Profile
}
func SettingsPath() string {
	return filepath.Join(os.Getenv("LOCALAPPDATA"), "Bubo", "gui", "settings.json")
}
func LoadSettings(path string) (Settings, string) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return DefaultSettings(), ""
	}
	if err != nil {
		return DefaultSettings(), "Settings could not be read. Defaults are active."
	}
	var saved persistedSettings
	if len(b) > 64<<10 || json.Unmarshal(b, &saved) != nil || saved.Version != 1 || saved.StallSeconds < 0 || saved.StallSeconds > 86400 || !validModelFolders(saved.ModelFolders) || !validModelFolders(saved.DownloadFolders) {
		return DefaultSettings(), "Settings are damaged. Defaults are active."
	}
	s := saved.Settings
	s.lastErrorVersion = saved.LastErrorVersion
	if s.LastError != nil {
		s.LastError = redactedFailure(s.LastError)
	}
	return s, ""
}
func redactedFailure(f *Failure) *Failure {
	if f == nil {
		return nil
	}
	c := f.Code
	source := "supervisor"
	if f.Source == "cli" {
		source = "cli"
	}
	return &Failure{Code: c, Source: source, Time: f.Time, Remedy: safeRemedy(c), Forced: f.Forced, Switch: f.Switch}
}

var settingsMu sync.Mutex

func SaveSettings(path string, s Settings) error {
	settingsMu.Lock()
	defer settingsMu.Unlock()
	if s.Version != 1 || s.StallSeconds < 0 || s.StallSeconds > 86400 || !validModelFolders(s.ModelFolders) || !validModelFolders(s.DownloadFolders) {
		return fmt.Errorf("invalid settings")
	}
	s.LastError = redactedFailure(s.LastError)
	if s.LastError == nil {
		s.lastErrorVersion = ""
	}
	b, err := json.MarshalIndent(persistedSettings{Settings: s, LastErrorVersion: s.lastErrorVersion}, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	dir := filepath.Dir(path)
	if err = os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	writeTemp := func(data []byte) (string, error) {
		f, err := os.CreateTemp(dir, "settings-*.tmp")
		if err != nil {
			return "", err
		}
		name := f.Name()
		if _, err = f.Write(data); err == nil {
			err = f.Sync()
		}
		closeErr := f.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			os.Remove(name)
			return "", err
		}
		return name, nil
	}
	name, err := writeTemp(b)
	if err != nil {
		return err
	}
	defer os.Remove(name)
	if old, readErr := os.ReadFile(path); readErr == nil {
		backup, err := writeTemp(old)
		if err != nil {
			return err
		}
		defer os.Remove(backup)
		if err = replace(backup, path+".bak"); err != nil {
			return err
		}
	} else if !os.IsNotExist(readErr) {
		return readErr
	}
	return replace(name, path)
}
func replace(from, to string) error {
	f, err := windows.UTF16PtrFromString(from)
	if err != nil {
		return err
	}
	t, err := windows.UTF16PtrFromString(to)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(f, t, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}

// StallTimeout is the stall-watchdog time for these settings: 120 s when unset.
func (s Settings) StallTimeout() time.Duration {
	if s.StallSeconds <= 0 {
		return DefaultStallTimeout
	}
	return time.Duration(s.StallSeconds) * time.Second
}

// turnLANOff saves lan_access=false after the key store failed, so the next
// start is a loopback start and the setting shows the truth.
func (s *Supervisor) turnLANOff() {
	if s.config.SettingsPath == "" {
		return
	}
	cur, _ := LoadSettings(s.config.SettingsPath)
	cur.LanAccess = false
	_ = SaveSettings(s.config.SettingsPath, cur)
}

// validModelFolders accepts at most MaxModelFolders clean, absolute folder paths.
func validModelFolders(folders []string) bool {
	if len(folders) > MaxModelFolders {
		return false
	}
	for _, f := range folders {
		if c, err := NormalizeModelFolderPath(f); err != nil || c != f {
			return false
		}
	}
	return true
}
