//go:build windows

package supervisor

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// PreferenceValues is a partial UserOverrides input. Pointers mean omitted;
// clearing is exclusively expressed by PreferencesPatch.Clear, never by null.
// Settings metadata and switch overrides are not preference fields.
type PreferenceValues struct {
	Port               *int    `json:"port,omitempty"`
	Profile            *string `json:"profile,omitempty"`
	Context            *uint64 `json:"context,omitempty"`
	Sessions           *uint32 `json:"sessions,omitempty"`
	RuntimePath        *string `json:"runtime_path,omitempty"`
	ModelPath          *string `json:"model_path,omitempty"`
	DataFolder         *string `json:"data_folder,omitempty"`
	ManifestPath       *string `json:"manifest_path,omitempty"`
	NPUPrefill         *string `json:"npu_prefill,omitempty"`
	Effort             *string `json:"effort,omitempty"`
	StartGUIAtLogin    *bool   `json:"start_gui_at_login,omitempty"`
	StartServerWithGUI *bool   `json:"start_server_with_gui,omitempty"`
	CloseToTray        *bool   `json:"close_to_tray,omitempty"`
	InferenceExpanded  *bool   `json:"inference_expanded,omitempty"`
	AllowTightSetup    *bool   `json:"allow_tight_setup,omitempty"`
	LANAccess          *bool   `json:"lan_access,omitempty"`
	StallSeconds       *int    `json:"stall_seconds,omitempty"`
	CheckForUpdates    *bool   `json:"check_for_updates,omitempty"`
}

type PreferencesPatch struct {
	Set   PreferenceValues `json:"set"`
	Clear []string         `json:"clear"`
	// Registry controls submit CLI scalars; the supervisor owns their mapping
	// to app preferences. PatchPreferences remains the only preference writer.
	SwitchValues []SwitchOverride `json:"switch_values,omitempty"`
}

func normalizePreferencePatch(patch PreferencesPatch) (PreferencesPatch, error) {
	b, err := json.Marshal(patch.Set)
	if err != nil {
		return patch, err
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(b, &fields); err != nil {
		return patch, err
	}
	var overrides []SwitchOverride
	for _, value := range patch.SwitchValues {
		var owner *serveArgument
		for _, arg := range serveArguments(Settings{}, "", 0) {
			if arg.name == value.Name && arg.preference != "" {
				copy := arg
				owner = &copy
				break
			}
		}
		if owner == nil {
			overrides = append(overrides, value)
			continue
		}
		if fields[owner.preference] != nil {
			return patch, appError("internal_error", "Invalid preference switch write: "+value.Name)
		}
		var raw []byte
		switch preferenceType(owner.preference) {
		case "boolean":
			// LAN access has one canonical host mapping, shared with Start.
			if owner.preference != "lan_access" {
				return patch, appError("internal_error", "Unknown boolean preference switch: "+value.Name)
			}
			if value.Value == requestedHost(Settings{LanAccess: true}) {
				raw = []byte("true")
			} else if value.Value == requestedHost(Settings{LanAccess: false}) {
				raw = []byte("false")
			} else {
				return patch, appError("use_connect_for_lan", "This listen address cannot be set by the app's local-network preference.")
			}
		case "string":
			raw, err = json.Marshal(value.Value)
		default:
			// encoding/json checks the app transport's numeric shape; the CLI
			// still validates engine limits when the saved setup starts.
			raw = []byte(value.Value)
			if len(raw) == 0 || raw[0] != '-' && (raw[0] < '0' || raw[0] > '9') || !json.Valid(raw) {
				return patch, appError("invalid_arguments", "Invalid numeric preference value: "+value.Name)
			}
		}
		if err != nil {
			return patch, err
		}
		fields[owner.preference] = raw
	}
	b, err = json.Marshal(fields)
	if err != nil {
		return patch, err
	}
	if err = json.Unmarshal(b, &patch.Set); err != nil {
		return patch, appError("invalid_arguments", "Invalid preference transport value: "+err.Error())
	}
	patch.SwitchValues = overrides
	return patch, nil
}

// Registry removal also blocks the preference route of an owned switch.
// Clear remains available to recover an obsolete saved request.
func validatePreferenceRegistryWrites(patch PreferencesPatch, registry *Switches) error {
	if registry == nil {
		return nil
	}
	b, err := json.Marshal(patch.Set)
	if err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	for _, arg := range serveArguments(Settings{}, "", 0) {
		if arg.preference == "" || fields[arg.preference] == nil {
			continue
		}
		for _, row := range registry.Switches {
			if row.Name == arg.name && row.Class == "REMOVED" {
				return appError("setting_removed", "The CLI removed this setting: "+row.Name)
			}
		}
	}
	return nil
}

// ApplyPreferencesPatch merges only named changes. encoding/json handles the
// common wire names and types; no engine limits or defaults are duplicated.
func ApplyPreferencesPatch(current Settings, patch PreferencesPatch) (Settings, error) {
	b, err := json.Marshal(current)
	if err != nil {
		return current, err
	}
	var fields map[string]json.RawMessage
	if err = json.Unmarshal(b, &fields); err != nil {
		return current, err
	}
	b, err = json.Marshal(patch.Set)
	if err != nil {
		return current, err
	}
	var set map[string]json.RawMessage
	if err = json.Unmarshal(b, &set); err != nil {
		return current, err
	}
	allowed := map[string]bool{}
	t := reflect.TypeOf(patch.Set)
	for i := 0; i < t.NumField(); i++ {
		allowed[strings.Split(t.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	for _, name := range patch.Clear {
		if !allowed[name] || set[name] != nil {
			return current, appError("internal_error", fmt.Sprintf("Invalid preference clear: %s", name))
		}
		delete(fields, name)
	}
	for name, value := range set {
		fields[name] = value
	}
	b, err = json.Marshal(fields)
	if err != nil {
		return current, err
	}
	var next Settings
	if err = json.Unmarshal(b, &next); err != nil {
		return current, err
	}
	next.Version = DefaultSettings().Version
	if tightSetupChanged(current, next) {
		next.AllowTightSetup = false
		next.TightSetupFingerprint = ""
	}
	return next, nil
}
