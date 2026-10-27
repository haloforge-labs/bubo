//go:build windows

package supervisor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
)

// A consent applies to the concrete setup identified by the CLI. Empty saved
// fields select CLI defaults, so their current effective values belong in the
// key too. Missing CLI facts cannot establish which setup was approved.
func tightSetupFingerprint(saved Settings, st *Status, registries ...*Switches) (string, bool) {
	if st == nil {
		return "", false
	}
	model := ""
	switch {
	case saved.ModelPath != "":
		model = "model:" + saved.ModelPath
	case saved.ManifestPath != "":
		model = "manifest:" + saved.ManifestPath
	case st.PreparedDefault != nil && *st.PreparedDefault != "":
		model = "default:" + *st.PreparedDefault
	default:
		return "", false
	}
	profile := saved.Profile
	if profile == "" {
		if row, ok := SettingRow(st.Effective, "profile"); ok {
			if json.Unmarshal(row.Effective, &profile) != nil || profile == "" {
				return "", false
			}
		} else if len(registries) > 0 {
			var known bool
			profile, known = tightRegistryDefault(registries[0], "--profile", "")
			if !known {
				return "", false
			}
		} else {
			return "", false
		}
	}
	context := saved.Context
	if context == 0 {
		value, ok := tightDefaultCount(registries, st, "--context", "context", profile)
		if !ok || value == 0 {
			return "", false
		}
		context = value
	}
	sessions := saved.Sessions
	if sessions == 0 {
		value, ok := tightDefaultCount(registries, st, "--sessions", "sessions", profile)
		if !ok || value == 0 || value > uint64(^uint32(0)) {
			return "", false
		}
		sessions = uint32(value)
	}
	data, err := json.Marshal(struct {
		Model    string `json:"model"`
		Context  uint64 `json:"context"`
		Sessions uint32 `json:"sessions"`
		Profile  string `json:"profile"`
	}{model, context, sessions, profile})
	if err != nil {
		return "", false
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), true
}

func tightConsentActive(saved Settings, st *Status, registries ...*Switches) bool {
	if !saved.AllowTightSetup || saved.TightSetupFingerprint == "" {
		return false
	}
	key, ok := tightSetupFingerprint(saved, st, registries...)
	return ok && key == saved.TightSetupFingerprint
}

// A stopped CLI reports no effective_settings. Its registry still declares
// defaults; profile-specific values take precedence without inventing a Mode
// name or a context/session count in the supervisor.
func tightRegistryDefault(registry *Switches, name, profile string) (string, bool) {
	if registry == nil {
		return "", false
	}
	for _, row := range registry.Switches {
		if row.Name != name {
			continue
		}
		value := row.DefaultText
		if declared := row.Values["default"]; declared != nil {
			value = *declared
		}
		if declared := row.Values[profile]; profile != "" && declared != nil {
			value = *declared
		}
		if name == "--profile" {
			var allowed struct {
				Type   string   `json:"type"`
				Values []string `json:"values"`
			}
			if json.Unmarshal(row.Allowed, &allowed) != nil || allowed.Type != "enum" {
				return "", false
			}
			for _, choice := range allowed.Values {
				if choice == value {
					return value, true
				}
			}
			return "", false
		}
		_, err := strconv.ParseUint(value, 10, 64)
		return value, err == nil
	}
	return "", false
}

func tightDefaultCount(registries []*Switches, st *Status, name, key, profile string) (uint64, bool) {
	if row, ok := SettingRow(st.Effective, key); ok {
		var value Count
		if json.Unmarshal(row.Effective, &value) != nil {
			return 0, false
		}
		return uint64(value), true
	}
	if len(registries) > 0 {
		if text, ok := tightRegistryDefault(registries[0], name, profile); ok {
			value, err := strconv.ParseUint(text, 10, 64)
			return value, err == nil
		}
	}
	return 0, false
}

func registryHasSwitch(registry *Switches, name string) bool {
	if registry == nil {
		return false
	}
	for _, row := range registry.Switches {
		if row.Name == name {
			return true
		}
	}
	return false
}
