//go:build windows

package supervisor

import "encoding/json"

type PendingChange struct {
	Key     string `json:"key"`
	Label   string `json:"label"`
	Running string `json:"running"`
	Next    string `json:"next"`
}

func requestedHost(saved Settings) string {
	if saved.LanAccess {
		return "0.0.0.0"
	}
	return "127.0.0.1"
}

// Read CLI rows by registry name. The older effective_settings preference key
// remains an accepted wire input; no default or label is supplied here.
func observedPreference(st *Status, arg serveArgument, instances ...*Instance) json.RawMessage {
	if st == nil {
		return nil
	}
	if raw := effectiveSwitchValue(st.EffectiveSwitches, arg.name); len(raw) > 0 {
		return raw
	}
	for _, key := range []string{arg.name, arg.preference} {
		if row, ok := SettingRow(st.Effective, key); ok {
			return row.Effective
		}
		if raw := st.Effective[key]; len(raw) > 0 && valueFromJSON(raw).Kind != "object" {
			return raw
		}
	}
	if st.ActiveModel != nil {
		var value *string
		switch arg.preference {
		case "model_path":
			value = st.ActiveModel.ModelPath
		case "manifest_path":
			value = st.ActiveModel.ManifestPath
		}
		if value != nil {
			b, _ := json.Marshal(*value)
			return b
		}
	}
	var instance *Instance
	if len(instances) > 0 {
		instance = instances[0]
	}
	if endpoint := boundEndpoint(st, instance); endpoint != nil {
		switch arg.preference {
		case "port":
			return json.RawMessage(endpoint.Port())
		case "lan_access":
			b, _ := json.Marshal(endpoint.Hostname())
			return b
		}
	}
	return nil
}

// RestartNeeded retains its public contract. Callers provide the CLI registry
// for labels; absent registry data stays unavailable instead of inventing text.
func RestartNeeded(saved Settings, running *Instance, effective map[string]json.RawMessage, registries ...*Switches) []PendingChange {
	var registry *Switches
	if len(registries) > 0 {
		registry = registries[0]
	}
	return pendingChanges(ViewInput{Saved: saved, Registry: registry, Snapshot: Snapshot{Instance: running, Status: &Status{Effective: effective}}})
}

func pendingChanges(in ViewInput) []PendingChange {
	if in.Snapshot.Instance == nil || in.Registry == nil {
		return nil
	}
	var out []PendingChange
	for _, v := range switchOverrideViews(in) {
		if v.RestartState != SwitchPending || v.Saved == nil {
			continue
		}
		label := ""
		for _, row := range in.Registry.Switches {
			if row.Name == v.Name {
				label = row.Label
				break
			}
		}
		key := v.Name
		if v.WriteVia == "preference" {
			key = v.PreferenceField
		}
		now := v.Effective.Text
		if v.Effective.Kind == "boolean" && v.Effective.Boolean != nil {
			if *v.Effective.Boolean {
				now = "true"
			} else {
				now = "false"
			}
		}
		out = append(out, PendingChange{Key: key, Label: label, Running: now, Next: *v.Saved})
	}
	return out
}
