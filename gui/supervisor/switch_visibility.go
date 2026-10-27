package supervisor

import "encoding/json"

// echo is an environment-row reporting policy. Flag values remain visible
// unless the CLI explicitly marks the row secret.
func hideSwitchValue(row Switch) bool { return row.Secret || row.Kind == "env" && !row.Echo }

func hiddenRegistryValue(registry *Switches, name string) bool {
	if registry == nil {
		return false
	}
	for _, row := range registry.Switches {
		if row.Name == name {
			return hideSwitchValue(row)
		}
	}
	return false
}

func hiddenValue(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return raw
	}
	return json.RawMessage(`"[hidden]"`)
}

// RedactSettings applies the CLI metadata to every setting projection.
func RedactSettings(in map[string]json.RawMessage, registry *Switches) map[string]json.RawMessage {
	if in == nil {
		return nil
	}
	out := make(map[string]json.RawMessage, len(in))
	for name, raw := range in {
		out[name] = raw
		if !hiddenRegistryValue(registry, name) {
			continue
		}
		var row map[string]json.RawMessage
		if json.Unmarshal(raw, &row) == nil && row != nil {
			for _, key := range []string{"requested", "effective"} {
				if value, ok := row[key]; ok {
					row[key] = hiddenValue(value)
				}
			}
			out[name], _ = json.Marshal(row)
		} else {
			out[name] = hiddenValue(raw)
		}
	}
	return out
}
