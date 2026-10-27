//go:build windows

package supervisor

import (
	"encoding/json"
	"fmt"
	"math/big"
	"reflect"
	"sort"
	"strings"
)

type SwitchOverride struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type SwitchOverridesPatch struct {
	Set   []SwitchOverride `json:"set"`
	Clear []string         `json:"clear"`
}

// This CLI setting names a file. "none" is a UI placeholder, not a file to
// open; clearing the override selects the CLI's built-in marker list.
func markerListUnset(name, value string) bool {
	return name == "GUFO_THINK_MARKER_LIST" && strings.EqualFold(strings.TrimSpace(value), "none")
}

type SwitchWriteBlock string

const (
	SwitchWritable        SwitchWriteBlock = ""
	SwitchSupervisorOwned SwitchWriteBlock = "supervisor_owned"
	SwitchSecret          SwitchWriteBlock = "secret"
	SwitchReadOnly        SwitchWriteBlock = "read_only"
	SwitchUnknownKind     SwitchWriteBlock = "unsupported_kind"
	SwitchRemoved         SwitchWriteBlock = "removed"
)

func switchWriteBlock(row Switch) SwitchWriteBlock {
	if row.Class == "REMOVED" {
		return SwitchRemoved
	}
	for _, owned := range serveArguments(Settings{}, "", 0) {
		if row.Name == owned.name || row.Kind == "env" && strings.EqualFold(row.Name, owned.name) {
			return SwitchSupervisorOwned
		}
	}
	// echo controls CLI reporting, not write eligibility. Older registries omit
	// secret; explicit supervisor ownership still protects credential handoff.
	if row.Secret {
		return SwitchSecret
	}
	if !row.GUIControl {
		return SwitchReadOnly
	}
	if row.Kind != "flag" && row.Kind != "env" {
		return SwitchUnknownKind
	}
	return SwitchWritable
}

func overrideRow(registry *Switches, name string) (Switch, error) {
	if registry != nil {
		for _, row := range registry.Switches {
			if row.Name == name {
				if block := switchWriteBlock(row); block != SwitchWritable {
					return row, appLabelError("setting_read_only", fmt.Sprintf("Switch %s is not writable (%s).", name, block), row.Label)
				}
				return row, nil
			}
		}
	}
	return Switch{}, appError("setting_removed", fmt.Sprintf("Switch %s is absent from the CLI registry.", name))
}

func ApplySwitchOverrides(current map[string]string, patch SwitchOverridesPatch, registry *Switches) (map[string]string, error) {
	next := make(map[string]string, len(current)+len(patch.Set))
	for name, value := range current {
		next[name] = value
	}
	changed := map[string]bool{}
	for _, name := range patch.Clear {
		delete(next, name)
		changed[name] = true
	}
	for _, set := range patch.Set {
		if changed[set.Name] {
			return nil, appError("internal_error", "A switch may be changed only once per patch: "+set.Name)
		}
		if _, err := overrideRow(registry, set.Name); err != nil {
			return nil, err
		}
		changed[set.Name] = true
		if markerListUnset(set.Name, set.Value) {
			delete(next, set.Name)
		} else {
			next[set.Name] = set.Value
		}
	}
	return next, nil
}

// A changed CLI registry can invalidate saved requests. Only a registry that
// actually arrived can justify dropping one; without it, retain the settings.
func dropStaleOverrides(overrides map[string]string, registry *Switches) (map[string]string, []string) {
	if registry == nil || len(overrides) == 0 {
		return overrides, nil
	}
	next := make(map[string]string, len(overrides))
	var removed []string
	for name, value := range overrides {
		if markerListUnset(name, value) {
			continue
		}
		if _, err := overrideRow(registry, name); err != nil {
			removed = append(removed, name)
		} else {
			next[name] = value
		}
	}
	sort.Strings(removed)
	return next, removed
}

// switchCommand encodes registry rows without interpreting allowed, profiles or
// inject_with. The latter is a CLI profile trigger, not a GUI argument rule.
// A no-value flag uses an empty string or "true" for presence. Other spellings
// use the CLI's equals syntax so an invalid value cannot silently enable it;
// the CLI accepts or rejects that spelling. Clear removes the request.
func switchCommand(overrides map[string]string, registry *Switches) ([]string, map[string]string, error) {
	var names, args []string
	for name := range overrides {
		names = append(names, name)
	}
	sort.Strings(names)
	env := map[string]string{}
	for _, name := range names {
		if markerListUnset(name, overrides[name]) {
			continue
		}
		row, err := overrideRow(registry, name)
		if err != nil {
			return nil, nil, err
		}
		if row.Kind == "env" {
			env[name] = overrides[name]
			continue
		}
		if row.TakesValue {
			args = append(args, name, overrides[name])
		} else if overrides[name] == "" || overrides[name] == "true" {
			args = append(args, name)
		} else {
			args = append(args, name+"="+overrides[name])
		}
	}
	return args, env, nil
}

// Environment replacement follows Windows' case-insensitive variable names and
// removes inherited duplicates. Only the serve child receives these overrides.
func switchEnvironment(parent []string, overrides map[string]string) []string {
	out := make([]string, 0, len(parent)+len(overrides))
	for _, entry := range parent {
		name, _, _ := strings.Cut(entry, "=")
		replaced := false
		for key := range overrides {
			if strings.EqualFold(name, key) {
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, entry)
		}
	}
	var names []string
	for name := range overrides {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		out = append(out, name+"="+overrides[name])
	}
	return out
}

type SwitchRestartState string

type SwitchWriteVia string

const (
	WriteViaPreference SwitchWriteVia = "preference"
	WriteViaOverride   SwitchWriteVia = "override"
)

const (
	SwitchNotRunning          SwitchRestartState = "not_running"
	SwitchUnchanged           SwitchRestartState = "unchanged"
	SwitchPending             SwitchRestartState = "pending"
	SwitchUnknownUntilRestart SwitchRestartState = "unknown_until_restart"
	SwitchRestartNotRequired  SwitchRestartState = "not_required"
)

// SwitchOverrideView joins the saved request with the CLI's effective value.
// Unknown is explicit; Go never derives a default from profile metadata.
type SwitchOverrideView struct {
	WriteVia        SwitchWriteVia     `json:"write_via,omitempty"`
	PreferenceField string             `json:"preference_field,omitempty"`
	PreferenceType  string             `json:"preference_type,omitempty"`
	Name            string             `json:"name"`
	Saved           *string            `json:"saved"`
	Effective       Value              `json:"effective"`
	WriteBlock      SwitchWriteBlock   `json:"write_block"`
	RestartState    SwitchRestartState `json:"restart_state"`
}

func switchOverrideViews(in ViewInput) []SwitchOverrideView {
	if in.Registry == nil {
		return nil
	}
	effective := map[string]json.RawMessage{}
	if in.Snapshot.Status != nil {
		for _, row := range in.Snapshot.Status.EffectiveSwitches {
			effective[row.Name] = row.Effective
		}
	}
	var out []SwitchOverrideView
	for _, row := range in.Registry.Switches {
		v := SwitchOverrideView{Name: row.Name, WriteVia: WriteViaOverride, WriteBlock: switchWriteBlock(row), RestartState: SwitchUnchanged, Effective: Value{Kind: "null"}}
		var owner *serveArgument
		for _, arg := range serveArguments(in.Saved, requestedHost(in.Saved), in.Saved.Port) {
			if arg.name == row.Name && arg.preference != "" {
				copy := arg
				owner = &copy
				v.WriteVia = WriteViaPreference
				v.PreferenceField = arg.preference
				v.PreferenceType = preferenceType(arg.preference)
				break
			}
		}
		if hideSwitchValue(row) {
			v.Effective = valueFromJSON(hiddenValue(effective[row.Name]))
			out = append(out, v)
			continue
		}
		if row.Class == "REMOVED" || v.WriteBlock != SwitchWritable && owner == nil {
			out = append(out, v)
			continue
		}
		raw := effective[row.Name]
		if len(raw) == 0 && owner != nil && in.Snapshot.Status != nil {
			raw = observedPreference(in.Snapshot.Status, *owner, in.Snapshot.Instance)
		}
		v.Effective = valueFromJSON(raw)
		if owner != nil {
			if owner.value != "" {
				saved := owner.value
				v.Saved = &saved
			}
		} else if saved, ok := in.Saved.SwitchOverrides[row.Name]; ok {
			v.Saved = &saved
		}
		_, wasSet := in.Snapshot.RunningSwitchOverrides[row.Name]
		if v.Saved != nil || wasSet {
			switch {
			case in.Snapshot.Instance == nil:
				v.RestartState = SwitchNotRunning
			case !row.RestartRequired:
				v.RestartState = SwitchRestartNotRequired
			case v.Saved == nil:
				v.RestartState = SwitchUnknownUntilRestart // cleared: CLI chooses its next default
			case len(raw) == 0 || string(raw) == "null":
				v.RestartState = SwitchUnknownUntilRestart
			default:
				now, known := switchScalar(raw)
				next := *v.Saved
				if row.Kind == "flag" && !row.TakesValue && next == "" {
					next = "true"
				}
				if !known {
					v.RestartState = SwitchUnknownUntilRestart
				} else if !equalSwitchValue(raw, now, next) {
					v.RestartState = SwitchPending
				}
			}
		}
		out = append(out, v)
	}
	return out
}

// Field shape belongs to the app preference contract, not to CLI argument
// spelling (for example a boolean preference may render an argv host string).
func preferenceType(name string) string {
	t := reflect.TypeOf(PreferenceValues{})
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if strings.Split(field.Tag.Get("json"), ",")[0] != name {
			continue
		}
		kind := field.Type.Elem().Kind()
		switch kind {
		case reflect.Bool:
			return "boolean"
		case reflect.String:
			return "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return "integer"
		default:
			return "number"
		}
	}
	return ""
}

func effectiveSwitchValue(rows []EffectiveSwitch, name string) json.RawMessage {
	for _, row := range rows {
		if row.Name == name {
			return row.Effective
		}
	}
	return nil
}

func equalSwitchValue(raw json.RawMessage, running, saved string) bool {
	if valueFromJSON(raw).Kind == "number" {
		// Compare numeric spellings without validating a value or imposing CLI
		// limits. 1 and 1.0 must not produce a spurious restart warning.
		a, aok := new(big.Rat).SetString(running)
		b, bok := new(big.Rat).SetString(saved)
		return aok && bok && a.Cmp(b) == 0
	}
	return running == saved
}

func switchScalar(raw json.RawMessage) (string, bool) {
	v := valueFromJSON(raw)
	switch v.Kind {
	case "number", "string":
		return v.Text, true
	case "boolean":
		if v.Boolean != nil {
			if *v.Boolean {
				return "true", true
			}
			return "false", true
		}
	}
	return "", false
}
