//go:build windows

package supervisor

import "testing"

func TestRegistryPreferenceMetadataUsesTypedAppFields(t *testing.T) {
	in := actionTestInput(Stopped)
	in.Registry = &Switches{Switches: []Switch{{Name: "--host", Kind: "flag", GUIControl: true}}}
	in.Saved.LanAccess = false
	rows := switchOverrideViews(in)
	if len(rows) != 1 || rows[0].WriteVia != WriteViaPreference || rows[0].PreferenceField != "lan_access" || rows[0].PreferenceType != "boolean" {
		t.Fatal("argv host spelling replaced the preference's boolean shape", rows)
	}
	view := new(ViewBuilder).Build(in)
	if view.Saved.LANAccess {
		t.Fatal("explicit saved false was omitted", view.Saved)
	}
}
