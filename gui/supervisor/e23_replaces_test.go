package supervisor

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

// E23: a repair reason in the CLI's parallel plan list must survive parsing,
// part assembly and the bound plan without changing its approved digest.
func TestE23ReplacementReachesConsentPlan(t *testing.T) {
	want := []string{"SHA-256 differs from the pin"}
	var parts []Plan
	for _, name := range allPlanParts {
		record := fixtureMap(t, "valid/"+name+".json")
		payload := payloadOf(record)
		payload["file_states"] = []string{"download"}
		payload["replaces"] = want
		event, err := mustParse(t, record)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		part := *event.Value.(*Plan)
		encoded, err := json.Marshal(part)
		if err != nil {
			t.Fatal(err)
		}
		var roundtrip struct {
			Replaces []string `json:"replaces"`
		}
		if err := json.Unmarshal(encoded, &roundtrip); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(roundtrip.Replaces, want) {
			t.Fatalf("%s lost the CLI's replacement list: %s", name, encoded)
		}
		parts = append(parts, part)
	}
	plan, err := assemblePlan(parts, "model")
	if err != nil {
		t.Fatal(err)
	}
	if plan.Digest != parts[0].Digest || len(plan.Files) != 1 || plan.Files[0].State != "download" {
		t.Fatalf("plan identity, source or state changed: %+v", plan)
	}
	encoded, err := json.Marshal(Filled(plan))
	if err != nil {
		t.Fatal(err)
	}
	var consent struct {
		Files []struct {
			Replaces string `json:"replaces"`
		} `json:"files"`
	}
	if err := json.Unmarshal(encoded, &consent); err != nil {
		t.Fatal(err)
	}
	if len(consent.Files) != 1 || consent.Files[0].Replaces != want[0] {
		t.Fatalf("consent lost the replacement reason: %s", encoded)
	}
	progress := jobProgress(partJobFiles(parts), "model", "main", "download", 0)
	if progress == nil || progress.BytesTotal != plan.Files[0].Size {
		t.Fatalf("replacement not counted as a download: %+v", progress)
	}
}

func TestE23ReplacementRejectsInvalidCLIShape(t *testing.T) {
	for name, invalid := range map[string]any{
		"not an array": "SHA-256 differs from the pin",
		"not strings":  []any{123},
	} {
		t.Run(name, func(t *testing.T) {
			record := fixtureMap(t, "valid/plan.json")
			payloadOf(record)["replaces"] = invalid
			_, err := mustParse(t, record)
			var protocolErr *ProtocolError
			if !errors.As(err, &protocolErr) {
				t.Fatalf("malformed replacement accepted: %v", err)
			}
		})
	}
}

func TestE23ReplacementRejectsUnpairedOrDisagreeingParts(t *testing.T) {
	for _, tc := range []struct {
		name   string
		header []string
		source []string
	}{
		{"wrong length", []string{"damaged", "extra"}, []string{"damaged", "extra"}},
		{"different parts", []string{"damaged"}, []string{""}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts := planFixtureParts(t, allPlanParts...)
			for i := range parts {
				parts[i].Replaces = tc.header
			}
			parts[1].Replaces = tc.source
			_, err := assemblePlan(parts, "model")
			var protocolErr *ProtocolError
			if !errors.As(err, &protocolErr) {
				t.Fatalf("unpaired replacement accepted: %v", err)
			}
		})
	}
}
