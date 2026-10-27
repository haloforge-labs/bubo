package supervisor

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func fixtureMap(t *testing.T, rel string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(readFixtureFile(t, rel), &m); err != nil {
		t.Fatal(err)
	}
	return m
}
func mustParse(t *testing.T, m map[string]any) (Event, error) {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return ParseEvent(b)
}
func payloadOf(m map[string]any) map[string]any { return m["payload"].(map[string]any) }

func checkModelRowBytes(t *testing.T, row json.Marshaler, raw json.RawMessage) {
	t.Helper()
	got, err := row.MarshalJSON()
	if err != nil || !bytes.Equal(got, raw) {
		t.Fatal("raw CLI row changed on the way through", err)
	}
	// json.Marshal HTML-escapes even a Marshaler's raw output (for example the
	// CLI recommendation's >= becomes \u003e=). Test exact bytes at the raw
	// boundary above, and exact JSON values at the standard encoding boundary.
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	decode := func(data []byte) any {
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber() // Keep integer and decimal spellings exact.
		var value any
		if err := decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	if !reflect.DeepEqual(decode(encoded), decode(raw)) {
		t.Fatal("standard JSON encoding changed CLI row values")
	}
}

func TestModelsFixtureIsTypedAndKeepsNormalizedRowsByteForByte(t *testing.T) {
	for _, name := range []string{"valid/models.json", "valid/models-shipped-catalogue.json"} {
		e := parseFixture(t, name)
		m, ok := e.Value.(*Models)
		if !ok || len(m.Catalogue) == 0 {
			t.Fatalf("%s: not typed: %#v", name, e.Value)
		}
		var raw struct {
			Catalogue []json.RawMessage `json:"catalogue"`
			Scanned   []json.RawMessage `json:"scanned"`
		}
		// Numeric wire spellings are normalized at ParseEvent. From that
		// boundary onward the rows must still survive byte for byte.
		if err := json.Unmarshal(e.Payload, &raw); err != nil {
			t.Fatal(err)
		}
		for i, row := range m.Catalogue {
			checkModelRowBytes(t, row, raw.Catalogue[i])
		}
		for i, row := range m.Scanned {
			checkModelRowBytes(t, row, raw.Scanned[i])
		}
	}
	m := parseFixture(t, "valid/models.json").Value.(*Models)
	if m.Listing.ScanDepthLimit != 3 || m.Listing.AvailableBytes == nil || *m.Listing.AvailableBytes != 68719476736 {
		t.Fatalf("listing: %+v", m.Listing)
	}
	if len(m.Scanned) != 3 || m.Scanned[1].Runnable || m.Scanned[1].Reason == nil || len(m.ScanWarnings) != 1 {
		t.Fatalf("scanned rows: %+v", m.Scanned)
	}
	if c, ok := m.Entry("tiny-dense"); !ok || c.PreparedState != "prepared" || !c.DownloadEnabled || c.Estimate.State != "estimated" {
		t.Fatalf("entry: %+v", c)
	}
	if c, _ := m.Entry("absent"); c.DownloadEnabled || c.DisabledReason == nil || c.Estimate.State != "none_yet" {
		t.Fatalf("disabled entry: %+v", c)
	}
}

func TestModelEstimateFixturesAreTyped(t *testing.T) {
	for name, want := range map[string]string{"valid/model-estimate.json": "estimated", "valid/model-estimate-none.json": "none_yet", "valid/model-estimate-flash-next.json": "estimated"} {
		e := parseFixture(t, name)
		est, ok := e.Value.(*ModelEstimate)
		if !ok || est.State != want || est.Subject == "" || est.FitVerified {
			t.Fatalf("%s: %#v", name, e.Value)
		}
	}
	none := parseFixture(t, "valid/model-estimate-none.json").Value.(*ModelEstimate)
	if none.Reason == nil || none.hasMemoryEstimate() {
		t.Fatalf("none_yet carries numbers or no reason: %+v", none)
	}
}

func TestModelsRejectsBrokenRulesAsProtocolErrors(t *testing.T) {
	cases := map[string]func(p map[string]any){
		"estimated without numbers": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["estimate"].(map[string]any)["memory_estimate"] = nil
		},
		"estimated with a reason": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["estimate"].(map[string]any)["reason"] = "x"
		},
		"none_yet without a reason": func(p map[string]any) {
			p["catalogue"].([]any)[1].(map[string]any)["estimate"].(map[string]any)["reason"] = nil
		},
		"fit_verified true": func(p map[string]any) {
			p["scanned"].([]any)[0].(map[string]any)["estimate"].(map[string]any)["fit_verified"] = true
		},
		"duplicate id": func(p map[string]any) {
			c := p["catalogue"].([]any)
			c[1].(map[string]any)["id"] = c[0].(map[string]any)["id"]
		},
		"download_bytes is not the sum": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["download_bytes"] = 1
		},
		"disabled without a reason": func(p map[string]any) {
			p["catalogue"].([]any)[1].(map[string]any)["disabled_reason"] = nil
		},
		"enabled without a licence line": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["licence"].(map[string]any)["prompt_line"] = ""
		},
		"enabled file without a pin": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["files"].([]any)[0].(map[string]any)["sha256"] = ""
		},
		"more shards found than expected": func(p map[string]any) {
			p["scanned"].([]any)[0].(map[string]any)["shards_found"] = 4
		},
		"file that cannot run has no reason": func(p map[string]any) {
			p["scanned"].([]any)[1].(map[string]any)["reason"] = nil
		},
		"sidecar marked runnable": func(p map[string]any) {
			p["scanned"].([]any)[2].(map[string]any)["runnable"] = true
		},
		"missing catalogue": func(p map[string]any) { delete(p, "catalogue") },
		"unknown prepared_state": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["prepared_state"] = "ready"
		},
		"licence url not https": func(p map[string]any) {
			p["catalogue"].([]any)[0].(map[string]any)["licence"].(map[string]any)["url"] = "http://x"
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := fixtureMap(t, "valid/models.json")
			mutate(payloadOf(m))
			_, err := mustParse(t, m)
			var pe *ProtocolError
			if !errors.As(err, &pe) {
				t.Fatalf("accepted or wrong error: %v", err)
			}
		})
	}
	// The same mutation helper must not reject the untouched fixture.
	if _, err := mustParse(t, fixtureMap(t, "valid/models.json")); err != nil {
		t.Fatal(err)
	}
}

func TestModelEstimateRejectsBrokenRules(t *testing.T) {
	for name, mutate := range map[string]func(p map[string]any){
		"no subject":         func(p map[string]any) { p["subject"] = "" },
		"bad subject kind":   func(p map[string]any) { p["subject_kind"] = "url" },
		"none_yet no reason": func(p map[string]any) { p["reason"] = nil },
	} {
		t.Run(name, func(t *testing.T) {
			m := fixtureMap(t, "valid/model-estimate-none.json")
			mutate(payloadOf(m))
			if _, err := mustParse(t, m); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}

func planFixtureParts(t *testing.T, names ...string) []Plan {
	t.Helper()
	var parts []Plan
	for _, n := range names {
		parts = append(parts, *parseFixture(t, "valid/"+n+".json").Value.(*Plan))
	}
	return parts
}

var allPlanParts = []string{"plan", "plan-source-part", "plan-license-part", "plan-path-part", "plan-volume-part"}

func TestAssemblePlanJoinsPartsLicenceAndSpace(t *testing.T) {
	plan, err := assemblePlan(planFixtureParts(t, allPlanParts...), "model")
	if err != nil {
		t.Fatal(err)
	}
	if plan.Digest != strings.Repeat("c", 64) || plan.Ready || len(plan.Files) != 1 || plan.Files[0].FileID != "main" || plan.Files[0].Size != 4 {
		t.Fatalf("plan: %+v", plan)
	}
	if len(plan.Licences) != 1 || plan.Licences[0].Path != "LICENSE" || plan.Licences[0].Text != "Fixture license." {
		t.Fatalf("licences: %+v", plan.Licences)
	}
	if plan.Paths["model"] != "C:/test" {
		t.Fatalf("paths: %v", plan.Paths)
	}
	if !plan.EnoughSpace || plan.Drive != "C:" || plan.FreeBytes != 100 || plan.RequiredBytes != 50 {
		t.Fatalf("space: %+v", plan)
	}
	// A short drive is reported, not hidden.
	parts := planFixtureParts(t, allPlanParts...)
	parts[4].Volumes[0].FreeBytes = 10
	short, err := assemblePlan(parts, "model")
	if err != nil || short.EnoughSpace || short.FreeBytes != 10 {
		t.Fatalf("short drive: %+v %v", short, err)
	}
}

func TestAssemblePlanRejectsBrokenSets(t *testing.T) {
	all := func() []Plan { return planFixtureParts(t, allPlanParts...) }
	missing := all()[:4]
	if _, err := assemblePlan(missing, "model"); err == nil {
		t.Fatal("a plan with a missing part was accepted")
	}
	dup := append(all(), all()[1])
	if _, err := assemblePlan(dup, "model"); err == nil {
		t.Fatal("a repeated part was accepted")
	}
	split := all()
	split[2].Digest = strings.Repeat("e", 64)
	if _, err := assemblePlan(split, "model"); err == nil {
		t.Fatal("parts with different digests were accepted")
	}
	gap := all()
	gap[2].Licenses[0].TextOffset = 2
	gap[2].Licenses[0].TextBytes = 18
	if _, err := assemblePlan(gap, "model"); err == nil {
		t.Fatal("licence text with a gap was accepted")
	}
	short := all()
	short[2].Licenses[0].TextBytes = 40
	if _, err := assemblePlan(short, "model"); err == nil {
		t.Fatal("incomplete licence text was accepted")
	}
	if _, err := assemblePlan(all(), "runtime"); err == nil {
		t.Fatal("a plan for another component was accepted")
	}
}

func TestAssemblePlanJoinsLicenceChunks(t *testing.T) {
	parts := planFixtureParts(t, allPlanParts...)
	first := parts[2]
	second := first
	first.Licenses = []License{{Path: "LICENSE", Text: "Fixture ", SHA256: parts[2].Licenses[0].SHA256, TextOffset: 0, TextBytes: 16}}
	second.Licenses = []License{{Path: "LICENSE", Text: "license.", SHA256: parts[2].Licenses[0].SHA256, TextOffset: 8, TextBytes: 16}}
	second.Part = 5
	for i := range parts {
		parts[i].Parts = 6
	}
	first.Parts, second.Parts = 6, 6
	parts[2] = first
	parts = append(parts, second)
	plan, err := assemblePlan(parts, "model")
	if err != nil || plan.Licences[0].Text != "Fixture license." {
		t.Fatalf("%+v %v", plan.Licences, err)
	}
}

func TestNormalizeModelFolderPath(t *testing.T) {
	good := map[string]string{
		`D:\Models`:         `D:\Models`,
		`D:\Models\`:        `D:\Models`,
		`d:/Models/sub/..`:  `d:\Models`,
		`\\server\share\m`:  `\\server\share\m`,
		`C:\Users\me\x y\z`: `C:\Users\me\x y\z`,
	}
	for in, want := range good {
		got, err := NormalizeModelFolderPath(in)
		if err != nil || got != want {
			t.Errorf("%q: %q %v (want %q)", in, got, err, want)
		}
	}
	for _, in := range []string{``, `   `, `C:\`, `C:`, `D:/`, `\\server\share`, `\\server\share\`, `models`, `.\models`, `-rf`, "D:\\a\x00b", "D:\\a\nb"} {
		if got, err := NormalizeModelFolderPath(in); err == nil {
			t.Errorf("%q accepted as %q", in, got)
		}
	}
}

func TestCatalogueIDAndSameFolder(t *testing.T) {
	for _, id := range []string{"qwen36-27b-ud-q4-k-xl", "a", "x.y_z-1"} {
		if !ValidCatalogueID(id) {
			t.Errorf("%q refused", id)
		}
	}
	for _, id := range []string{"", "-a", "a b", `a\b`, "a/b", "..", "a;b", strings.Repeat("a", 200)} {
		if ValidCatalogueID(id) {
			t.Errorf("%q accepted", id)
		}
	}
	if !SameFolder(`D:\Models\`, `d:\models`) || SameFolder(`D:\Models`, `D:\Other`) {
		t.Fatal("SameFolder")
	}
}
