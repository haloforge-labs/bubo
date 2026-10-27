//go:build windows

package supervisor

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestB4PageFactsPreserveCLIValuesAndViewOwnership(t *testing.T) {
	for _, name := range []string{"page-facts-false.json", "page-facts-true.json"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "b4", name))
			if err != nil {
				t.Fatal(err)
			}
			e, err := ParseEvent(data)
			if err != nil {
				t.Fatal(err)
			}
			st := e.Value.(*Status)
			var builder ViewBuilder
			in := ViewInput{Snapshot: Snapshot{State: Ready, Status: st, Instance: &st.Instances[0], Discovered: st.Instances}}
			v := builder.Build(in)
			if st.ActiveModel == nil || len(st.ClientSnippets) < 3 || !reflect.DeepEqual(v.ActiveModel, st.ActiveModel) || !reflect.DeepEqual(v.ClientSnippets, st.ClientSnippets) || !reflect.DeepEqual(v.Instance.ActiveModel, st.Instances[0].ActiveModel) || !reflect.DeepEqual(v.Discovered[0].ActiveModel, st.Instances[0].ActiveModel) {
				t.Fatal("CLI page facts lost")
			}
			// Ready shares the same runtime detail contract with status.
			ready := viewFixture(t, "ready")
			life := ready.Value.(*Lifecycle)
			life.RuntimeDetail = st.RuntimeDetail
			ready.Payload, err = json.Marshal(life)
			if err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(ready)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := ParseEvent(raw)
			if err != nil || !reflect.DeepEqual(parsed.Value.(*Lifecycle).ActiveModel, st.ActiveModel) || !reflect.DeepEqual(parsed.Value.(*Lifecycle).ClientSnippets, st.ClientSnippets) {
				t.Fatal("ready lost page facts", err)
			}
			*v.ActiveModel.ModelPath = "changed"
			v.ClientSnippets[0].Template = "changed"
			if v.ClientSnippets[0].DefaultAPIKey != nil {
				*v.ClientSnippets[0].DefaultAPIKey = "changed"
			}
			*v.Instance.ActiveModel.ModelPath = "changed"
			*v.Discovered[0].ActiveModel.ModelPath = "changed"
			fresh := builder.Build(in)
			if *fresh.ActiveModel.ModelPath == "changed" || fresh.ClientSnippets[0].Template == "changed" || *fresh.Instance.ActiveModel.ModelPath == "changed" || *fresh.Discovered[0].ActiveModel.ModelPath == "changed" || fresh.ClientSnippets[0].DefaultAPIKey != nil && *fresh.ClientSnippets[0].DefaultAPIKey == "changed" {
				t.Fatal("View aliases CLI page facts")
			}
			if fresh.ClientSnippets[0].RequiresKey != (name == "page-facts-true.json") {
				t.Fatal("key requirement lost")
			}
			if name == "page-facts-true.json" && (fresh.ActiveModel.CatalogueID != nil || fresh.ActiveModel.ManifestPath != nil) {
				t.Fatal("invented imported model provenance")
			}
		})
	}
	var builder ViewBuilder
	old := builder.Build(ViewInput{Snapshot: Snapshot{Status: viewFixture(t, "status-observed").Value.(*Status)}})
	// Absent lists are published empty, never null (TestViewNeverPublishesNullLists);
	// "not invented" means no entries.
	if old.ActiveModel != nil || len(old.ClientSnippets) != 0 {
		t.Fatal("invented absent page facts")
	}
}

func TestB4PlanVolumesSurviveAssemblyWithoutRecalculation(t *testing.T) {
	f, err := os.Open(filepath.Join("..", "tests", "scenarios", "recordings", "b4", "prepare.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var parts []Plan
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 4096), 1<<20)
	for sc.Scan() {
		e, err := ParseEvent(sc.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		if p, ok := e.Value.(*Plan); ok {
			parts = append(parts, *p)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	var builder ViewBuilder
	v := builder.Build(ViewInput{Snapshot: Snapshot{State: Preparing, PlanParts: parts}})
	if len(v.Preparation) != 1 || len(v.Preparation[0].Volumes) == 0 || v.PlanError != nil {
		t.Fatal("recorded plan lost", v.PlanError)
	}
	p, err := assemblePlan(parts, "model")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(v.Preparation[0].Volumes, p.Volumes) {
		t.Fatal("volume facts changed")
	}
	volume := v.Preparation[0].Volumes[0]
	if volume.FreeAfterDownloadBytes == nil || volume.BytesToWrite == nil || volume.SafetyMarginBytes == nil {
		t.Fatal("new volume facts lost")
	}
	before := *p.Volumes[0].FreeAfterDownloadBytes
	*volume.FreeAfterDownloadBytes = 0
	if *p.Volumes[0].FreeAfterDownloadBytes != before {
		t.Fatal("View aliases plan volume pointers")
	}
	legacy := viewFixture(t, "plan-volume-part").Value.(*Plan)
	for _, v := range legacy.Volumes {
		if v.FreeAfterDownloadBytes != nil || v.BytesToWrite != nil || v.SafetyMarginBytes != nil {
			t.Fatal("invented old volume facts")
		}
	}
}

func TestEchoFalseFlagsAreWritableAndExplicitSecretsNeverAre(t *testing.T) {
	registry := viewFixture(t, "switches").Value.(*Switches)
	var exposed int
	for _, row := range registry.Switches {
		if row.Kind != "flag" || row.Echo || !row.GUIControl || switchWriteBlock(row) == SwitchSupervisorOwned {
			continue
		}
		if switchWriteBlock(row) != SwitchWritable {
			t.Fatal("echo blocked flag", row.Name)
		}
		if _, _, err := switchCommand(map[string]string{row.Name: "future-cli-value"}, registry); err != nil {
			t.Fatal(err)
		}
		exposed++
	}
	if exposed == 0 {
		t.Fatal("no production echo:false flags tested")
	}
	t.Logf("echo:false writable flags: %d", exposed)
	for _, kind := range []string{"flag", "env"} {
		t.Run(kind, func(t *testing.T) {
			var row Switch
			body := []byte(`{"name":"future-credential","kind":"` + kind + `","gui_control":true,"echo":false,"secret":true}`)
			if err := json.Unmarshal(body, &row); err != nil {
				t.Fatal(err)
			}
			reg := &Switches{Switches: []Switch{row}}
			if !row.Secret || switchWriteBlock(row) != SwitchSecret {
				t.Fatal("explicit secret writable")
			}
			if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: row.Name, Value: "sentinel"}}}, reg); err == nil {
				t.Fatal("secret patch accepted")
			}
			if _, _, err := switchCommand(map[string]string{row.Name: "sentinel"}, reg); err == nil {
				t.Fatal("persisted secret launched")
			}
			var builder ViewBuilder
			v := builder.Build(ViewInput{Registry: reg, Saved: Settings{SwitchOverrides: map[string]string{row.Name: "sentinel"}}})
			raw, _ := json.Marshal(v)
			if v.Switches[0].Secret != true || v.SwitchOverrides[0].WriteBlock != SwitchSecret || strings.Contains(string(raw), "sentinel") {
				t.Fatal("secret exposed in View")
			}
			cleared, err := ApplySwitchOverrides(map[string]string{row.Name: "sentinel"}, SwitchOverridesPatch{Clear: []string{row.Name}}, reg)
			if err != nil || len(cleared) != 0 {
				t.Fatal("secret cannot be cleared", err)
			}
		})
	}
}

func TestMergedMemoryFieldsRemainDistinctAndOwnedInView(t *testing.T) {
	fits, recommended := true, false
	reason := "CLI headroom reason"
	st := viewFixture(t, "settings-memory-v1").Value.(*Status)
	m := st.MemoryEstimate
	m.Fits, m.Recommended, m.RecommendationReason = &fits, &recommended, &reason
	m.PerSessionVision, m.VisionSessionTotal = &MemoryAmount{Bytes: 64}, &MemoryAmount{Bytes: 128}
	m.GPU, m.Host = &MemoryGPUBudget{Need: MemoryAmount{Bytes: 256}}, &MemoryHostBudget{Need: MemoryAmount{Bytes: 512}}
	m.RecommendedContext = &RecommendedContext{BySessions: []RecommendedContextRow{{Fits: true, Recommended: &recommended}}}
	var builder ViewBuilder
	in := ViewInput{Snapshot: Snapshot{Status: st}}
	v := builder.Build(in).MemoryEstimate
	if v == nil || v.Fits == nil || !*v.Fits || v.Recommended == nil || *v.Recommended || v.RecommendationReason == nil || *v.RecommendationReason != reason || v.PerSessionVision == nil || v.PerSessionVision.Bytes != 64 || v.VisionSessionTotal == nil || v.VisionSessionTotal.Bytes != 128 || v.GPU == nil || v.GPU.Need.Bytes != 256 || v.Host == nil || v.Host.Need.Bytes != 512 || v.RecommendedContext.BySessions[0].Recommended == nil || *v.RecommendedContext.BySessions[0].Recommended {
		t.Fatal("merged memory fields lost or verdicts conflated")
	}
	*v.Recommended = true
	*v.RecommendationReason = "changed"
	*v.RecommendedContext.BySessions[0].Recommended = true
	v.PerSessionVision.Bytes = 0
	fresh := builder.Build(in).MemoryEstimate
	if *fresh.Recommended || *fresh.RecommendedContext.BySessions[0].Recommended || *fresh.RecommendationReason == "changed" || fresh.PerSessionVision.Bytes != 64 {
		t.Fatal("memory View aliases CLI fields")
	}
	m.Fits, m.Recommended = nil, nil
	unknown := builder.Build(in).MemoryEstimate
	if unknown.Fits != nil || unknown.Recommended != nil {
		t.Fatal("unknown verdict invented")
	}
}
