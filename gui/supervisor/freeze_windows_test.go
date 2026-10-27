//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckForUpdatesPreservesAbsentOnAndExplicitChoices(t *testing.T) {
	for _, choice := range []string{"missing", "legacy", "true", "false"} {
		t.Run(choice, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "settings.json")
			if choice != "missing" {
				body := `{"version":1}`
				if choice != "legacy" {
					body = `{"version":1,"check_for_updates":` + choice + `}`
				}
				if err := os.WriteFile(path, []byte(body), 0600); err != nil {
					t.Fatal(err)
				}
			}
			s, warning := LoadSettings(path)
			if warning != "" {
				t.Fatal(warning)
			}
			var b ViewBuilder
			v := b.Build(ViewInput{Saved: s})
			if v.CheckForUpdates != (choice != "false") {
				t.Fatal("wrong effective update preference")
			}
			if (choice == "missing" || choice == "legacy") && v.Saved.CheckForUpdates != nil {
				t.Fatal("invented a user override")
			}
			if err := SaveSettings(path, s); err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var stored map[string]json.RawMessage
			if err = json.Unmarshal(raw, &stored); err != nil {
				t.Fatal(err)
			}
			_, exists := stored["check_for_updates"]
			if exists != (choice == "true" || choice == "false") {
				t.Fatal("persisted an absent update preference")
			}
			loaded, warning := LoadSettings(path)
			if warning != "" || loaded.UpdatesEnabled() != v.CheckForUpdates {
				t.Fatal("update choice lost on round trip")
			}
		})
	}
}

func TestFrozenViewPreservesAndOwnsNewCLIMemoryAndUpdateFields(t *testing.T) {
	fits, enabled := false, false
	total := uint64(20)
	label := "CLI only label"
	m := &MemoryEstimate{GPU: &MemoryGPUBudget{Available: &MemoryAmount{Bytes: 12}}, Host: &MemoryHostBudget{Available: &MemoryAmount{Bytes: 4}}, Fits: &fits, FitScope: "planning", CacheMode: "explicit", RecommendedContext: &RecommendedContext{BySessions: []RecommendedContextRow{{Sessions: 1, ContextPerSession: 2}}}}
	st := &Status{RuntimeDetail: RuntimeDetail{MemoryEstimate: m, Endpoints: &Endpoints{OpenAIBaseURL: "CLI OpenAI URL", AnthropicBaseURL: "CLI Anthropic URL"}, APIKeyEnv: "CLI_KEY_ENV", EffectiveSwitches: []EffectiveSwitch{{Name: "switch", Label: label}}}}
	in := ViewInput{Snapshot: Snapshot{State: Ready, Status: st}, Saved: Settings{CheckForUpdates: &enabled}, Update: Update{State: UpdateInProgress, CurrentVersion: "from updater", Progress: &UpdateProgress{Phase: "download", BytesDone: 3, BytesTotal: &total}, SizeBytes: &total, LastResultNotice: "last result"}}
	var builder ViewBuilder
	v := builder.Build(in)
	if v.MemoryEstimate.Fits == nil || *v.MemoryEstimate.Fits || v.EffectiveSwitches[0].Label != label || v.APIKeyEnv != st.APIKeyEnv || v.Endpoints.OpenAIBaseURL != st.Endpoints.OpenAIBaseURL || v.Update.LastResultNotice != "last result" {
		t.Fatal("new CLI or update fields lost")
	}
	*v.MemoryEstimate.Fits = true
	v.MemoryEstimate.GPU.Available.Bytes = 1
	v.MemoryEstimate.RecommendedContext.BySessions[0].ContextPerSession = 7
	v.Endpoints.OpenAIBaseURL = "changed"
	*v.Saved.CheckForUpdates = true
	*v.Update.Progress.BytesTotal = 1
	if fits || m.GPU.Available.Bytes != 12 || m.RecommendedContext.BySessions[0].ContextPerSession != 2 || st.Endpoints.OpenAIBaseURL != "CLI OpenAI URL" || enabled || total != 20 {
		t.Fatal("frozen View shares mutable inputs")
	}
	unknown := builder.Build(ViewInput{Snapshot: Snapshot{Status: &Status{RuntimeDetail: RuntimeDetail{MemoryEstimate: &MemoryEstimate{}}}}})
	if unknown.Endpoints != nil || unknown.MemoryEstimate.GPU != nil || unknown.MemoryEstimate.Host != nil || unknown.MemoryEstimate.Fits != nil {
		t.Fatal("invented absent CLI budget facts")
	}
}
