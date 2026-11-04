//go:build windows

package supervisor

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

// A saved --refusal-direction on, --prefill-mode fast or --kv-cache-type q8_0 must
// not reach serve for a model the CLI says cannot use it; every other override,
// f16, and every override of a model whose facts are not known, is kept.
func TestSavedSwitchesTheModelCannotUseAreNotPassedToServe(t *testing.T) {
	cur := Settings{
		ModelPath:       `S:\models\Model-00001-of-00002.gguf`,
		SwitchOverrides: map[string]string{"--refusal-direction": "on", "--prefill-mode": "fast", "--kv-cache-type": "q8_0", "--max-tokens": "512"},
	}
	app := func(caps *EstimateCapabilities) *AppService {
		a := newTestApp(&fakeBackend{view: Snapshot{Status: &Status{}}}, Settings{}, nil, "")
		if caps != nil {
			subject, _ := savedEstimateSubject(cur, a.backend.Snapshot().Status)
			a.caps = capabilityState{key: capabilityKey(subject, cur), value: caps, at: time.Now()}
		}
		return a
	}
	all := cur.SwitchOverrides
	cases := []struct {
		name string
		caps *EstimateCapabilities
		want map[string]string
	}{
		{"facts not known", nil, all},
		{"the CLI reports nothing about any", &EstimateCapabilities{}, all},
		{"all work", &EstimateCapabilities{
			PrefillModeFast:  &PrefillModeFast{Available: true},
			RefusalDirection: &RefusalDirection{Available: true},
			KVCacheQ8:        &KVCacheQ8{Available: true},
		}, all},
		{"abliteration unavailable", &EstimateCapabilities{
			RefusalDirection: &RefusalDirection{Available: false, Reason: "This model is already abliterated."},
		}, map[string]string{"--prefill-mode": "fast", "--kv-cache-type": "q8_0", "--max-tokens": "512"}},
		{"q8 unavailable", &EstimateCapabilities{
			KVCacheQ8: &KVCacheQ8{Available: false, Reason: "Q8 is for Qwen3.8 Flash Next only."},
		}, map[string]string{"--refusal-direction": "on", "--prefill-mode": "fast", "--max-tokens": "512"}},
		{"all unavailable", &EstimateCapabilities{
			PrefillModeFast:  &PrefillModeFast{Available: false},
			RefusalDirection: &RefusalDirection{Available: false},
			KVCacheQ8:        &KVCacheQ8{Available: false},
		}, map[string]string{"--max-tokens": "512"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := app(c.caps).dropUnavailableSwitches(cur)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("kept %v, want %v", got, c.want)
			}
		})
	}
	if len(cur.SwitchOverrides) != 4 {
		t.Errorf("the saved map was changed: %v", cur.SwitchOverrides)
	}
	// f16 always works, whatever the CLI says about the compressed cache.
	f16 := Settings{ModelPath: cur.ModelPath, SwitchOverrides: map[string]string{"--kv-cache-type": "f16"}}
	a := newTestApp(&fakeBackend{view: Snapshot{Status: &Status{}}}, Settings{}, nil, "")
	subject, _ := savedEstimateSubject(f16, a.backend.Snapshot().Status)
	a.caps = capabilityState{key: capabilityKey(subject, f16), value: &EstimateCapabilities{KVCacheQ8: &KVCacheQ8{Available: false}}, at: time.Now()}
	if got := a.dropUnavailableSwitches(f16); !reflect.DeepEqual(got, f16.SwitchOverrides) {
		t.Errorf("f16 was dropped: %v", got)
	}
}

// Facts kept for another model are never used: after a model switch the saved
// overrides of the new model are judged by the new model's own answer.
func TestFactsOfAnotherModelAreNotUsed(t *testing.T) {
	old := Settings{ModelPath: `S:\models\Huihui-00001-of-00004.gguf`}
	next := Settings{
		ModelPath:       `S:\models\Stock-00001-of-00002.gguf`,
		SwitchOverrides: map[string]string{"--refusal-direction": "on", "--kv-cache-type": "q8_0"},
	}
	a := newTestApp(&fakeBackend{view: Snapshot{Status: &Status{}}}, Settings{}, nil, "")
	subject, _ := savedEstimateSubject(old, a.backend.Snapshot().Status)
	a.caps = capabilityState{key: capabilityKey(subject, old), at: time.Now(), value: &EstimateCapabilities{
		RefusalDirection: &RefusalDirection{Available: false, Reason: "This model is already abliterated."},
		KVCacheQ8:        &KVCacheQ8{Available: false},
	}}
	// The old model's negative facts must not strip the new model's values. With
	// no CLI to ask (a.models is nil here), the new model's facts are unknown and
	// every override stays.
	if got := a.dropUnavailableSwitches(next); !reflect.DeepEqual(got, next.SwitchOverrides) {
		t.Errorf("the previous model's facts were used: kept %v", got)
	}
	// An estimate state of another model is not a source either.
	a.estimate = savedEstimateState{value: &MemoryEstimate{Capabilities: &EstimateCapabilities{RefusalDirection: &RefusalDirection{Available: false}}}}
	if got := a.dropUnavailableSwitches(next); !reflect.DeepEqual(got, next.SwitchOverrides) {
		t.Errorf("an estimate of another model was used: kept %v", got)
	}
	// And the other way: the new model's own facts do apply.
	subject, _ = savedEstimateSubject(next, a.backend.Snapshot().Status)
	a.caps = capabilityState{key: capabilityKey(subject, next), at: time.Now(), value: &EstimateCapabilities{
		RefusalDirection: &RefusalDirection{Available: false},
	}}
	if got := a.dropUnavailableSwitches(next); !reflect.DeepEqual(got, map[string]string{"--kv-cache-type": "q8_0"}) {
		t.Errorf("the model's own facts were not applied: kept %v", got)
	}
}

// The guard is wired: a backend that takes a guard gets the app's own.
func TestAppServiceInstallsTheSwitchGuardOnItsBackend(t *testing.T) {
	b := &guardBackend{}
	a := newTestApp(b, Settings{}, nil, "")
	if b.guard == nil {
		t.Fatal("the app did not install its switch guard")
	}
	cur := Settings{ModelPath: `S:\models\m.gguf`, SwitchOverrides: map[string]string{"--refusal-direction": "on"}}
	subject, _ := savedEstimateSubject(cur, nil)
	a.caps = capabilityState{key: capabilityKey(subject, cur), at: time.Now(), value: &EstimateCapabilities{
		RefusalDirection: &RefusalDirection{Available: false},
	}}
	if got := b.guard(cur); len(got) != 0 {
		t.Errorf("an unavailable switch reached serve: %v", got)
	}
}

type guardBackend struct {
	fakeBackend
	guard func(Settings) map[string]string
}

func (b *guardBackend) SetSwitchGuard(guard func(Settings) map[string]string) {
	b.guard = guard
}

// The capabilities reach the page even when the saved estimate cannot be asked
// for (a saved serve switch is not applied yet), and come with the estimate
// otherwise.
func TestViewCarriesTheSavedModelCapabilities(t *testing.T) {
	caps := &EstimateCapabilities{RefusalDirection: &RefusalDirection{Available: false, Reason: "This model is already abliterated."}}
	var builder ViewBuilder
	v := builder.Build(ViewInput{SavedCapabilities: caps, Snapshot: Snapshot{Initialized: true, State: Stopped}})
	if v.SavedEstimate.Capabilities == nil || v.SavedEstimate.Capabilities.RefusalDirection == nil || v.SavedEstimate.Capabilities.RefusalDirection.Reason != "This model is already abliterated." {
		t.Errorf("probed capabilities missing from the view: %+v", v.SavedEstimate.Capabilities)
	}
	v = builder.Build(ViewInput{SavedEstimate: &MemoryEstimate{Capabilities: caps}, Snapshot: Snapshot{Initialized: true, State: Stopped}})
	if v.SavedEstimate.Capabilities == nil || v.SavedEstimate.Capabilities.RefusalDirection == nil {
		t.Errorf("estimate capabilities missing from the view: %+v", v.SavedEstimate.Capabilities)
	}
	v = builder.Build(ViewInput{Snapshot: Snapshot{Initialized: true, State: Stopped}})
	if v.SavedEstimate.Capabilities != nil {
		t.Errorf("capabilities invented: %+v", v.SavedEstimate.Capabilities)
	}
}

// A saved Fast / On that the CLI says the running model cannot use is dropped by
// the launch, so the restart notice must not promise it (Coder started with the
// values remembered from Flash Next). A value the model can use stays pending.
func TestSavedValuesTheModelCannotUseAreNotPendingChanges(t *testing.T) {
	registry := fixtureEvent(t, "switches").Value.(*Switches)
	saved := DefaultSettings()
	saved.SwitchOverrides = map[string]string{"--prefill-mode": "fast", "--refusal-direction": "on", "--max-tokens": "512"}
	status := &Status{EffectiveSwitches: []EffectiveSwitch{
		{Name: "--prefill-mode", Effective: json.RawMessage(`"normal"`)},
		{Name: "--refusal-direction", Effective: json.RawMessage(`"off"`)},
		{Name: "--max-tokens", Effective: json.RawMessage(`256`)},
	}}
	pending := func(caps *EstimateCapabilities) map[string]bool {
		in := ViewInput{Saved: saved, Registry: registry, SavedCapabilities: caps, Snapshot: Snapshot{Instance: &Instance{}, Status: status}}
		got := map[string]bool{}
		for _, change := range pendingChanges(in) {
			got[change.Key] = true
		}
		return got
	}
	if got := pending(nil); !got["--prefill-mode"] || !got["--refusal-direction"] {
		t.Fatalf("without facts the CLI judges: %v", got)
	}
	got := pending(&EstimateCapabilities{
		PrefillModeFast:  &PrefillModeFast{Available: false},
		RefusalDirection: &RefusalDirection{Available: false},
	})
	if got["--prefill-mode"] || got["--refusal-direction"] {
		t.Errorf("an unusable saved value is promised after restart: %v", got)
	}
	if !got["--max-tokens"] {
		t.Errorf("a usable saved value lost its pending change: %v", got)
	}
}
