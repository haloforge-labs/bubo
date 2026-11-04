//go:build windows

package supervisor

import (
	"encoding/json"
	"path/filepath"
	"time"
)

// App cache policy: changing machine budgets must not pin a refusal forever.
const savedEstimateMaxAge = 30 * time.Second

type savedEstimateState struct {
	key   string
	busy  bool
	value *MemoryEstimate
	err   *CLIError
	at    time.Time
}

// capabilityState keeps the CLI's facts about the saved model (what the model
// can use) when the saved estimate cannot be asked for, because a saved serve
// switch is not applied yet. The facts do not depend on those switches.
type capabilityState struct {
	key   string
	busy  bool
	value *EstimateCapabilities
	at    time.Time
}

func capabilityKey(subject string, saved Settings) string {
	return subject + "|" + saved.RuntimePath + "|" + saved.Profile
}

// probeCapabilities asks the CLI for the model's capabilities in the
// background, once per saved model, and never waits.
func (a *AppService) probeCapabilities(subject string, saved Settings) {
	key := capabilityKey(subject, saved)
	a.mu.Lock()
	if a.caps.key == key && (a.caps.busy || time.Since(a.caps.at) < savedEstimateMaxAge) {
		a.mu.Unlock()
		return
	}
	kept := a.caps.value
	if a.caps.key != key {
		kept = nil
	}
	a.caps = capabilityState{key: key, busy: true, value: kept}
	a.mu.Unlock()
	go func() {
		value, err := a.models.estimateRuntime(subject, 0, 0, saved.RuntimePath, saved.Profile, nil)
		a.mu.Lock()
		defer a.mu.Unlock()
		a.caps = capabilityState{key: key, at: time.Now()}
		if err == nil {
			a.caps.value = value.Capabilities
		}
	}()
}

// capabilitySwitches are the serve switches whose use the CLI reports per
// model. When it says a switch value does not work for the saved model, that
// saved value is not passed to serve.
var capabilitySwitches = map[string]func(c *EstimateCapabilities, value string) bool{
	"--prefill-mode": func(c *EstimateCapabilities, _ string) bool {
		return c.PrefillModeFast != nil && !c.PrefillModeFast.Available
	},
	"--refusal-direction": func(c *EstimateCapabilities, _ string) bool {
		return c.RefusalDirection != nil && !c.RefusalDirection.Available
	},
	// Only the compressed values need the capability; f16 always works.
	"--kv-cache-type": func(c *EstimateCapabilities, value string) bool {
		return c.KVCacheQ8 != nil && !c.KVCacheQ8.Available && value != "f16"
	},
}

// inapplicableSwitch reports that the CLI says the model cannot use this saved
// value. The launch drops such a value, so nothing that compares saved values
// with a running or estimated setup (the restart notice, the estimate) counts it.
func inapplicableSwitch(caps *EstimateCapabilities, name, value string) bool {
	unavailable, ok := capabilitySwitches[name]
	return caps != nil && ok && unavailable(caps, value)
}

// ensureCapabilities returns the CLI's facts about this exact model. A fact
// kept for another model is never used. When none is kept yet it asks the CLI
// now and waits for the answer: the start must not go on with a guess. An
// answer that cannot be had leaves the overrides to the CLI to judge.
func (a *AppService) ensureCapabilities(subject string, cur Settings) *EstimateCapabilities {
	key := capabilityKey(subject, cur)
	a.mu.RLock()
	if a.caps.key == key && a.caps.value != nil {
		value := a.caps.value
		a.mu.RUnlock()
		return value
	}
	a.mu.RUnlock()
	if a.models == nil {
		return nil
	}
	value, err := a.models.estimateRuntime(subject, 0, 0, cur.RuntimePath, cur.Profile, nil)
	if err != nil || value.Capabilities == nil {
		return nil
	}
	a.mu.Lock()
	a.caps = capabilityState{key: key, value: value.Capabilities, at: time.Now()}
	a.mu.Unlock()
	return value.Capabilities
}

// dropUnavailableSwitches returns the saved overrides without the values that
// the CLI says the model of cur cannot use. The facts must belong to that exact
// model: the running status's estimate when it matches the setup, else the
// kept or freshly asked answer for the model's own key.
func (a *AppService) dropUnavailableSwitches(cur Settings) map[string]string {
	overrides := cur.SwitchOverrides
	if len(overrides) == 0 {
		return overrides
	}
	st := a.backend.Snapshot().Status
	var caps *EstimateCapabilities
	if statusEstimateMatches(cur, st) {
		caps = st.MemoryEstimate.Capabilities
	}
	if caps == nil {
		if subject, _ := savedEstimateSubject(cur, st); subject != "" {
			caps = a.ensureCapabilities(subject, cur)
		}
	}
	if caps == nil {
		return overrides
	}
	kept := make(map[string]string, len(overrides))
	for name, value := range overrides {
		if inapplicableSwitch(caps, name, value) {
			continue
		}
		kept[name] = value
	}
	return kept
}

// savedCapabilities returns the probed capabilities of the saved model, or nil.
func (a *AppService) savedCapabilities(saved Settings, st *Status) *EstimateCapabilities {
	if st == nil {
		return nil
	}
	if statusEstimateMatches(saved, st) && st.MemoryEstimate != nil && st.MemoryEstimate.Capabilities != nil {
		return st.MemoryEstimate.Capabilities
	}
	subject, _ := savedEstimateSubject(saved, st)
	if subject == "" {
		return nil
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.caps.key != capabilityKey(subject, saved) {
		return nil
	}
	return a.caps.value
}

// GetView never waits on CLI I/O. A saved request gets its own CLI estimate;
// the form's independent Estimate binding cannot populate this cache.
func (a *AppService) savedEstimate(saved Settings, st *Status, generations ...uint64) (*MemoryEstimate, *CLIError) {
	if len(generations) > 0 && a.clearedAppliedPreference(saved, generations[0]) {
		return nil, codedAction("estimate_needs_restart", "The CLI cannot estimate the default after clearing a preference used by the running server.").Error
	}
	if statusEstimateMatches(saved, st) {
		return st.MemoryEstimate, nil
	}
	if a.models == nil || st == nil {
		return nil, nil
	}
	subject, activeManifest := savedEstimateSubject(saved, st)
	if subject == "" {
		return nil, codedAction("estimate_no_model", "The CLI has not identified the saved model for estimation.").Error
	}
	contextTokens, sessions := saved.Context, saved.Sessions
	// Zero leaves defaults with the CLI, including before the first server.
	// A prior allocation's values are not defaults for a new model request.
	// This planner forwards model/context/sessions only. Other unapplied serve
	// preferences cannot be silently treated as covered by that narrower request.
	for _, arg := range serveArguments(saved, "", 0) {
		if !arg.base && !arg.reserved && arg.preference != "model_path" && arg.preference != "context" && arg.preference != "sessions" && arg.value != "" {
			if arg.preference == "manifest_path" && activeManifest != "" && SameFolder(arg.value, activeManifest) {
				continue
			}
			// The estimate of a package folder covers its manifest.
			if arg.preference == "manifest_path" && SameFolder(subject, filepath.Dir(arg.value)) {
				continue
			}
			now, ok := switchScalar(observedPreference(st, arg))
			if !ok || now != arg.value {
				a.probeCapabilities(subject, saved)
				return nil, codedAction("estimate_needs_restart", "The CLI cannot estimate this saved serve preference before it is applied.").Error
			}
		}
	}
	// Every saved serve override goes to the estimate, as `--switch NAME=VALUE`: the CLI
	// resolves them as serve does, so the fit is the one the launch will have.
	overrides := estimateSwitchArgs(saved.SwitchOverrides)
	request := struct {
		Subject, Runtime, Profile, Manifest string
		Context                             uint64
		Sessions                            uint32
		Overrides                           []string
	}{subject, saved.RuntimePath, saved.Profile, saved.ManifestPath, contextTokens, sessions, overrides}
	keyBytes, _ := json.Marshal(request)
	key := string(keyBytes)
	a.mu.Lock()
	if a.estimate.key == key && (a.estimate.busy || time.Since(a.estimate.at) < savedEstimateMaxAge) {
		value, err := a.estimate.value, a.estimate.err
		a.mu.Unlock()
		return value, err
	}
	if a.estimate.busy {
		a.mu.Unlock()
		return nil, nil
	}
	a.estimate = savedEstimateState{key: key, busy: true}
	a.mu.Unlock()
	go func() {
		value, err := a.models.estimateRuntime(subject, contextTokens, sessions, saved.RuntimePath, saved.Profile, saved.SwitchOverrides)
		a.mu.Lock()
		a.estimate = savedEstimateState{key: key, value: value.MemoryEstimate, err: ActionResultFromError(err).Error, at: time.Now()}
		if err == nil && value.Capabilities != nil {
			a.caps = capabilityState{key: capabilityKey(subject, saved), value: value.Capabilities, at: time.Now()}
		}
		// A package without a measured memory table has no fit verdict: the CLI
		// answers none_yet with no estimate, or an estimate whose fits is null.
		// That is not a failure; the server checks memory when it starts.
		if err == nil && value.MemoryEstimate == nil {
			reason := "The CLI has no memory estimate for this saved setup."
			if value.Reason != nil && *value.Reason != "" {
				reason = *value.Reason
			}
			a.estimate.err = codedAction("estimate_no_verdict", reason).Error
		} else if err == nil && value.MemoryEstimate.fitVerdict() == nil {
			a.estimate.err = codedAction("estimate_no_verdict", "The CLI has no fit verdict for this saved setup.").Error
		}
		a.mu.Unlock()
	}()
	return nil, nil
}

// One rule for every action on a model: pass its path. A package is estimated
// by its folder (the CLI reads the marker there), a model file by itself, never
// by a catalogue id. The folder comes from the saved manifest, else from the
// manifest the CLI reports as active.
func savedEstimateSubject(saved Settings, st *Status) (subject, activeManifest string) {
	if st == nil {
		return saved.ModelPath, ""
	}
	if st.ActiveModel != nil && st.ActiveModel.ManifestPath != nil {
		activeManifest = *st.ActiveModel.ManifestPath
	}
	if activeManifest == "" {
		activeManifest, _ = switchScalar(st.Paths["manifest"])
	}
	if activeManifest == "" && st.PreparedDefault != nil {
		activeManifest = *st.PreparedDefault
	}
	if saved.ModelPath != "" {
		return saved.ModelPath, activeManifest
	}
	if saved.ManifestPath != "" {


		return filepath.Dir(saved.ManifestPath), activeManifest
	}
	if activeManifest != "" {
		return filepath.Dir(activeManifest), activeManifest
	}
	if st.ActiveModel != nil && st.ActiveModel.ModelPath != nil && *st.ActiveModel.ModelPath != "" {
		return *st.ActiveModel.ModelPath, activeManifest
	}
	return "", activeManifest
}
