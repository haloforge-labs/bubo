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
				return nil, codedAction("estimate_needs_restart", "The CLI cannot estimate this saved serve preference before it is applied.").Error
			}
		}
	}
	for name, value := range saved.SwitchOverrides {
		raw := effectiveSwitchValue(st.EffectiveSwitches, name)
		now, ok := switchScalar(raw)
		if !ok || !equalSwitchValue(raw, now, value) {
			return nil, codedAction("estimate_needs_restart", "The CLI estimate command does not accept saved serve-switch overrides.").Error
		}
	}
	request := struct {
		Subject, Runtime, Profile, Manifest string
		Context                             uint64
		Sessions                            uint32
	}{subject, saved.RuntimePath, saved.Profile, saved.ManifestPath, contextTokens, sessions}
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
		value, err := a.models.estimateRuntime(subject, contextTokens, sessions, saved.RuntimePath, saved.Profile)
		a.mu.Lock()
		a.estimate = savedEstimateState{key: key, value: value.MemoryEstimate, err: ActionResultFromError(err).Error, at: time.Now()}
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
