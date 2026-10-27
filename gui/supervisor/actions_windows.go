//go:build windows

package supervisor

import "fmt"

// ActionAvailability is app policy, never an inferred engine verdict.
type ActionAvailability struct {
	Allowed    bool   `json:"allowed"`
	ReasonCode string `json:"reason_code"`
	Reason     string `json:"reason"`
}

type LifecycleActions struct {
	Start           ActionAvailability `json:"start"`
	Stop            ActionAvailability `json:"stop"`
	Restart         ActionAvailability `json:"restart"`
	Prepare         ActionAvailability `json:"prepare"`
	DownloadDefault ActionAvailability `json:"download_default"`
	Check           ActionAvailability `json:"check"`
	Install         ActionAvailability `json:"install"`
	Recover         ActionAvailability `json:"recover"`
	Diagnose        ActionAvailability `json:"diagnose"`
	SupportBundle   ActionAvailability `json:"support_bundle"`
}

// ActionNotApplicable distinguishes an absent lifecycle action from one whose
// prerequisites the user can repair. The frontend hides only the former.
const ActionNotApplicable = "not_applicable"

func blockedAction(code, reason string) ActionAvailability {
	return ActionAvailability{ReasonCode: code, Reason: reason}
}

// fitUnchecked allows a start or restart whose saved setup the CLI could not
// estimate (an unapplied profile, NPU, effort, manifest or override request).
// Serve applies the CLI's admission rule when it starts and refuses a setup
// that does not fit, so an unknown verdict must not lock the saved setting out
// (G-ENG round 2, B1). Only the CLI's fits:false blocks. The reason stays as a
// note on the allowed action.
func fitUnchecked(why string) ActionAvailability {
	return ActionAvailability{Allowed: true, ReasonCode: "estimate_unavailable", Reason: why + " The server checks memory when it starts."}
}

func hasTransition(state State, action string) bool {
	for _, t := range transitionTable {
		if t.From == state && t.Action == action {
			return true
		}
	}
	return false
}

// lifecycleActions is shared by View, native tray and action dispatch. Fit is
// always the CLI verdict for the saved requests; drafts never enter this rule.
func lifecycleActions(in ViewInput) LifecycleActions {
	s := in.Snapshot
	state := blockedAction(ActionNotApplicable, fmt.Sprintf("This action is unavailable while %s.", s.State))
	a := LifecycleActions{Start: state, Stop: state, Restart: state}
	a.Prepare, a.DownloadDefault = state, state
	a.Check, a.Install, a.Recover = ActionAvailability{Allowed: true}, state, state
	if !in.Update.Ready && !in.Update.CheckFailed {
		a.Check = blockedAction("updater_starting", "Bubo is still getting ready to check for updates.")
		if in.Update.Error != nil {
			a.Check = blockedAction(in.Update.Error.Code, in.Update.Error.Remedy)
		}
	}
	// A failed start is retried by the next Check (CheckUpdate builds the
	// updater again), so every can't-check state offers the same Check.
	a.Diagnose, a.SupportBundle = ActionAvailability{Allowed: true}, ActionAvailability{Allowed: true}
	if !s.Initialized {
		a.Diagnose = blockedAction("initializing", "Waiting for Bubo to report its status.")
	} else if s.LastError != nil && s.LastError.Code == "cli_missing" {
		a.Diagnose = blockedAction("cli_missing", "Bubo's CLI is missing.")
	} else if s.State == Preparing || s.State == Starting || s.State == Stopping || (in.Models != nil && in.Models.Download != nil && in.Models.Download.State == "running") {
		a.Diagnose = blockedAction("busy", "Bubo is busy.")
	}
	a.SupportBundle = a.Diagnose
	if in.Update.State == UpdateRollbackPending {
		a.Check = blockedAction("rollback_pending", "Recover the previous version first.")
		a.Recover = ActionAvailability{Allowed: true}
	} else if in.Update.State == UpdateReady {
		a.Install = ActionAvailability{Allowed: true}
	}
	if !s.Initialized {
		a.Prepare = blockedAction("initializing", "Waiting for Bubo to report its status.")
		a.DownloadDefault = a.Prepare
	} else if s.LastError != nil && s.LastError.Code == "cli_missing" {
		a.Prepare = blockedAction("cli_missing", "Bubo's CLI is missing.")
		a.DownloadDefault = a.Prepare
	} else if s.State == Preparing || s.State == Starting || (in.Models != nil && in.Models.Download != nil && in.Models.Download.State == "running") {
		a.Prepare = blockedAction("busy", "Bubo is busy.")
		a.DownloadDefault = a.Prepare
	} else if s.LastError != nil && (s.LastError.Code == "runtime_missing" || s.LastError.Code == "source_missing") {
		a.Prepare = ActionAvailability{Allowed: true}
	} else if s.Status != nil {
		setupState := s.State == NotPrepared || s.State == Stopped || s.State == Failed
		if !s.Status.Readiness.Runtime && setupState {
			a.Prepare = ActionAvailability{Allowed: true}
			a.DownloadDefault = blockedAction("runtime_missing", "Download the engine files first.")
		} else if !s.Status.Readiness.Model && setupState {
			if in.Models != nil {
				for _, entry := range in.Models.Catalogue {
					if entry.Role == "recommended" && entry.DefaultStatus == "default" && entry.DownloadEnabled {
						a.DownloadDefault = ActionAvailability{Allowed: true}
						break
					}
				}
			}
		}
	}
	managed := s.State == AttachedExternal && s.Managed && s.ManagementSupported
	if hasTransition(s.State, "stop") || managed {
		a.Stop = ActionAvailability{Allowed: true}
	}
	if s.State == AttachedExternal && !managed {
		a.Stop = blockedAction("external_read_only", s.ManagementMessage)
		if !s.ManagementSupported || a.Stop.Reason == "" {
			a.Stop.Reason = "This server was started outside Bubo, for example in a terminal. Stop it there, then start it here."
		}
		a.Restart = a.Stop
	}
	if (hasTransition(s.State, "start") || s.State == NotPrepared) && s.Instance == nil {
		a.Start = savedSetupAvailability(in)
		if s.State == NotPrepared && a.Start.Allowed {
			a.Start = blockedAction("not_prepared", "Prepare the saved setup before starting.")
		}
	}
	if s.State == Ready || managed {
		a.Restart = savedSetupAvailability(in)
	}
	if s.State == Preparing || s.State == Starting || (in.Models != nil && in.Models.Download != nil && in.Models.Download.State == "running") {
		a.Start = blockedAction("busy", "Bubo is busy.")
		a.Restart = a.Start
	}
	if s.LastError != nil && s.LastError.Code == "cli_missing" {
		a.Start = blockedAction("cli_missing", "Bubo's CLI is missing.")
	}
	if in.Update.RuntimeFollowupRequired {
		a.Start = blockedAction("runtime_missing", "Prepare the engine files for this Bubo version before starting.")
	}
	if in.Update.State == UpdateRollbackPending {
		a.Start = blockedAction("rollback_pending", "Recover the previous version before starting.")
	}
	if in.Window.Closing {
		b := blockedAction("closing", "The GUI is exiting.")
		return LifecycleActions{Start: b, Stop: b, Restart: b, Prepare: b, DownloadDefault: b, Check: b, Install: b, Recover: b, Diagnose: b, SupportBundle: b}
	}
	return a
}

func savedSetupAvailability(in ViewInput) ActionAvailability {
	s := in.Snapshot
	if !s.Initialized {
		return blockedAction("initializing", "Waiting for Bubo to report its status.")
	}
	if s.Status == nil {
		return blockedAction("readiness_unavailable", "The CLI has not reported readiness.")
	}
	// A failed preparation/start can leave an older readiness sample intact.
	// Retain the CLI's missing-file verdict until preparation or a fresh sample
	// clears it; lifecycle Stop does not depend on the next setup's readiness.
	var failureCode, failureRemedy string
	if s.LastError != nil {
		if name := s.LastError.Switch; name != "" {
			saved, stillSet := in.Saved.SwitchOverrides[name]
			attempted, known := s.RunningSwitchOverrides[name]
			if stillSet && known && saved == attempted {
				return blockedAction("setting_rejected", "Reset or review the rejected server setting.")
			}
		}
		failureCode, failureRemedy = s.LastError.Code, s.LastError.Remedy
	} else if s.Status.LastError != nil {
		failureCode, failureRemedy = s.Status.LastError.Code, s.Status.LastError.Remedy
	}
	switch failureCode {
	case "runtime_missing", "model_missing", "source_missing":
		return blockedAction(failureCode, failureRemedy)
	}
	if !s.Status.Readiness.Runtime {
		return blockedAction("runtime_missing", "Download or select the runtime before starting.")
	}
	// The readiness describes the default package; a selected model does not need it.
	if !s.Status.Readiness.Model && in.Saved.ModelPath == "" && in.Saved.ManifestPath == "" {
		return blockedAction("model_missing", "Download or select a model before starting.")
	}
	if !s.Prepared {
		return blockedAction("not_prepared", "Prepare the saved setup before starting.")
	}
	// The unchanged running setup's sample includes its own allocation. A
	// changed setup's separate estimate must still enforce Tight consent.
	// Every actual launch is checked again after releasing the old instance.
	if s.State == Ready && (in.RunningSetupUnchanged || statusEstimateMatches(in.Saved, s.Status)) {
		return ActionAvailability{Allowed: true, ReasonCode: "fit_unchecked", Reason: "Bubo's server holds the memory now. Memory is checked when it restarts."}
	}
	for name := range s.RunningSwitchOverrides {
		if _, saved := in.Saved.SwitchOverrides[name]; !saved {
			return fitUnchecked("The CLI cannot estimate the default after clearing a running serve override.")
		}
	}
	if in.SavedEstimateError != nil {
		return fitUnchecked(in.SavedEstimateError.Remedy)
	}
	estimate := in.SavedEstimate
	if estimate == nil && statusEstimateMatches(in.Saved, s.Status) {
		estimate = s.Status.MemoryEstimate
	}
	if estimate == nil {
		return blockedAction("estimate_unavailable", "The CLI has not supplied a memory estimate for the saved setup.")
	}
	if estimate.fitVerdict() == nil {
		// No verdict: no measured table for this package, or an estimate that is
		// only a lower bound. The server checks memory when it starts.
		reason := "The CLI has no fit verdict for this saved setup."
		if estimate.Fits == nil && estimate.FitReason != nil && *estimate.FitReason != "" {
			reason = *estimate.FitReason
		}
		return fitUnchecked(reason)
	}
	fit := buildFit(estimate)
	if !*estimate.Fits || fit.Class == "memory_tight" && !tightConsentActive(in.Saved, s.Status, in.Registry) {
		if !*estimate.Fits && estimate.FitReason != nil && *estimate.FitReason != "" {
			return blockedAction(fit.Class, *estimate.FitReason)
		}
		return blockedAction(fit.Class, fit.Sentence)
	}
	return ActionAvailability{Allowed: true}
}

// An observed estimate belongs to the CLI's observed model/settings. Never
// apply it to a changed saved context, sessions, profile or model request.
func statusEstimateMatches(saved Settings, st *Status) bool {
	if st == nil || st.MemoryEstimate == nil {
		return false
	}
	if saved.RuntimePath != "" {
		if path, known := switchScalar(st.Paths["runtime"]); !known || !SameFolder(path, saved.RuntimePath) {
			return false
		}
	}
	if saved.Context != 0 && saved.Context != uint64(st.MemoryEstimate.ContextTokens) {
		return false
	}
	if saved.Sessions != 0 && uint64(saved.Sessions) != uint64(st.MemoryEstimate.Sessions) {
		return false
	}
	for _, arg := range serveArguments(saved, "", 0) {
		if arg.value == "" || arg.base || arg.reserved {
			continue
		}
		if arg.preference != "context" && arg.preference != "sessions" {
			raw := observedPreference(st, arg)
			current, ok := switchScalar(raw)
			if !ok || current != arg.value {
				return false
			}
		}
	}
	// An override may change memory behavior; the models estimate command does
	// not accept arbitrary serve overrides. Do not claim it covers them.
	for name, value := range saved.SwitchOverrides {
		raw := effectiveSwitchValue(st.EffectiveSwitches, name)
		now, ok := switchScalar(raw)
		if !ok || !equalSwitchValue(raw, now, value) {
			return false
		}
	}
	return true
}
