//go:build windows

package supervisor

import "strconv"

// Check after the previous instance has exited. This is the common launch
// boundary for Start, Restart and GPU recovery; an earlier estimate may include
// the old instance's allocation. Terminal serve is unaffected.
func startSettingsForArguments(saved Settings, args []string) Settings {
	// Judge the arguments actually being launched. Save can run during Stop;
	// newer preferences must not change the request already in progress.
	saved.ModelPath, saved.ManifestPath, saved.Profile = "", "", ""
	saved.Context, saved.Sessions = 0, 0
	for i := 0; i+1 < len(args); i++ {
		switch args[i] {
		case "--model":
			saved.ModelPath = args[i+1]
		case "--manifest":
			saved.ManifestPath = args[i+1]
		case "--profile":
			saved.Profile = args[i+1]
		case "--context":
			saved.Context, _ = strconv.ParseUint(args[i+1], 10, 64)
		case "--sessions":
			value, _ := strconv.ParseUint(args[i+1], 10, 32)
			saved.Sessions = uint32(value)
		}
	}
	return saved
}

// recommendedSessionContext is the per-session context of the CLI's recommended
// row for this session count, or 0 when the CLI recommends none.
func recommendedSessionContext(estimate *MemoryEstimate, sessions uint32) uint64 {
	if estimate == nil || estimate.RecommendedContext == nil {
		return 0
	}
	for _, row := range estimate.RecommendedContext.BySessions {
		if uint64(row.Sessions) == uint64(sessions) && row.Recommended != nil && *row.Recommended {
			return uint64(row.ContextPerSession)
		}
	}
	return 0
}

func (s *Supervisor) admitSavedStart(saved Settings, st *Status) error {
	// Bare supervisors can have callers without app preferences. The desktop
	// backend always supplies its settings path.
	if s.config.SettingsPath == "" {
		return nil
	}
	subject, _ := savedEstimateSubject(saved, st)
	if subject == "" {
		return appError("estimate_unavailable", "Select a model before checking memory.")
	}
	// The request is the one behind the fit shown before the start, every saved
	// override included: a setup that fits with a q8_0 cache must not be judged at
	// f16, and the side files (MTP, mmproj) count.
	query := estimateArguments(subject, saved.Context, saved.Sessions, saved.Profile, saved.SwitchOverrides)
	unavailable := &actionUnavailableError{blockedAction("estimate_unavailable", "Bubo could not check memory for this setup. The server was not started. Try again after resolving the memory check failure.")}
	events, err := s.captureEvents(s.ctx, query, 1)
	if err != nil {
		return unavailable
	}
	estimate, ok := events[0].Value.(*ModelEstimate)
	if !ok {
		return protocol("memory estimate answered with another event")
	}
	// With several sessions and no saved context, serve starts at the context it
	// recommends for that session count, not at the default the estimate used
	// above. Judge the setup serve will run.
	if saved.Context == 0 && saved.Sessions > 1 {
		if context := recommendedSessionContext(estimate.MemoryEstimate, saved.Sessions); context != 0 {
			events, err = s.captureEvents(s.ctx, estimateArguments(subject, context, saved.Sessions, saved.Profile, saved.SwitchOverrides), 1)
			if err != nil {
				return unavailable
			}
			if estimate, ok = events[0].Value.(*ModelEstimate); !ok {
				return protocol("memory estimate answered with another event")
			}
		}
	}
	if estimate.MemoryEstimate == nil || estimate.MemoryEstimate.fitVerdict() == nil {
		// Supported models without a measured table retain the CLI's own
		// load-time admission.
		return nil
	}
	fit := buildFit(estimate.MemoryEstimate)
	if !*estimate.MemoryEstimate.Fits {
		return &actionUnavailableError{blockedAction(fit.Class, fit.Sentence)}
	}
	if fit.Class == "memory_tight" {
		if s.Registry() == nil {
			if err := s.LoadRegistry(); err != nil {
				return err
			}
		}
		// Recovery pins its launch arguments, not a past permission. A Save
		// may have changed the setup or explicitly unticked the override since
		// Ready. Only the consent still saved on disk can authorize this start.
		current, _ := LoadSettings(s.config.SettingsPath)
		saved.AllowTightSetup, saved.TightSetupFingerprint = current.AllowTightSetup, current.TightSetupFingerprint
		if !tightConsentActive(saved, st, s.Registry()) {
			return &actionUnavailableError{blockedAction("memory_tight", fit.Sentence)}
		}
	}
	return nil
}
