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
	query := []string{"models", "estimate", "--model", subject, "--json"}
	if saved.Context != 0 {
		query = append(query, "--context", strconv.FormatUint(saved.Context, 10))
	}
	if saved.Sessions != 0 {
		query = append(query, "--sessions", strconv.FormatUint(uint64(saved.Sessions), 10))
	}
	if saved.Profile != "" {
		query = append(query, "--profile", saved.Profile)
	}
	events, err := s.captureEvents(s.ctx, query, 1)
	if err != nil {
		return &actionUnavailableError{blockedAction("estimate_unavailable", "Bubo could not check memory for this setup. The server was not started. Try again after resolving the memory check failure.")}
	}
	estimate, ok := events[0].Value.(*ModelEstimate)
	if !ok {
		return protocol("memory estimate answered with another event")
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
