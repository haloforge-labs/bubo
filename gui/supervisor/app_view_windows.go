//go:build windows

package supervisor

import (
	"reflect"
	"strings"
	"time"
)

// GetView supplies the sole read binding; projection and copies live here.
type registryProvider interface{ Registry() *Switches }

func (a *AppService) GetView() View {
	a.mu.RLock()
	saved, warning, warningCode := a.settings, a.warning, a.warningCode
	applied, appliedGeneration := a.appliedSettings, a.appliedGeneration
	window := WindowView{TrayAction: a.trayAction, TrayAvailable: a.trayAvailable, Closing: a.closing}
	a.mu.RUnlock()
	snap := a.backend.Snapshot()
	lan := a.lanState(snap, saved)
	var registry *Switches
	if p, ok := a.backend.(registryProvider); ok {
		registry = p.Registry()
	}
	var notice *Notice
	if source, ok := a.backend.(interface{ CodedNotice() Notice }); ok {
		value := source.CodedNotice()
		if value.Code != "" {
			notice = &value
		}
	} else if detail := a.backend.Notice(); detail != "" {
		notice = &Notice{Code: "internal_error", Detail: detail}
	}
	if warning != "" {
		notice = &Notice{Code: warningCode, Detail: warning}
	}
	var update Update
	if a.updates != nil {
		update = a.updates.State()
	}
	estimate, estimateErr := a.savedEstimate(saved, snap.Status, snap.Generation)
	unchanged := applied != nil && (appliedGeneration == snap.Generation || snap.AutoRestarted) && !tightSetupChanged(*applied, saved)
	return a.view.Build(ViewInput{RunningSetupUnchanged: unchanged, SavedEstimate: estimate, SavedEstimateError: estimateErr, Snapshot: snap, Registry: registry, Models: a.modelsFrame(), Saved: saved, Window: window, Key: KeyView{LANAccess: lan.LanAccess, LANActive: lan.LanActive, Present: lan.APIKeyPresent, Mask: lan.APIKeyMask, RestartNeeded: lan.KeyRestartNeeded, Unavailable: lan.Unavailable, CopyWithKey: lan.CopyWithKey, Addresses: lan.Addresses, AddressesUnavailable: lan.AddressesUnavailable}, Logs: a.backend.Logs(), Notice: notice, Diagnostic: a.backend.Diagnostic(), BundlePreview: a.backend.BundlePreview(), Update: update})
}

func (a *AppService) invoke(action, path string) ActionResult {
	return ActionResultFromError(a.runAction(action, path))
}
func (a *AppService) Start() ActionResult                   { return a.invoke("start", "") }
func (a *AppService) Stop() ActionResult                    { return a.invoke("stop", "") }
func (a *AppService) Restart() ActionResult                 { return a.invoke("restart", "") }
func (a *AppService) Prepare() ActionResult                 { return a.invoke("prepare", "") }
func (a *AppService) Cancel() ActionResult                  { return a.invoke("cancel", "") }
func (a *AppService) Approve() ActionResult                 { return a.invoke("approve", "") }
func (a *AppService) Decline() ActionResult                 { return a.invoke("decline", "") }
func (a *AppService) DismissError() ActionResult            { return a.invoke("dismiss", "") }
func (a *AppService) Manage() ActionResult                  { return a.invoke("manage", "") }
func (a *AppService) SelectInstance(id string) ActionResult { return a.invoke("select", id) }
func (a *AppService) Diagnose() ActionResult                { return a.invoke("diagnose", "") }
func (a *AppService) PreviewBundle() ActionResult           { return a.invoke("preview", "") }
func (a *AppService) SaveBundle(path string) ActionResult   { return a.invoke("bundle", path) }

// Exit retains the window-close contract.
func (a *AppService) Exit() ActionResult {
	a.mu.RLock()
	ctx := a.ctx
	a.mu.RUnlock()
	a.closeFlow(ctx)
	return ActionResult{}
}

func (a *AppService) CheckUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.CheckUpdate()
}
func (a *AppService) InstallUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.InstallUpdate()
}
func (a *AppService) RecoverUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.InstallUpdate()
}
func (a *AppService) ClearReleaseHistory() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.ClearReleaseHistory()
}
func (a *AppService) DownloadUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.DownloadUpdate()
}
func (a *AppService) CancelUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.CancelUpdate()
}
func (a *AppService) DismissUpdate() ActionResult {
	if a.updates == nil {
		return codedAction("internal_error", "Updater is not attached.")
	}
	return a.updates.DismissUpdate()
}

// The public model actions expose one typed result convention to Wails.
func (a *AppService) Download(id string) ModelPlanResult {
	p, err := a.StartModelDownload(id)
	if err != nil {
		return ModelPlanResult{Error: ActionResultFromError(err).Error}
	}
	return ModelPlanResult{Plan: &p}
}
func (a *AppService) ConfirmDownload(id string) ActionResult {
	return ActionResultFromError(a.ConfirmModelDownload(id))
}
func (a *AppService) CancelDownload() ActionResult {
	return ActionResultFromError(a.CancelModelDownload())
}

// PatchPreferences is the preference write for new pages. The supervisor owns
// the settings version; omitted fields survive and clear removes an override.
func (a *AppService) PatchPreferences(patch PreferencesPatch) ActionResult {
	var err error
	patch, err = normalizePreferencePatch(patch)
	if err != nil {
		return ActionResultFromError(err)
	}
	a.folderMu.Lock()
	defer a.folderMu.Unlock()
	a.mu.RLock()
	current := a.settings
	a.mu.RUnlock()
	var registry *Switches
	if provider, ok := a.backend.(registryProvider); ok {
		registry = provider.Registry()
	}
	// A single patch can change app preferences and writable registry rows.
	// Split clear names by the preference schema; old override names may be
	// cleared even after a registry update removed their row.
	var switchClear []string
	preferenceFields := map[string]bool{}
	typeOfPreferences := reflect.TypeOf(PreferenceValues{})
	for i := 0; i < typeOfPreferences.NumField(); i++ {
		preferenceFields[strings.Split(typeOfPreferences.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	var preferenceClear []string
	for _, name := range patch.Clear {
		if preferenceFields[name] {
			preferenceClear = append(preferenceClear, name)
		} else if _, saved := current.SwitchOverrides[name]; saved {
			switchClear = append(switchClear, name)
		} else if registryHasSwitch(registry, name) {
			switchClear = append(switchClear, name)
		} else {
			return ActionResultFromError(appError("internal_error", "Invalid preference clear: "+name))
		}
	}
	patch.Clear = preferenceClear
	next, err := ApplyPreferencesPatch(current, patch)
	if err != nil {
		return ActionResultFromError(err)
	}
	if registry != nil {
		if err := validatePreferenceRegistryWrites(patch, registry); err != nil {
			return ActionResultFromError(err)
		}
	}
	baseOverrides := current.SwitchOverrides
	if modelChanged(current, next) {
		// Old model parts must be removed before merging explicit choices for
		// the new model in the same setup transaction.
		baseOverrides = dropModelFileOverrides(baseOverrides, registry)
	}
	next.SwitchOverrides, err = ApplySwitchOverrides(baseOverrides, SwitchOverridesPatch{Set: patch.SwitchValues, Clear: switchClear}, registry)
	if err != nil {
		return ActionResultFromError(err)
	}
	if patch.Set.AllowTightSetup != nil {
		if *patch.Set.AllowTightSetup {
			if tightSetupChanged(current, next) {
				return ActionResultFromError(appError("invalid_arguments", "Save the setup before allowing a tight start."))
			}
			key, ok := tightSetupFingerprint(next, a.backend.Snapshot().Status, registry)
			if !ok {
				return ActionResultFromError(appError("estimate_unavailable", "Bubo cannot identify the current saved setup for a tight start."))
			}
			next.TightSetupFingerprint = key
		} else {
			next.TightSetupFingerprint = ""
		}
	}
	for _, name := range patch.Clear {
		if name == "allow_tight_setup" {
			next.TightSetupFingerprint = ""
		}
	}
	folderChanged, err := a.applyDownloadFolderPatch(current, &next, patch)
	if err != nil {
		return ActionResultFromError(err)
	}
	if err := a.saveSettingsLocked(next); err != nil {
		return ActionResultFromError(err)
	}
	if folderChanged && a.models != nil {
		// The space line and the package list follow the new folder at once.
		a.mu.Lock()
		a.estimate.at = time.Time{}
		a.mu.Unlock()
		a.models.Refresh()
	}
	return ActionResult{}
}

// SetSwitchOverrides is the data-driven engine write for new pages. Values are
// opaque strings; only the CLI validates their engine meaning at start.
func (a *AppService) SetSwitchOverrides(patch SwitchOverridesPatch) ActionResult {
	a.folderMu.Lock()
	defer a.folderMu.Unlock()
	a.mu.RLock()
	next := a.settings
	a.mu.RUnlock()
	var registry *Switches
	if provider, ok := a.backend.(registryProvider); ok {
		registry = provider.Registry()
	}
	overrides, err := ApplySwitchOverrides(next.SwitchOverrides, patch, registry)
	if err != nil {
		return ActionResultFromError(err)
	}
	next.SwitchOverrides = overrides
	return ActionResultFromError(a.saveSettingsLocked(next))
}

func (a *AppService) Estimate(subject string, contextTokens uint64, sessions uint32) ModelEstimateResult {
	value, err := a.EstimateModel(subject, contextTokens, sessions)
	return ModelEstimateResultFrom(value, err)
}
