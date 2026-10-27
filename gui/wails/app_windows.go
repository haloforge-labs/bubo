//go:build windows

package main

import (
	"bubo/gui/supervisor"
	"context"
	"net/url"
	"strings"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

// App is the native Wails adapter. All app state and decisions live in AppService.
type App struct {
	service *supervisor.AppService
	ctx     context.Context
}

var copyTextNative = runtime.ClipboardSetText
var openExternalNative = func(ctx context.Context, url string) error {
	runtime.BrowserOpenURL(ctx, url)
	return nil
}
var appHooks = func() supervisor.AppHooks {
	return supervisor.AppHooks{
		Quit: runtime.Quit, Show: runtime.WindowShow, Hide: runtime.WindowHide,
		Tray: nativeTray{}, Dialogs: wailsDialogs{}, Folders: nativeFolderOpener{}, Events: emitWailsStream,
	}
}
var rejectLANPreference = func(supervisor.PreferencesPatch) *supervisor.ActionResult { return nil }

// Preserve the generated main.BrowseResult name; fields come from the service.
type BrowseResult supervisor.BrowseResult

func newApp(b supervisor.Backend, settings supervisor.Settings, lease *supervisor.SingleInstance, warning string) *App {
	return &App{service: supervisor.NewAppService(b, settings, lease, warning, appHooks())}
}
func (a *App) startup(ctx context.Context)          { a.ctx = ctx; a.service.Startup(ctx) }
func (a *App) domReady(ctx context.Context)         { a.service.DOMReady(ctx) }
func (a *App) beforeClose(ctx context.Context) bool { return a.service.BeforeClose(ctx) }
func (a *App) shutdown(ctx context.Context)         { a.service.Shutdown(ctx) }
func (a *App) emitStream(e supervisor.StreamEvent)  { a.service.EmitStream(e) }
func (a *App) quitForUpdate()                       { a.service.QuitForUpdate() }

func (a *App) RevealKey() supervisor.StringResult { return supervisor.Filled(a.service.RevealKey()) }
func (a *App) ResetKey() supervisor.ActionResult  { return supervisor.Filled(a.service.ResetKey()) }
func (a *App) TakeTrayAction() supervisor.StringResult {
	return supervisor.Filled(a.service.TakeTrayAction())
}
func (a *App) RefreshModels() supervisor.ActionResult {
	return supervisor.Filled(a.service.RefreshModels())
}
func (a *App) CheckModel(id string, repair bool) supervisor.ModelPlanResult {
	return supervisor.Filled(a.service.CheckModel(id, repair))
}
func (a *App) RepairPackage(id string, output string) supervisor.ModelPlanResult {
	return supervisor.Filled(a.service.RepairPackage(id, output))
}
func (a *App) AddModelFolder(path string) supervisor.StringResult {
	return supervisor.Filled(a.service.AddModelFolder(path))
}
func (a *App) RemoveModelFolder(path string) supervisor.ActionResult {
	return supervisor.Filled(a.service.RemoveModelFolder(path))
}
func (a *App) BrowseDownloadFolder() BrowseResult {
	return supervisor.Filled(BrowseResult(a.service.BrowseDownloadFolder()))
}
func (a *App) BrowseModelFolder() BrowseResult {
	return supervisor.Filled(BrowseResult(a.service.BrowseModelFolder()))
}
func (a *App) BrowseModelFile() BrowseResult {
	return supervisor.Filled(BrowseResult(a.service.BrowseModelFile()))
}
func (a *App) OpenModelFolder(path string) supervisor.ActionResult {
	return supervisor.Filled(a.service.OpenModelFolder(path))
}
func (a *App) GetView() supervisor.View         { return supervisor.Filled(a.service.GetView()) }
func (a *App) Start() supervisor.ActionResult   { return supervisor.Filled(a.service.Start()) }
func (a *App) Stop() supervisor.ActionResult    { return supervisor.Filled(a.service.Stop()) }
func (a *App) Restart() supervisor.ActionResult { return supervisor.Filled(a.service.Restart()) }
func (a *App) Prepare() supervisor.ActionResult { return supervisor.Filled(a.service.Prepare()) }
func (a *App) Cancel() supervisor.ActionResult  { return supervisor.Filled(a.service.Cancel()) }
func (a *App) Approve() supervisor.ActionResult { return supervisor.Filled(a.service.Approve()) }
func (a *App) Decline() supervisor.ActionResult { return supervisor.Filled(a.service.Decline()) }
func (a *App) DismissError() supervisor.ActionResult {
	return supervisor.Filled(a.service.DismissError())
}
func (a *App) Manage() supervisor.ActionResult { return supervisor.Filled(a.service.Manage()) }
func (a *App) SelectInstance(id string) supervisor.ActionResult {
	return supervisor.Filled(a.service.SelectInstance(id))
}
func (a *App) Diagnose() supervisor.ActionResult { return supervisor.Filled(a.service.Diagnose()) }
func (a *App) PreviewBundle() supervisor.ActionResult {
	return supervisor.Filled(a.service.PreviewBundle())
}
func (a *App) SaveBundle(path string) supervisor.ActionResult {
	return supervisor.Filled(a.service.SaveBundle(path))
}
func (a *App) Exit() supervisor.ActionResult { return supervisor.Filled(a.service.Exit()) }
func (a *App) CheckUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.CheckUpdate())
}
func (a *App) InstallUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.InstallUpdate())
}
func (a *App) RecoverUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.RecoverUpdate())
}
func (a *App) ClearReleaseHistory() supervisor.ActionResult {
	return supervisor.Filled(a.service.ClearReleaseHistory())
}
func (a *App) DownloadUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.DownloadUpdate())
}
func (a *App) CancelUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.CancelUpdate())
}
func (a *App) DismissUpdate() supervisor.ActionResult {
	return supervisor.Filled(a.service.DismissUpdate())
}
func (a *App) Download(id string) supervisor.ModelPlanResult {
	return supervisor.Filled(a.service.Download(id))
}
func (a *App) ConfirmDownload(id string) supervisor.ActionResult {
	return supervisor.Filled(a.service.ConfirmDownload(id))
}
func (a *App) CancelDownload() supervisor.ActionResult {
	return supervisor.Filled(a.service.CancelDownload())
}
func (a *App) PatchPreferences(patch supervisor.PreferencesPatch) supervisor.ActionResult {
	if refused := rejectLANPreference(patch); refused != nil {
		return supervisor.Filled(*refused)
	}
	return supervisor.Filled(a.service.PatchPreferences(patch))
}
func (a *App) SetSwitchOverrides(patch supervisor.SwitchOverridesPatch) supervisor.ActionResult {
	return supervisor.Filled(a.service.SetSwitchOverrides(patch))
}
func (a *App) Estimate(subject string, contextTokens uint64, sessions uint32) supervisor.ModelEstimateResult {
	return supervisor.Filled(a.service.Estimate(subject, contextTokens, sessions))
}
func (a *App) SetVisible(visible bool) supervisor.ActionResult {
	return supervisor.Filled(a.service.SetVisible(visible))
}
func (a *App) DismissModelJob() supervisor.ActionResult {
	return supervisor.Filled(a.service.DismissModelJob())
}

// CopyText keeps clipboard access at the native boundary, where the test build
// can capture the request without changing the owner's clipboard.
func (a *App) CopyText(value string) supervisor.ActionResult {
	if err := copyTextNative(a.ctx, value); err != nil {
		return supervisor.Filled(supervisor.ActionResult{Error: &supervisor.CLIError{Code: "clipboard_failed", Remedy: err.Error()}})
	}
	return supervisor.Filled(supervisor.ActionResult{})
}

// OpenExternal opens a web page in the user's browser. Only https:// pages are
// opened: some links come from CLI data, and the shell would also run file:
// paths and custom protocol handlers (FAILURE-LISTS W12).
func (a *App) OpenExternal(target string) supervisor.ActionResult {
	u, err := url.Parse(target)
	if err != nil || !strings.EqualFold(u.Scheme, "https") || u.Host == "" {
		return supervisor.Filled(supervisor.ActionResult{Error: &supervisor.CLIError{Code: "internal_error", Remedy: "refused a non-https link"}})
	}
	if err := openExternalNative(a.ctx, u.String()); err != nil {
		return supervisor.Filled(supervisor.ActionResult{Error: &supervisor.CLIError{Code: "internal_error", Remedy: err.Error()}})
	}
	return supervisor.Filled(supervisor.ActionResult{})
}
