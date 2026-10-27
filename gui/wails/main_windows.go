//go:build windows && !bindings

package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"bubo/gui/supervisor"
	"bubo/gui/updater"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/windows"
)

var startHidden bool
var nativeMessageBox = func(text string) {
	message, _ := syscall.UTF16PtrFromString(text)
	title, _ := syscall.UTF16PtrFromString("Bubo")
	syscall.NewLazyDLL("user32.dll").NewProc("MessageBoxW").Call(
		0, uintptr(unsafe.Pointer(message)), uintptr(unsafe.Pointer(title)), 0x10,
	)
}
var updaterConfiguration = func(installDir string) updater.Config {
	return updater.Config{InstallDir: installDir}
}

func main() {
	supervisor.SetCrashHandler(nativeMessageBox)
	defer supervisor.ReportPanic()
	// Direct Go builds do not contain the Wails CLI's Windows manifest. Set
	// process awareness before any window; this does not change display scale.
	if err := enableProcessDPI(); err != nil {
		panic(err)
	}
	cli := flag.String("cli", "", "Absolute installed bubo.exe path")
	healthToken := flag.String(strings.TrimPrefix(updater.HealthFlag, "--"), "", "Updater health token")
	healthDeadline := flag.String(strings.TrimPrefix(updater.HealthDeadlineFlag, "--"), "", "Updater health deadline")
	flag.Parse()
	lease, primary, err := supervisor.AcquireSingleInstance()
	if err != nil {
		panic(err)
	}
	if !primary {
		return
	}
	defer lease.Close()
	settingsPath := supervisor.SettingsPath()
	settings, warning := supervisor.LoadSettings(settingsPath)
	path := supervisor.DefaultCLIPath()
	if *cli != "" {
		path = *cli
	}
	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}
	installDir := supervisor.GUIPackageRoot(exe)
	health, err := supervisor.NewUpdateHealth(installDir, *healthToken, *healthDeadline)
	if err != nil {
		panic(err)
	}
	// One key manager (Windows Credential Manager) serves the supervisor and the window.
	keys := supervisor.NewKeyManager(nil)
	b, err := supervisor.NewBackend(supervisor.Config{Exe: path, Port: settings.Port, LogDir: filepath.Join(filepath.Dir(settingsPath), "logs"), SettingsPath: settingsPath, StallTimeout: settings.StallTimeout(), Keys: keys, GPU: supervisor.NewGPUSource()}, settings, exe)
	if err != nil {
		panic(err)
	}
	loginStartErr := b.RestoreLoginStart()
	a := newApp(b, settings, lease, warning)
	if loginStartErr != nil {
		a.service.SetStartupNotice("login_start_failed", loginStartErr.Error())
	}
	a.service.UseKeys(keys)
	a.service.AttachModels(settingsPath, supervisor.DefaultCLIPath(), *cli)
	var updates *supervisor.UpdateService
	updates = supervisor.NewUpdateService(supervisor.UpdateServiceConfig{
		Updater: updaterConfiguration(installDir), Snapshot: b.Snapshot,
		LogDir:         filepath.Join(filepath.Dir(settingsPath), "logs"),
		PrepareRuntime: a.service.Prepare,
		Enabled:        a.service.UpdatesEnabled,
		Quiesce:        b.QuiesceForUpdate(a.service.Models()), Events: a.emitStream, Exit: a.quitForUpdate, Health: health,
	})
	a.service.UseUpdates(updates)
	defer a.shutdown(nil)
	frontend, err := fs.Sub(assets, "frontend/dist")
	if err != nil {
		panic(err)
	}
	err = wails.Run(&options.App{
		Title: "Bubo control", Width: 900, Height: 640, MinWidth: 560, MinHeight: 360,
		StartHidden:      startHidden,
		BackgroundColour: options.NewRGB(250, 248, 241),
		AssetServer:      &assetserver.Options{Assets: frontend},
		OnStartup: func(ctx context.Context) {
			defer supervisor.ReportPanic()
			a.startup(ctx)
		},
		OnDomReady: func(ctx context.Context) {
			defer supervisor.ReportPanic()
			a.domReady(ctx)
		},
		OnBeforeClose: func(ctx context.Context) bool {
			defer supervisor.ReportPanic()
			return a.beforeClose(ctx)
		},
		OnShutdown: func(ctx context.Context) {
			defer supervisor.ReportPanic()
			a.shutdown(ctx)
		},
		Bind: []interface{}{a},
		Windows: &windows.Options{
			Theme: windows.SystemDefault, BackdropType: windows.None,
			WebviewGpuIsDisabled: true,
			WebviewUserDataPath:  filepath.Join(filepath.Dir(settingsPath), "webview2"),
		},
	})
	if err != nil {
		panic(err)
	}
}

func enableProcessDPI() error {
	user32 := syscall.NewLazyDLL("user32.dll")
	set := user32.NewProc("SetProcessDpiAwarenessContext")
	if err := set.Find(); err != nil {
		return fmt.Errorf("per-monitor DPI awareness unsupported: %w", err)
	}
	// DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4.
	set.Call(^uintptr(3))
	var awareness int32 = -1
	hr, _, _ := syscall.NewLazyDLL("shcore.dll").NewProc("GetProcessDpiAwareness").Call(0, uintptr(unsafe.Pointer(&awareness)))
	if hr != 0 || awareness != 2 {
		return fmt.Errorf("per-monitor DPI awareness failed: HRESULT %x, awareness %d", hr, awareness)
	}
	return nil
}
