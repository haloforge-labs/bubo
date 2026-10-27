//go:build windows && bubo_test && !bindings

package main

import (
	"bubo/gui/supervisor"
	"bubo/gui/updater"
	"context"
	"crypto/x509"
	"encoding/json"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

var captureMu sync.Mutex
var trayActionMu sync.Mutex
var testTrayAction func(string)

func captureNative(kind string, value string) error {
	captureMu.Lock()
	defer captureMu.Unlock()
	record, err := json.Marshal(struct {
		Kind  string `json:"kind"`
		Value string `json:"value"`
	}{kind, value})
	if err != nil {
		return err
	}
	f, err := os.OpenFile(supervisor.TestConfiguration().CaptureFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(record, '\n'))
	return err
}

func mustCaptureNative(kind, value string) {
	if err := captureNative(kind, value); err != nil {
		panic(err)
	}
}

type capturedDialogs struct{}

func (capturedDialogs) OpenFile(_ context.Context, title string) (string, error) {
	if err := captureNative("dialog_file", title); err != nil {
		return "", err
	}
	return supervisor.TestConfiguration().DialogFile, nil
}
func (capturedDialogs) OpenFolder(_ context.Context, title string) (string, error) {
	if err := captureNative("dialog_folder", title); err != nil {
		return "", err
	}
	return supervisor.TestConfiguration().DialogFolder, nil
}

type capturedFolders struct{}

func (capturedFolders) OpenFolder(_ context.Context, path string) error {
	return captureNative("open_folder", path)
}

type capturedTray struct{}

func (capturedTray) Start(action func(string), menu func() supervisor.TrayMenu) (func(), error) {
	trayActionMu.Lock()
	testTrayAction = action
	trayActionMu.Unlock()
	initial := menu()
	if err := captureNative("tray_start", initial.Tip); err != nil {
		return nil, err
	}
	last, err := json.Marshal(initial)
	if err != nil {
		return nil, err
	}
	if err := captureNative("tray_menu", string(last)); err != nil {
		return nil, err
	}
	stop := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		tick := time.NewTicker(50 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				current, err := json.Marshal(menu())
				if err != nil {
					panic(err)
				}
				if string(current) != string(last) {
					mustCaptureNative("tray_menu", string(current))
					last = current
				}
			}
		}
	}()
	return func() {
		close(stop)
		<-stopped
		trayActionMu.Lock()
		testTrayAction = nil
		trayActionMu.Unlock()
		mustCaptureNative("tray_stop", "")
	}, nil
}

// TestTrayAction sends a captured menu action through the same callback as
// the native tray. The test runner checks that the current menu enables it.
func (a *App) TestTrayAction(action string) {
	trayActionMu.Lock()
	callback := testTrayAction
	trayActionMu.Unlock()
	if callback == nil {
		panic("test tray callback is unavailable")
	}
	mustCaptureNative("tray_dispatch", action)
	callback(action)
}

// TestPanic is bound only in the bubo_test exe for the native crash gate.
func (a *App) TestPanic() {
	supervisor.RunProtected(func() { panic("bubo_test panic trigger") })
}

func init() {
	c := supervisor.TestConfiguration()
	nativeMessageBox = func(text string) {
		captureMu.Lock()
		defer captureMu.Unlock()
		record, err := json.Marshal(struct {
			Kind string `json:"kind"`
			Text string `json:"text"`
		}{"message_box", text})
		if err != nil {
			panic(err)
		}
		f, err := os.OpenFile(c.CaptureFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		if _, err := f.Write(append(record, '\n')); err != nil {
			panic(err)
		}
	}
	// Record presence only: the launcher must remove inherited runtime paths.
	mustCaptureNative("inherited_runtime_present", strconv.FormatBool(os.Getenv("BUBO_RUNTIME_DIR") != ""))
	startHidden = true
	updaterConfiguration = func(installDir string) updater.Config {
		pem, err := os.ReadFile(c.RootCAFile)
		if err != nil {
			panic(err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			panic("bubo_test CA file has no certificate")
		}
		u, err := url.Parse(c.ManifestURL)
		if err != nil {
			panic(err)
		}
		return updater.Config{InstallDir: installDir, ManifestURL: c.ManifestURL, AllowedHosts: []string{u.Hostname()}, RootCAs: pool}
	}
	copyTextNative = func(_ context.Context, value string) error {
		return captureNative("copy", value)
	}
	openExternalNative = func(_ context.Context, value string) error {
		return captureNative("open_external", value)
	}
	appHooks = func() supervisor.AppHooks {
		return supervisor.AppHooks{
			Quit: func(ctx context.Context) { mustCaptureNative("quit", ""); runtime.Quit(ctx) },
			Show: func(context.Context) { mustCaptureNative("show", "") },
			Hide: func(context.Context) { mustCaptureNative("hide", "") },
			Tray: capturedTray{}, Dialogs: capturedDialogs{}, Folders: capturedFolders{}, Events: emitWailsStream,
		}
	}
}
