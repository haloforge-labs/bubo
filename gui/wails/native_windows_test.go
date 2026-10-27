//go:build windows

package main

import (
	"bubo/gui/supervisor"
	"golang.org/x/sys/windows"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	goruntime "runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func TestTrayOwlAndNativeStructureLayout(t *testing.T) {
	mask, pixels, err := rasterTrayOwl(trayOwl)
	if err != nil || len(mask) != 128 || len(pixels) != 4096 {
		t.Fatalf("owl raster failed: %v", err)
	}
	nonzero := 0
	for i := 3; i < len(pixels); i += 4 {
		if pixels[i] != 0 {
			nonzero++
		}
	}
	if nonzero < 64 || nonzero > 1000 {
		t.Fatalf("missing owl alpha shape: %d", nonzero)
	}
	if unsafe.Sizeof(uintptr(0)) == 8 {
		if unsafe.Sizeof(trayIconData{}) != 976 || unsafe.Sizeof(trayWindowClass{}) != 80 || unsafe.Sizeof(trayMessage{}) != 48 {
			t.Fatal("Windows native structure layout differs")
		}
	}
}

func TestFolderNativeCommandHasNoShellOrVisibleConsole(t *testing.T) {
	path := `C:\models\spaces & symbols (test)`
	cmd, err := nativeFolderCommand(path)
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(cmd.Path) || !strings.EqualFold(filepath.Base(cmd.Path), "explorer.exe") || len(cmd.Args) != 2 || cmd.Args[1] != path {
		t.Fatal("folder request interpreted as command text", cmd.Path, cmd.Args)
	}
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow || cmd.SysProcAttr.CreationFlags&windows.CREATE_NO_WINDOW == 0 {
		t.Fatal("folder helper can create a visible console")
	}
	// Do not start this command: all action tests use a fake FolderOpener.
}

func TestFrontendEmbedIncludesBuildPlaceholder(t *testing.T) {
	b, err := assets.ReadFile("frontend/dist/.gitkeep")
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 0 {
		t.Fatal("build placeholder must stay empty")
	}
}

func TestTrayOwlFollowsTaskbarTheme(t *testing.T) {
	light, dark := trayOwlSVG(trayOwl, false), trayOwlSVG(trayOwl, true)
	if !strings.Contains(string(light), "#a00000") || strings.Contains(string(dark), "#a00000") || !strings.Contains(string(dark), "#f3ede4") {
		t.Fatal("theme recolour wrong")
	}
	_, p1, err1 := rasterTrayOwl(light)
	_, p2, err2 := rasterTrayOwl(dark)
	if err1 != nil || err2 != nil || string(p1) == string(p2) {
		t.Fatal("light and dark icons are the same or failed")
	}
}

func TestBindingsGenerationIgnoresHeldInstanceLease(t *testing.T) {
	// Hold the same named mutex as a running GUI without opening its activation
	// pipe (which would send real input to the user's window).
	sid, err := supervisor.UserSID()
	if err != nil {
		t.Fatal(err)
	}
	name, _ := windows.UTF16PtrFromString("Local\\Bubo.Control." + sid)
	goruntime.LockOSThread()
	defer goruntime.UnlockOSThread()
	h, err := windows.CreateMutex(nil, false, name)
	if err != nil && err != windows.ERROR_ALREADY_EXISTS {
		t.Fatal(err)
	}
	defer windows.CloseHandle(h)
	state, err := windows.WaitForSingleObject(h, 0)
	if err != nil {
		t.Fatal(err)
	}
	if state == windows.WAIT_OBJECT_0 || state == windows.WAIT_ABANDONED {
		defer windows.ReleaseMutex(h)
	} else if state != uint32(windows.WAIT_TIMEOUT) {
		t.Fatalf("mutex state %d", state)
	}
	wails, err := exec.LookPath("wails")
	if err != nil {
		wails = filepath.Join(os.Getenv("USERPROFILE"), "go", "bin", "wails.exe")
		if _, e := os.Stat(wails); e != nil {
			t.Fatal("Install github.com/wailsapp/wails/v2/cmd/wails@v2.16.0:", e)
		}
	}
	cmd := exec.Command(wails, "generate", "module", "-nocolour")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("bindings with held lease: %v\n%s", err, out)
	}
	b, err := os.ReadFile("frontend/wailsjs/go/models.ts")
	if err != nil {
		t.Fatal(err)
	}
	s := strings.SplitN(string(b), "export namespace supervisor", 2)
	if len(s) != 2 {
		t.Fatal("supervisor namespace missing")
	}
	view := regexp.MustCompile(`(?s)export class View \{(.*?)static createFrom`).FindStringSubmatch(s[1])
	if len(view) != 2 {
		t.Fatal("View declaration missing")
	}
	if strings.Contains(view[1], "any") {
		t.Fatal("untyped View field", view[1])
	}
	t.Log("View field types:", strings.TrimSpace(view[1]))
}
