//go:build windows

package main

import (
	"bytes"
	"fmt"
	"image"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"github.com/fyne-io/oksvg"
	"github.com/srwiley/rasterx"
	"golang.org/x/sys/windows/registry"
)

// The tray lives on a native message thread in this same process. Its message
// window has no visible style and is never shown. No separate tray executable,
// HTTP listener, keyboard hook or input synthesis is used.
var (
	trayUser32   = syscall.NewLazyDLL("user32.dll")
	trayShell32  = syscall.NewLazyDLL("shell32.dll")
	trayKernel32 = syscall.NewLazyDLL("kernel32.dll")
)

const (
	trayCallback = 0x8001 // WM_APP + 1
	trayClose    = 0x8002
	trayIconID   = 1
)

type trayWindowClass struct {
	Size, Style                        uint32
	Proc                               uintptr
	ClassExtra, WindowExtra            int32
	Instance, Icon, Cursor, Background uintptr
	MenuName, ClassName                *uint16
	SmallIcon                          uintptr
}

type trayPoint struct{ X, Y int32 }
type trayMessage struct {
	Window         uintptr
	Message        uint32
	WParam, LParam uintptr
	Time           uint32
	Point          trayPoint
	Private        uint32
}

// NOTIFYICONDATAW uses the full modern structure, including the union's four
// bytes, GUID and balloon icon. Native pointer alignment is supplied by Go.
type trayIconData struct {
	Size                uint32
	Window              uintptr
	ID, Flags, Callback uint32
	Icon                uintptr
	Tip                 [128]uint16
	State, StateMask    uint32
	Info                [256]uint16
	Version             uint32
	InfoTitle           [64]uint16
	InfoFlags           uint32
	GUID                [16]byte
	BalloonIcon         uintptr
}

// rasterTrayOwl reuses the exact site owl; only pixel storage is changed to
// Windows' BGRA icon format. Rendering here creates no application window.
func rasterTrayOwl(svg []byte) (andMask, bgra []byte, err error) {
	const size = 32
	icon, err := oksvg.ReadIconStream(bytes.NewReader(svg))
	if err != nil {
		return nil, nil, err
	}
	bitmap := image.NewNRGBA(image.Rect(0, 0, size, size))
	icon.SetTarget(0, 0, size, size)
	icon.Draw(rasterx.NewDasher(size, size, rasterx.NewScannerGV(size, size, bitmap, bitmap.Bounds())), 1)
	andMask, bgra = make([]byte, size*size/8), make([]byte, size*size*4)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			pixel := bitmap.NRGBAAt(x, y)
			i := (y*size + x) * 4
			// Native 32-bit icons use premultiplied BGRA.
			bgra[i], bgra[i+1], bgra[i+2], bgra[i+3] = byte(uint16(pixel.B)*uint16(pixel.A)/255), byte(uint16(pixel.G)*uint16(pixel.A)/255), byte(uint16(pixel.R)*uint16(pixel.A)/255), pixel.A
			if pixel.A == 0 {
				andMask[y*4+x/8] |= 0x80 >> uint(x%8)
			}
		}
	}
	return andMask, bgra, nil
}

// nativeTray is the Shell_NotifyIcon adapter.
type nativeTray struct{}

// taskbarIsDark reads the per-user theme value (read only). Light is the
// default when the value is missing.
func taskbarIsDark() bool {
	key, err := registry.OpenKey(registry.CURRENT_USER, `Software\Microsoft\Windows\CurrentVersion\Themes\Personalize`, registry.QUERY_VALUE)
	if err != nil {
		return false
	}
	defer key.Close()
	v, _, err := key.GetIntegerValue("SystemUsesLightTheme")
	return err == nil && v == 0
}

func (nativeTray) Start(action func(string), menu func() trayMenu) (func(), error) {
	return startNativeTray(action, menu)
}

func startNativeTray(action func(string), menu func() trayMenu) (func(), error) {
	ready := make(chan struct {
		stop func()
		err  error
	}, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		fail := func(err error) {
			ready <- struct {
				stop func()
				err  error
			}{err: err}
		}
		instance, _, _ := trayKernel32.NewProc("GetModuleHandleW").Call(0)
		className, _ := syscall.UTF16PtrFromString("BuboWailsTray")
		var window, nativeIcon uintptr
		darkNow := taskbarIsDark()
		shellNotify := trayShell32.NewProc("Shell_NotifyIconW")
		data := trayIconData{ID: trayIconID, Flags: 1 | 2 | 4, Callback: trayCallback}
		data.Size = uint32(unsafe.Sizeof(data))
		copy(data.Tip[:], syscall.StringToUTF16("Bubo server control"))
		var destroyOnce sync.Once
		destroy := func() {
			destroyOnce.Do(func() {
				shellNotify.Call(2, uintptr(unsafe.Pointer(&data))) // NIM_DELETE
				if nativeIcon != 0 {
					trayUser32.NewProc("DestroyIcon").Call(nativeIcon)
				}
			})
		}
		taskbarCreated, _, _ := trayUser32.NewProc("RegisterWindowMessageW").Call(uintptr(unsafe.Pointer(mustUTF16("TaskbarCreated"))))
		proc := syscall.NewCallback(func(hwnd uintptr, message uint32, wp, lp uintptr) uintptr {
			switch message {
			case trayCallback:
				switch uint32(lp) {
				case 0x0202: // WM_LBUTTONUP: left-click shows the window
					go action("show")
				case 0x0205, 0x007B: // WM_RBUTTONUP / WM_CONTEXTMENU
					showTrayMenu(hwnd, menu().Items, action)
				}
				return 0
			case 0x001A: // WM_SETTINGCHANGE: follow the taskbar theme
				if text := wideString(lp); text == "ImmersiveColorSet" {
					if dark := taskbarIsDark(); dark != darkNow {
						darkNow = dark
						if icon, err := makeOwlIcon(instance, dark); err == nil {
							old := nativeIcon
							nativeIcon, data.Icon = icon, icon
							shellNotify.Call(1, uintptr(unsafe.Pointer(&data))) // NIM_MODIFY
							if old != 0 {
								trayUser32.NewProc("DestroyIcon").Call(old)
							}
						}
					}
				}
				return 0
			case trayClose:
				destroy()
				trayUser32.NewProc("DestroyWindow").Call(hwnd)
				trayUser32.NewProc("PostQuitMessage").Call(0)
				return 0
			default:
				if taskbarCreated != 0 && message == uint32(taskbarCreated) {
					shellNotify.Call(0, uintptr(unsafe.Pointer(&data)))
					return 0
				}
			}
			result, _, _ := trayUser32.NewProc("DefWindowProcW").Call(hwnd, uintptr(message), wp, lp)
			return result
		})
		class := trayWindowClass{Size: uint32(unsafe.Sizeof(trayWindowClass{})), Proc: proc, Instance: instance, ClassName: className}
		atom, _, err := trayUser32.NewProc("RegisterClassExW").Call(uintptr(unsafe.Pointer(&class)))
		if atom == 0 {
			fail(fmt.Errorf("register tray message class: %w", err))
			return
		}
		defer trayUser32.NewProc("UnregisterClassW").Call(uintptr(unsafe.Pointer(className)), instance)
		// Parent and style are zero: an invisible top-level window receives the
		// Explorer restart broadcast, unlike a message-only window.
		window, _, err = trayUser32.NewProc("CreateWindowExW").Call(0, uintptr(unsafe.Pointer(className)), uintptr(unsafe.Pointer(className)), 0, 0, 0, 0, 0, 0, 0, instance, 0)
		if window == 0 {
			fail(fmt.Errorf("create tray message window: %w", err))
			return
		}
		defer trayUser32.NewProc("DestroyWindow").Call(window)
		data.Window = window
		nativeIcon, err = makeOwlIcon(instance, darkNow)
		if err != nil {
			fail(err)
			return
		}
		data.Icon = nativeIcon
		defer destroy()
		added, _, err := shellNotify.Call(0, uintptr(unsafe.Pointer(&data))) // NIM_ADD
		if added == 0 {
			fail(fmt.Errorf("register system tray: %w", err))
			return
		}
		var stopOnce sync.Once
		ready <- struct {
			stop func()
			err  error
		}{stop: func() {
			stopOnce.Do(func() { trayUser32.NewProc("PostMessageW").Call(window, trayClose, 0, 0) })
		}}
		var msg trayMessage
		for {
			result, _, _ := trayUser32.NewProc("GetMessageW").Call(uintptr(unsafe.Pointer(&msg)), 0, 0, 0)
			if int32(result) <= 0 {
				return
			}
			trayUser32.NewProc("TranslateMessage").Call(uintptr(unsafe.Pointer(&msg)))
			trayUser32.NewProc("DispatchMessageW").Call(uintptr(unsafe.Pointer(&msg)))
		}
	}()
	result := <-ready
	return result.stop, result.err
}

// makeOwlIcon rasterises the site owl for the current taskbar theme.
func makeOwlIcon(instance uintptr, dark bool) (uintptr, error) {
	mask, pixels, err := rasterTrayOwl(trayOwlSVG(trayOwl, dark))
	if err != nil {
		return 0, err
	}
	icon, _, err := trayUser32.NewProc("CreateIcon").Call(instance, 32, 32, 1, 32, uintptr(unsafe.Pointer(&mask[0])), uintptr(unsafe.Pointer(&pixels[0])))
	if icon == 0 {
		return 0, fmt.Errorf("create owl tray icon: %w", err)
	}
	return icon, nil
}

// wideString reads the NUL-terminated UTF-16 text a WM_SETTINGCHANGE lParam
// points to. The system owns the memory; a zero pointer gives "".
func wideString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var units []uint16
	for i := 0; i < 64; i++ {
		at := ptr + uintptr(i)*2
		u := *(*uint16)(*(*unsafe.Pointer)(unsafe.Pointer(&at)))
		if u == 0 {
			break
		}
		units = append(units, u)
	}
	return syscall.UTF16ToString(units)
}

func mustUTF16(text string) *uint16 {
	value, err := syscall.UTF16PtrFromString(text)
	if err != nil {
		panic(err)
	}
	return value
}

func showTrayMenu(window uintptr, entries []trayItem, action func(string)) {
	menu, _, _ := trayUser32.NewProc("CreatePopupMenu").Call()
	if menu == 0 {
		return
	}
	defer trayUser32.NewProc("DestroyMenu").Call(menu)
	for _, entry := range entries {
		flags := uintptr(0)
		if entry.Separator {
			trayUser32.NewProc("AppendMenuW").Call(menu, 0x800, 0, 0) // MF_SEPARATOR
			continue
		}
		if !entry.Enabled {
			flags = 1
		} // MF_GRAYED
		trayUser32.NewProc("AppendMenuW").Call(menu, flags, entry.ID, uintptr(unsafe.Pointer(mustUTF16(entry.Label))))
	}
	var position trayPoint
	trayUser32.NewProc("GetCursorPos").Call(uintptr(unsafe.Pointer(&position))) // read only
	// Native menu input is supplied by the person. No focus-stealing call or
	// synthetic input is used; opening the control window requires Show.
	choice, _, _ := trayUser32.NewProc("TrackPopupMenu").Call(menu, 0x100|0x2, uintptr(position.X), uintptr(position.Y), 0, window, 0)
	trayUser32.NewProc("PostMessageW").Call(window, 0, 0, 0)
	for _, entry := range entries {
		if entry.ID == choice && entry.Enabled && entry.Action != "" {
			go action(entry.Action)
			return
		}
	}
}
