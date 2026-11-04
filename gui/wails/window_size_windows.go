//go:build windows && !bindings

package main

import (
	"syscall"
	"unsafe"
)

// The window opens large enough for two columns of settings at once. The size is
// the one the user left it at (remembered at close); a first start uses the
// default. Either way it never exceeds the work area of the screen.
const (
	defaultWindowWidth  = 1280
	defaultWindowHeight = 860
	minWindowWidth      = 560
	minWindowHeight     = 360
)

// clampWindowSize returns the size to open with: the saved size when there is
// one, else the default, no smaller than the minimum and no larger than the
// work area (all in the same logical pixels).
func clampWindowSize(savedWidth, savedHeight, workWidth, workHeight int) (int, int) {
	width, height := defaultWindowWidth, defaultWindowHeight
	if savedWidth > 0 && savedHeight > 0 {
		width, height = savedWidth, savedHeight
	}
	if workWidth > 0 && width > workWidth {
		width = workWidth
	}
	if workHeight > 0 && height > workHeight {
		height = workHeight
	}
	if width < minWindowWidth {
		width = minWindowWidth
	}
	if height < minWindowHeight {
		height = minWindowHeight
	}
	return width, height
}

// workAreaLogical is the primary screen's work area in logical pixels (the
// window sizes of Wails are logical; the process is DPI aware, so the system
// reports physical pixels).
var workAreaLogical = func() (int, int) {
	var rect struct{ Left, Top, Right, Bottom int32 }
	user32 := syscall.NewLazyDLL("user32.dll")
	const spiGetWorkArea = 0x0030
	if ok, _, _ := user32.NewProc("SystemParametersInfoW").Call(spiGetWorkArea, 0, uintptr(unsafe.Pointer(&rect)), 0); ok == 0 {
		return 0, 0
	}
	dpi, _, _ := user32.NewProc("GetDpiForSystem").Call()
	if dpi == 0 {
		dpi = 96
	}
	scale := func(physical int32) int { return int(physical) * 96 / int(dpi) }
	return scale(rect.Right - rect.Left), scale(rect.Bottom - rect.Top)
}
