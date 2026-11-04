//go:build windows

package main

import (
	"golang.org/x/sys/windows"

	"bubo/gui/updater"
)

// A GUI started by hand while an update is being installed would take the
// single-instance slot that the update's own new GUI needs to acknowledge health,
// and the update would be rolled back. The user is told to wait instead. The new
// GUI that the update helper starts carries the health token and is never refused.
var (
	updateInProgress = updater.InstallInProgress
	showNotice       = func(title, text string) {
		t, _ := windows.UTF16PtrFromString(title)
		m, _ := windows.UTF16PtrFromString(text)
		_, _ = windows.MessageBox(0, m, t, windows.MB_OK|windows.MB_ICONINFORMATION)
	}
)

const updateInProgressText = "Bubo is installing an update. Open Bubo again in a minute."

func refuseDuringUpdate(healthToken, installDir string) bool {
	if healthToken != "" || !updateInProgress(installDir) {
		return false
	}
	showNotice("Bubo", updateInProgressText)
	return true
}
