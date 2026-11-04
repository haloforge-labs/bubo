//go:build windows

package main

import "testing"

func TestRefuseDuringUpdate(t *testing.T) {
	defer func(p func(string) bool, n func(string, string)) { updateInProgress, showNotice = p, n }(updateInProgress, showNotice)
	var shown []string
	showNotice = func(title, text string) { shown = append(shown, title+": "+text) }
	busy := true
	updateInProgress = func(dir string) bool { return busy && dir == `C:\Bubo` }

	if !refuseDuringUpdate("", `C:\Bubo`) || len(shown) != 1 || shown[0] != "Bubo: "+updateInProgressText {
		t.Fatalf("a hand-started GUI during an update must be refused with the notice: %v", shown)
	}
	if refuseDuringUpdate("health-token", `C:\Bubo`) || len(shown) != 1 {
		t.Fatalf("the update's own new GUI must never be refused: %v", shown)
	}
	if refuseDuringUpdate("", `C:\Other`) || len(shown) != 1 {
		t.Fatalf("another installation is not refused: %v", shown)
	}
	busy = false
	if refuseDuringUpdate("", `C:\Bubo`) || len(shown) != 1 {
		t.Fatalf("no update in progress: %v", shown)
	}
}
