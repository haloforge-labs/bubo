//go:build windows

package main

import (
	"bubo/gui/supervisor"
	"bytes"
)

type trayItem = supervisor.TrayItem
type trayMenu = supervisor.TrayMenu

// trayOwlSVG recolours the site owl for the taskbar theme. The mark is dark
// red, which is unreadable on a dark taskbar, so dark gets a light ink.
func trayOwlSVG(svg []byte, darkTaskbar bool) []byte {
	if !darkTaskbar {
		return svg
	}
	return bytes.ReplaceAll(svg, []byte(`#a00000`), []byte(`#f3ede4`))
}
