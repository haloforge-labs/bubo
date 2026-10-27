//go:build windows

package supervisor

import (
	"net/url"
)

// The tray menu is a pure function of the supervisor snapshot. It holds no
// lifecycle or CLI logic: every item names an action that the App already
// routes to the same backend path as the window.

// TrayItem is one menu row. Separator rows carry no label or action.
type TrayItem struct {
	ID                 uintptr
	Label, Action      string
	Enabled, Separator bool
}

type TrayMenu struct {
	Tip   string
	Items []TrayItem
}

// TrayPlatform is the only native boundary. The Windows adapter owns its own
// OS thread and message loop; tests supply a fake and never create a tray.
type TrayPlatform interface {
	// Start registers the icon. action receives "show", "start", "stop",
	// "restart" or "quit". menu is read each time the menu opens.
	Start(action func(string), menu func() TrayMenu) (stop func(), err error)
}

func trayEndpoint(s Snapshot) string {
	if s.Instance == nil || s.Instance.Endpoint == nil {
		return ""
	}
	u, err := url.Parse(*s.Instance.Endpoint)
	if err != nil || u.Host == "" {
		return ""
	}
	return u.Host
}

// trayStatus uses the View's native status label and the observed host.
func trayStatus(s Snapshot) string {
	return "Bubo · " + s.Label + " · " + trayEndpoint(s)
}

func buildTrayMenu(s Snapshot, actions LifecycleActions) TrayMenu {
	running := actions.Stop.Allowed || actions.Stop.ReasonCode == "external_read_only"
	canStart := actions.Start.Allowed
	toggle := TrayItem{ID: 3, Label: "Start server", Action: "start", Enabled: canStart}
	if running {
		toggle = TrayItem{ID: 3, Label: "Stop server", Action: "stop", Enabled: actions.Stop.Allowed}
	}
	status := trayStatus(s)
	return TrayMenu{
		Tip: status,
		Items: []TrayItem{
			{ID: 1, Label: status},
			{Separator: true},
			{ID: 2, Label: "Show window", Action: "show", Enabled: true},
			toggle,
			{ID: 4, Label: "Restart server", Action: "restart", Enabled: actions.Restart.Allowed},
			{Separator: true},
			{ID: 5, Label: "Quit", Action: "quit", Enabled: true},
		},
	}
}
