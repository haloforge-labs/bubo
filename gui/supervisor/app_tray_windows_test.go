//go:build windows

package supervisor

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// fakeTray stands in for the native adapter. No icon, window or message loop.
type fakeTray struct {
	err    error
	onStop func()
	action func(string)
	menu   func() TrayMenu
	starts int
}

func (f *fakeTray) Start(action func(string), menu func() TrayMenu) (func(), error) {
	f.starts++
	f.action, f.menu = action, menu
	if f.err != nil {
		return nil, f.err
	}
	return func() {
		if f.onStop != nil {
			f.onStop()
		}
	}, nil
}

func appTestEndpoint(text string) *string { return &text }

func TestTrayMenuModelPerState(t *testing.T) {
	type row struct {
		snap                Snapshot
		toggle              string
		toggleOn, restartOn bool
	}
	ready := &Instance{Endpoint: appTestEndpoint("http://127.0.0.1:18110")}
	for name, r := range map[string]row{
		"Not prepared": {Snapshot{State: NotPrepared, Stale: true}, "Start server", false, false},
		"Preparing":    {Snapshot{State: Preparing}, "Start server", false, false},
		"Starting":     {Snapshot{State: Starting, Instance: ready}, "Stop server", true, false},
		"Ready":        {Snapshot{State: Ready, Instance: ready}, "Stop server", true, true},
		"Stopping":     {Snapshot{State: Stopping}, "Start server", false, false},
		"Stopped":      {Snapshot{State: Stopped, Prepared: true}, "Start server", true, false},
		"Stopped bare": {Snapshot{State: Stopped}, "Start server", false, false},
		"Failed":       {Snapshot{State: Failed, Prepared: true}, "Start server", true, false},
		"External":     {Snapshot{State: AttachedExternal, Instance: ready}, "Stop server", false, false},
	} {
		t.Run(name, func(t *testing.T) {
			m := appTestTrayMenu(r.snap)
			var labels []string
			for _, item := range m.Items {
				labels = append(labels, item.Label)
			}
			if m.Items[0].Action != "" || m.Items[0].Enabled {
				t.Fatalf("status line wrong: %+v", m.Items[0])
			}
			if !m.Items[1].Separator || !m.Items[5].Separator {
				t.Fatalf("separators missing: %v", labels)
			}
			show, toggle, restart, quit := m.Items[2], m.Items[3], m.Items[4], m.Items[6]
			if show.Label != "Show window" || show.Action != "show" || !show.Enabled || quit.Label != "Quit" || quit.Action != "quit" || !quit.Enabled {
				t.Fatalf("Show or Quit wrong: %v", labels)
			}
			if toggle.Label != r.toggle || toggle.Enabled != r.toggleOn || restart.Label != "Restart server" || restart.Action != "restart" || restart.Enabled != r.restartOn {
				t.Fatalf("toggle or restart wrong: %+v %+v", toggle, restart)
			}
			wantAction := map[string]string{"Start server": "start", "Stop server": "stop"}[r.toggle]
			if toggle.Action != wantAction {
				t.Fatalf("toggle action %q", toggle.Action)
			}
		})
	}
}

func TestTrayManagedExternalCanStopButUnmanagedCannot(t *testing.T) {
	managed := appTestTrayMenu(Snapshot{State: AttachedExternal, Managed: true})
	if managed.Items[3].Label != "Stop server" || !managed.Items[3].Enabled || !managed.Items[4].Enabled {
		t.Fatalf("managed server not controllable: %+v", managed.Items)
	}
	plain := appTestTrayMenu(Snapshot{State: AttachedExternal})
	if plain.Items[3].Enabled || plain.Items[4].Enabled {
		t.Fatalf("read-only server controllable: %+v", plain.Items)
	}
}

func TestTrayClicksCallBackendOnce(t *testing.T) {
	b := &fakeBackend{view: actionTestInput(Stopped).Snapshot}
	a := newTestApp(b, DefaultSettings(), nil, "")
	ft := &fakeTray{}
	a.tray = ft
	shown := 0
	a.show = func(context.Context) { shown++ }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	ft.action("start")
	ft.action("show")
	ft.action("stop")
	ft.action("restart")
	b.mu.Lock()
	actions := append([]string(nil), b.actions...)
	b.mu.Unlock()
	if len(actions) != 1 || actions[0] != "start" {
		t.Fatalf("backend calls %v; only Start goes straight through, Stop and Restart wait for the window confirmation", actions)
	}
	if shown != 3 {
		t.Fatalf("window shown %d times, want 3 (show, stop, restart)", shown)
	}
	if ft.starts != 1 {
		t.Fatal("tray registered more than once")
	}
	a.Shutdown(ctx)
}

func TestTrayQuitExternalLeavesServer(t *testing.T) {
	// External: exits at once through the detach path, no stop.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b2 := &fakeBackend{view: Snapshot{State: AttachedExternal}}
	a2 := newTestApp(b2, DefaultSettings(), nil, "")
	ft2 := &fakeTray{}
	a2.tray = ft2
	a2.show = func(context.Context) {}
	quit := make(chan struct{}, 1)
	a2.quit = func(context.Context) { quit <- struct{}{} }
	a2.Startup(ctx)
	ft2.action("quit")
	select {
	case <-quit:
	case <-time.After(time.Second):
		t.Fatal("external: Quit did not exit")
	}
	b2.mu.Lock()
	defer b2.mu.Unlock()
	for _, act := range b2.actions {
		if act == "stop" || act == "restart" {
			t.Fatalf("external server stopped: %v", b2.actions)
		}
	}
}

func TestCloseToTrayOffByDefaultAndHidesWhenOn(t *testing.T) {
	if DefaultSettings().CloseToTray {
		t.Fatal("close-to-tray must be off by default")
	}
	// Off: closing a running owned server never hides.
	b := &fakeBackend{view: Snapshot{State: Ready}}
	a := newTestApp(b, DefaultSettings(), nil, "")
	a.hide = func(context.Context) { t.Error("hid with close-to-tray off") }
	if !a.BeforeClose(context.Background()) {
		t.Fatal("off: close refused")
	}
	// On: window hides, nothing stopped, process stays.
	for _, state := range []State{Ready, Stopped, AttachedExternal} {
		b := &fakeBackend{view: Snapshot{State: state}}
		s := DefaultSettings()
		s.CloseToTray = true
		a := newTestApp(b, s, nil, "")
		hidden := 0
		a.hide = func(context.Context) { hidden++ }
		a.quit = func(context.Context) { t.Error("quit with close-to-tray on") }
		if !a.BeforeClose(context.Background()) || hidden != 1 {
			t.Fatalf("%s: close did not hide", state)
		}
		if len(b.actions) != 0 || len(b.visible) != 1 || b.visible[0] {
			t.Fatalf("%s: hide changed server or visibility wrongly: %v %v", state, b.actions, b.visible)
		}
	}
}

func TestCloseToTrayOnThenTrayQuitStillExits(t *testing.T) {
	b := &fakeBackend{view: Snapshot{State: Ready}, exitGate: make(chan struct{}), exitCalled: make(chan struct{}, 1)}
	s := DefaultSettings()
	s.CloseToTray = true
	a := newTestApp(b, s, nil, "")
	ft := &fakeTray{}
	a.tray = ft
	a.show = func(context.Context) {}
	a.hide = func(context.Context) {}
	quit := make(chan struct{}, 1)
	a.quit = func(context.Context) { quit <- struct{}{} }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	ft.action("quit")
	<-b.exitCalled
	close(b.exitGate)
	select {
	case <-quit:
	case <-time.After(time.Second):
		t.Fatal("no quit after tray Quit")
	}
	// The final window close after Exit must not be turned into a hide.
	if a.BeforeClose(context.Background()) {
		t.Fatal("close-to-tray blocked the final close")
	}
}

func TestTrayUnavailableDegradesToWindowOnly(t *testing.T) {
	b := &fakeBackend{view: Snapshot{State: Ready}}
	s := DefaultSettings()
	s.CloseToTray = true // saved earlier, tray now missing (for example no Explorer)
	a := newTestApp(b, s, nil, "")
	a.tray = &fakeTray{err: errors.New("the taskbar is not running")}
	a.hide = func(context.Context) { t.Error("hid without a tray") }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	note := a.GetView().Notice
	if note == nil || note.Code != "tray_unavailable" || !strings.Contains(note.Detail, "the taskbar is not running") || strings.Contains(note.Detail, "\n") {
		t.Fatalf("note missing or not one line: %+v", note)
	}
	if !a.BeforeClose(context.Background()) {
		t.Fatal("close with no tray must quit, not hide")
	}
	a.Shutdown(ctx)
}

func TestSecondLaunchShowsWindow(t *testing.T) {
	b := &fakeBackend{view: Snapshot{State: Ready}}
	signal := make(chan struct{}, 1)
	lease := &SingleInstance{ShowWindow: signal}
	a := newTestApp(b, DefaultSettings(), lease, "")
	a.tray = &fakeTray{}
	shown := make(chan struct{}, 1)
	a.show = func(context.Context) { shown <- struct{}{} }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	signal <- struct{}{}
	select {
	case <-shown:
	case <-time.After(time.Second):
		t.Fatal("second launch did not show the window")
	}
	deadline := time.Now().Add(time.Second)
	for {
		b.mu.Lock()
		n := len(b.visible)
		last := n > 0 && b.visible[n-1]
		b.mu.Unlock()
		if last && n >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("visibility not restored")
		}
		time.Sleep(time.Millisecond)
	}
	a.Shutdown(ctx)
}
