//go:build windows

package supervisor

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeBackend struct {
	mu                 sync.Mutex
	view               Snapshot
	settings           Settings
	settingsPath       string
	actions            []string
	saveErr, actionErr error
	exitGate           chan struct{}
	exitCalled         chan struct{}
	visible            []bool
}

func (b *fakeBackend) Snapshot() Snapshot    { return b.view }
func (b *fakeBackend) Logs() []string        { return []string{"sample CLI log"} }
func (b *fakeBackend) Notice() string        { return "" }
func (b *fakeBackend) Diagnostic() string    { return "{}" }
func (b *fakeBackend) BundlePreview() string { return "preview" }
func (b *fakeBackend) SetVisible(value bool) {
	b.mu.Lock()
	b.visible = append(b.visible, value)
	b.mu.Unlock()
}
func (b *fakeBackend) Save(s Settings) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.saveErr != nil {
		return b.saveErr
	}
	if b.settingsPath != "" {
		if err := SaveSettings(b.settingsPath, s); err != nil {
			return err
		}
	}
	b.settings = s
	return nil
}
func (b *fakeBackend) Action(action string, s Settings, path string) <-chan ResultReply {
	b.mu.Lock()
	b.actions = append(b.actions, action)
	b.settings = s
	err := b.actionErr
	b.mu.Unlock()
	r := make(chan ResultReply, 1)
	go func() {
		if action == "exit" && b.exitGate != nil {
			if b.exitCalled != nil {
				select {
				case b.exitCalled <- struct{}{}:
				default:
				}
			}
			<-b.exitGate
		}
		r <- ResultReply{Err: err}
	}()
	return r
}

func TestGetViewKeepsCLIFieldsAndUnavailableValues(t *testing.T) {
	b := &fakeBackend{view: Snapshot{
		State: Ready, Stale: true,
		Rates:     &Rates{DecodeTPS: nil},
		LastError: &Failure{Code: "device_lost", Remedy: "Restart on request."},
	}}
	a := newTestApp(b, DefaultSettings(), nil, "")
	data, err := json.Marshal(a.GetView())
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"decode_tokens_per_second":null`, `"memory_estimate":null`, `"stale":true`, `"code":"device_lost"`} {
		if !strings.Contains(string(data), field) {
			t.Fatalf("missing exact CLI/supervisor field %s: %s", field, data)
		}
	}
}

func TestActionUsesSavedSettingsAndPropagatesFailure(t *testing.T) {
	b := &fakeBackend{actionErr: errors.New("exact CLI failure")}
	a := newTestApp(b, DefaultSettings(), nil, "")
	s := DefaultSettings()
	s.Port = 18110
	s.Context = 131072
	if err := patchTestPreferences(a, s); err != nil {
		t.Fatal(err)
	}
	if err := actionTestError(a.Approve()); err == nil || err.Error() != "exact CLI failure" {
		t.Fatalf("error lost: %v", err)
	}
	if b.settings.Port != 18110 || b.settings.Context != 131072 || len(b.actions) != 1 || b.actions[0] != "approve" {
		t.Fatal("settings or delegation changed")
	}
}

func TestFailedSaveKeepsRequestedSettings(t *testing.T) {
	b := &fakeBackend{saveErr: errors.New("disk failure")}
	initial := DefaultSettings()
	initial.Port = 19110 // explicit saved override, independent of CLI defaults
	a := newTestApp(b, initial, nil, "")
	s := DefaultSettings()
	s.Port = 18110
	if err := patchTestPreferences(a, s); err == nil {
		t.Fatal("save failure hidden")
	}
	if a.GetView().Saved.Port == nil || *a.GetView().Saved.Port != initial.Port {
		t.Fatal("failed save changed requested settings")
	}
}

func TestCloseExternalDetachesAndQuits(t *testing.T) {
	b := &fakeBackend{view: Snapshot{State: AttachedExternal}}
	a := newTestApp(b, DefaultSettings(), nil, "")
	quit := make(chan struct{}, 1)
	a.quit = func(context.Context) { quit <- struct{}{} }
	if !a.BeforeClose(context.Background()) {
		t.Fatal("external close refused")
	}
	select {
	case <-quit:
	case <-time.After(time.Second):
		t.Fatal("external detach exit missing")
	}
}

func TestTrayRegistrationFailureIsVisibleAndPreventsHide(t *testing.T) {
	b := &fakeBackend{}
	a := newTestApp(b, DefaultSettings(), nil, "")
	a.tray = &fakeTray{err: errors.New("shell refused icon")}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	if n := a.GetView().Notice; a.trayAvailable || n == nil || n.Code != "tray_unavailable" || !strings.Contains(n.Detail, "shell refused icon") {
		t.Fatal("tray registration failure hidden")
	}
	s := DefaultSettings()
	s.CloseToTray = true
	if err := patchTestPreferences(a, s); err != nil || a.settings.CloseToTray {
		t.Fatal("unavailable tray accepted")
	}
	a.Shutdown(ctx)
}

func TestTrayLifecycleUsesSameProcessHooksAndStopsOnce(t *testing.T) {
	b := &fakeBackend{view: actionTestInput(Stopped).Snapshot}
	a := newTestApp(b, DefaultSettings(), nil, "")
	var dispatch func(string)
	stopped, shown := 0, 0
	a.show = func(context.Context) { shown++ }
	ft := &fakeTray{onStop: func() { stopped++ }}
	a.tray = ft
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.Startup(ctx)
	dispatch = ft.action
	dispatch("show")
	dispatch("start")
	dispatch("stop")
	if shown != 2 || a.GetView().Window.TrayAction != "stop" || a.TakeTrayAction().Value != "stop" || a.TakeTrayAction().Value != "" {
		t.Fatal("tray Show or confirmation handoff missing")
	}
	if len(b.actions) != 1 || b.actions[0] != "start" {
		t.Fatal("tray bypassed panel stop confirmation")
	}
	dispatch("restart")
	if a.TakeTrayAction().Value != "restart" {
		t.Fatal("restart handoff missing")
	}
	a.Shutdown(ctx)
	a.Shutdown(ctx)
	if stopped != 1 {
		t.Fatal("tray stopped more than once")
	}
}

func TestSaveAcceptsAvailableTray(t *testing.T) {
	b := &fakeBackend{}
	a := newTestApp(b, DefaultSettings(), nil, "")
	s := DefaultSettings()
	s.CloseToTray = true
	if err := patchTestPreferences(a, s); err != nil || !a.GetView().Saved.CloseToTray {
		t.Fatalf("available tray setting not saved: %v", err)
	}
}

func TestTrayMenuMatchesOwnedAndExternalStates(t *testing.T) {
	for _, test := range []struct {
		state                State
		start, stop, restart bool
	}{
		{Ready, false, true, true}, {Starting, false, true, false},
		{Stopped, true, false, false}, {Failed, true, false, false},
		{AttachedExternal, false, false, false}, {NotPrepared, false, false, false},
	} {
		m := appTestTrayMenu(Snapshot{State: test.state, Prepared: true})
		enabled := map[string]bool{}
		for _, item := range m.Items {
			if item.Action != "" {
				enabled[item.Label] = item.Enabled
			}
		}
		if !enabled["Show window"] || !enabled["Quit"] || enabled["Start server"] != test.start || enabled["Stop server"] != test.stop || enabled["Restart server"] != test.restart {
			t.Fatalf("unsafe tray actions in %s: %+v", test.state, m.Items)
		}
	}
}

func TestCloseWaitsForSupervisorExitBeforeQuitting(t *testing.T) {
	b := &fakeBackend{exitGate: make(chan struct{}), exitCalled: make(chan struct{}, 1)}
	a := newTestApp(b, DefaultSettings(), nil, "")
	quit := make(chan struct{}, 1)
	a.quit = func(context.Context) { quit <- struct{}{} }
	if !a.BeforeClose(context.Background()) {
		t.Fatal("close allowed before Exit")
	}
	select {
	case <-b.exitCalled:
	case <-time.After(time.Second):
		t.Fatal("Exit not requested")
	}
	if !a.BeforeClose(context.Background()) {
		t.Fatal("repeat close allowed while Exit pending")
	}
	select {
	case <-quit:
		t.Fatal("quit before owned Job Objects finished")
	default:
	}
	if err := actionTestError(a.Start()); err == nil {
		t.Fatal("Start accepted during close")
	}
	close(b.exitGate)
	select {
	case <-quit:
	case <-time.After(time.Second):
		t.Fatal("quit missing after Exit")
	}
	if a.BeforeClose(context.Background()) {
		t.Fatal("completed Exit blocks final close")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.actions) != 1 || b.actions[0] != "exit" {
		t.Fatalf("close sent wrong actions: %v", b.actions)
	}
}

func TestCloseFailureRemainsVisibleAndDoesNotQuit(t *testing.T) {
	b := &fakeBackend{actionErr: errors.New("drain failed")}
	a := newTestApp(b, DefaultSettings(), nil, "")
	quit := make(chan struct{}, 1)
	a.quit = func(context.Context) { quit <- struct{}{} }
	if !a.BeforeClose(context.Background()) {
		t.Fatal("premature close")
	}
	deadline := time.Now().Add(time.Second)
	for n := a.GetView().Notice; n == nil || n.Code == "" || !strings.Contains(n.Detail, "drain failed"); n = a.GetView().Notice {
		if time.Now().After(deadline) {
			t.Fatal("exit error hidden")
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case <-quit:
		t.Fatal("quit despite failed stop/drain")
	default:
	}
}
