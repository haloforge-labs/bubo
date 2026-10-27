//go:build windows

package supervisor

import (
	"context"
	"testing"
	"time"
)

func TestExitBindingQuitsWithoutStoppingExternalServerDirectly(t *testing.T) {
	b := &fakeBackend{view: Snapshot{State: AttachedExternal}}
	a := newTestApp(b, DefaultSettings(), nil, "")
	quit := make(chan struct{})
	a.quit = func(context.Context) { close(quit) }
	if result := a.Exit(); result.Error != nil {
		t.Fatal(result)
	}
	select {
	case <-quit:
	case <-time.After(time.Second):
		t.Fatal("Exit did not finish the native shutdown")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.actions) != 1 || b.actions[0] != "exit" {
		t.Fatal("Exit bypassed supervisor shutdown", b.actions)
	}
}

func TestSetVisibleBindingUsesNativeHooksAndStatusVisibility(t *testing.T) {
	b := &fakeBackend{}
	a := newTestApp(b, DefaultSettings(), nil, "")
	shown, hidden := 0, 0
	a.show = func(context.Context) { shown++ }
	a.hide = func(context.Context) { hidden++ }
	if result := a.SetVisible(true); result.Error != nil {
		t.Fatal(result.Error)
	}
	if result := a.SetVisible(false); result.Error != nil {
		t.Fatal(result.Error)
	}
	if shown != 1 || hidden != 1 || len(b.visible) != 2 || !b.visible[0] || b.visible[1] {
		t.Fatal("native and supervisor visibility diverged", shown, hidden, b.visible)
	}
}
