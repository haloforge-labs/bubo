//go:build windows

package supervisor

import (
	"context"
	"testing"
)

// The updater's helper starts the new GUI with a hidden-window start request,
// which the system spends on the first ShowWindow. The window of an update launch
// shows itself once the page is ready; any other launch does not touch it.
func TestAnUpdateLaunchShowsItsWindowWhenThePageIsReady(t *testing.T) {
	for _, update := range []bool{true, false} {
		shown := 0
		a := newTestApp(&fakeBackend{}, Settings{}, nil, "")
		a.show = func(context.Context) { shown++ }
		a.ShowOnDOMReady(update)
		a.DOMReady(context.Background())
		a.DOMReady(context.Background()) // a page reload does not show it again
		want := 0
		if update {
			want = 1
		}
		if shown != want {
			t.Errorf("update launch %v: the window was shown %d times, want %d", update, shown, want)
		}
		select {
		case <-a.windowReady:
		default:
			t.Errorf("update launch %v: windowReady was not closed", update)
		}
	}
}
