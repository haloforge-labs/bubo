//go:build windows

package supervisor

import "time"

type streamProvider interface{ SetEventSink(func(StreamEvent)) }

func (a *AppService) emitStream(e StreamEvent) { a.EmitStream(e) }

func (a *AppService) EmitStream(e StreamEvent) {
	if e.Kind == runtimeReadyStream {
		a.RefreshModels()
		return
	}
	a.mu.RLock()
	ctx, send := a.ctx, a.events
	a.mu.RUnlock()
	if ctx != nil && send != nil {
		// The View still owns the full current value. Events only hasten a
		// repaint, so bound their combined cadence across progress and logs.
		a.streamMu.Lock()
		now := time.Now()
		if now.Sub(a.lastStreamAt) < 150*time.Millisecond {
			a.streamMu.Unlock()
			return
		}
		a.lastStreamAt = now
		a.streamMu.Unlock()
		send(ctx, e)
	}
}

func (a *AppService) TrayMenu() TrayMenu {
	v := a.GetView()
	return buildTrayMenu(Snapshot{Instance: viewTrayInstance(v), Label: v.StatusLabel}, v.Actions)
}

func viewTrayInstance(v View) *Instance {
	if v.Instance == nil {
		return nil
	}
	return &Instance{Endpoint: v.Instance.Endpoint}
}
