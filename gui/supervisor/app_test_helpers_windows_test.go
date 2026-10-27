//go:build windows

package supervisor

import "context"

func newTestApp(b Backend, settings Settings, lease *SingleInstance, warning string) *AppService {
	noop := func(context.Context) {}
	return NewAppService(b, settings, lease, warning, AppHooks{Quit: noop, Show: noop, Hide: noop, Tray: &fakeTray{}, Events: func(context.Context, StreamEvent) {}})
}

func appTestTrayMenu(s Snapshot) TrayMenu {
	s.Initialized = true
	if s.State == Ready || s.State == AttachedExternal && s.Managed {
		s.Prepared = true
	}
	if s.Managed {
		s.ManagementSupported = true
	}
	fit := true
	s.Status = &Status{Readiness: Readiness{Runtime: true, Model: true}, MemoryEstimate: &MemoryEstimate{Fits: &fit}}
	return buildTrayMenu(s, lifecycleActions(ViewInput{Snapshot: s}))
}
