//go:build windows

package main

import (
	"bubo/gui/supervisor"
	"context"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

const (
	PrepareProgressEvent  = "prepare:progress"
	DownloadProgressEvent = "download:progress"
	LogLineEvent          = "log:line"
	UpdateProgressEvent   = "update:progress"
)

func emitWailsStream(ctx context.Context, e supervisor.StreamEvent) {
	switch e.Kind {
	case supervisor.PrepareProgress:
		runtime.EventsEmit(ctx, PrepareProgressEvent, supervisor.Filled(e.Progress))
	case supervisor.DownloadProgress:
		runtime.EventsEmit(ctx, DownloadProgressEvent, supervisor.Filled(e.Job))
	case supervisor.LogLine:
		runtime.EventsEmit(ctx, LogLineEvent, supervisor.Filled(e.Line))
	case supervisor.UpdateProgressStream:
		runtime.EventsEmit(ctx, UpdateProgressEvent, supervisor.Filled(e.Update))
	}
}
