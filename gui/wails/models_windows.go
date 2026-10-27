//go:build windows

package main

import (
	"context"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type wailsDialogs struct{}

func (wailsDialogs) OpenFile(ctx context.Context, title string) (string, error) {
	return runtime.OpenFileDialog(ctx, runtime.OpenDialogOptions{
		Title:   title,
		Filters: []runtime.FileFilter{{DisplayName: "GGUF model files (*.gguf)", Pattern: "*.gguf"}},
	})
}
func (wailsDialogs) OpenFolder(ctx context.Context, title string) (string, error) {
	return runtime.OpenDirectoryDialog(ctx, runtime.OpenDialogOptions{Title: title})
}
