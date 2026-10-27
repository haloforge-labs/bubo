//go:build windows && bindings

package main

import (
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
)

// Wails' bindings build runs main. This entry point deliberately constructs no
// lease, settings, credential manager, CLI backend, model service or GPU source.
// With the bindings tag wails.Run reflects methods and exits without a window.
func main() {
	if err := wails.Run(&options.App{Bind: []interface{}{&App{}}}); err != nil {
		panic(err)
	}
}
