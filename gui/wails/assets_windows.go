//go:build windows

package main

import "embed"

//go:embed all:frontend/dist
var assets embed.FS

// The native tray also works in Go-only builds, before Vite has produced dist.
//go:embed icons/owl.svg
var trayOwl []byte
