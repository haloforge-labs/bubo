//go:build windows

package main

import (
	"archive/zip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"bubo/gui/updater"
)

// package is a loopback-only fixture generator. The caller supplies the
// already built bubo_test GUI, fake CLI and real updater helper, all copied
// under A4 evidence. No packaged file is executed by this generator.
func main() {
	var folder, origin, gui, cli, helper string
	var changed bool
	flag.StringVar(&folder, "folder", "", "fixture directory")
	flag.StringVar(&origin, "origin", "", "loopback HTTPS origin")
	flag.StringVar(&gui, "gui", "", "bubo_test GUI")
	flag.StringVar(&cli, "cli", "", "fake CLI")
	flag.StringVar(&helper, "helper", "", "real updater helper")
	flag.BoolVar(&changed, "changed", false, "manifest runtime.changed")
	flag.Parse()
	if folder == "" || origin == "" || gui == "" || cli == "" || helper == "" {
		panic("folder, origin, gui, cli, helper are required")
	}
	if err := generate(folder, origin, gui, cli, helper, changed); err != nil {
		panic(err)
	}
}

func generate(folder, origin, gui, cli, helper string, changed bool) error {
	if err := os.MkdirAll(folder, 0700); err != nil {
		return err
	}
	const top = "bubo-v0.1.1-windows-x64-gfx1151"
	archive := filepath.Join(folder, top+".zip")
	f, err := os.Create(archive)
	if err != nil {
		return err
	}
	w := zip.NewWriter(f)
	files := map[string]string{"bin/Bubo.Control.Web.exe": gui, "bin/bubo.exe": cli, "bin/bubo-update.exe": helper}
	for _, rel := range updater.PackageExecutables() {
		entry, err := w.Create(top + "/" + rel)
		if err != nil {
			return err
		}
		input, err := os.Open(files[rel])
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(entry, input)
		closeErr := input.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	entry, err := w.Create(top + "/BUILD-MANIFEST.json")
	if err != nil {
		return err
	}
	build := map[string]any{"product": map[string]string{"product_version": "0.1.1"},
		"gui": map[string]string{"commit": "129f723fb238211868099255d9468b896cc6865a", "runtime_id": "rocm-10.0.0-gfx1151"}}
	if err := json.NewEncoder(entry).Encode(build); err != nil {
		return err
	}
	version, err := w.Create(top + "/fixture-version")
	if err != nil {
		return err
	}
	if _, err := version.Write([]byte("0.1.1\n")); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	modelsChanged, runtimeBytes := false, int64(0)
	if changed {
		runtimeBytes = 1014229359
	}
	m, _, err := updater.GenerateReleaseManifest(archive, updater.ReleaseOptions{
		RuntimeChanged: &changed, RuntimeDownloadBytes: &runtimeBytes, ModelsChanged: &modelsChanged, ModelsNeedsPrepare: []string{},
		MinUpdatableFrom: "0.1.0", Notes: "The Bubo app. Pick a model, press Start, and watch memory and requests. Update Bubo with one click.",
		Released: time.Date(2026, 10, 25, 15, 34, 48, 0, time.UTC), InstallPathLength: 100})
	if err != nil {
		return err
	}
	const releasePath = "/haloforge-labs/bubo/releases/"
	m.Assets[0].URL = origin + releasePath + "download/" + m.Tag + "/" + m.Assets[0].Name
	m.Notes.URL = origin + releasePath + "tag/" + m.Tag

	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(folder, "bubo-release.json"), data, 0600); err != nil {
		return err
	}
	fmt.Println(archive)
	return nil
}
