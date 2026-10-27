package main

import (
	"archive/zip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"bubo/gui/updater"
)

// This fixture uses the release generator and updater's package layout. It
// never executes a packaged file or invokes an update handoff.
func main() {
	var folder, origin string
	flag.StringVar(&folder, "folder", "", "output folder")
	flag.StringVar(&origin, "origin", "", "loopback HTTPS origin")
	flag.Parse()
	if folder == "" || origin == "" {
		panic("folder and origin are required")
	}
	if err := generate(folder, origin); err != nil {
		panic(err)
	}
}

func generate(folder, origin string) error {
	if err := os.MkdirAll(folder, 0700); err != nil {
		return err
	}
	const version = "0.1.1"
	const top = "bubo-v0.1.1-windows-x64-gfx1151"
	archive := filepath.Join(folder, top+".zip")
	f, err := os.Create(archive)
	if err != nil {
		return err
	}
	w := zip.NewWriter(f)
	for _, rel := range updater.PackageExecutables() {
		entry, err := w.Create(top + "/" + rel)
		if err != nil {
			return err
		}
		if _, err := entry.Write([]byte("inert test package entry")); err != nil {
			return err
		}
	}
	entry, err := w.Create(top + "/BUILD-MANIFEST.json")
	if err != nil {
		return err
	}
	build := map[string]any{"product": map[string]string{"product_version": version}, "gui": map[string]string{
		"commit": "129f723fb238211868099255d9468b896cc6865a", "runtime_id": "rocm-10.0.0-gfx1151"}}
	if err := json.NewEncoder(entry).Encode(build); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	changed, runtimeBytes := false, int64(0)
	m, _, err := updater.GenerateReleaseManifest(archive, updater.ReleaseOptions{
		RuntimeChanged: &changed, RuntimeDownloadBytes: &runtimeBytes, ModelsChanged: &changed, ModelsNeedsPrepare: []string{},
		MinUpdatableFrom: "0.1.0", Notes: "The Bubo app. Pick a model, press Start, and watch memory and requests. Update Bubo with one click.",
		Released: time.Date(2026, 10, 25, 15, 34, 48, 0, time.UTC), InstallPathLength: 100,
	})
	if err != nil {
		return err
	}
	const releasePath = "/haloforge-labs/bubo/releases/"
	m.Assets[0].URL = origin + releasePath + "download/" + m.Tag + "/" + m.Assets[0].Name
	m.Notes.URL = origin + releasePath + "tag/" + m.Tag
	manifest, err := os.Create(filepath.Join(folder, "bubo-release.json"))
	if err != nil {
		return err
	}
	if err := json.NewEncoder(manifest).Encode(m); err != nil {
		return err
	}
	if err := manifest.Close(); err != nil {
		return err
	}
	fmt.Println(archive)
	return nil
}
