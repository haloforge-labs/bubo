//go:build windows

package supervisor

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// A draft file or an image file is chosen with the file dialog and saved as the
// CLI's own --mtp-model / --mmproj override. The dialog adds no folder to the
// model list.
func TestBrowseSideFileReturnsThePathAndAddsNoFolder(t *testing.T) {
	m := newModelsApp(t, "")
	folder := m.folder(t, "side-files")
	m.dialogs.file = filepath.Join(folder, "mtp-draft.GGUF")
	for _, kind := range []string{"draft", "image"} {
		res := m.a.BrowseSideFile(kind)
		if res.Error != nil || res.Cancelled || res.File != m.dialogs.file || res.Folder != "" || res.Added {
			t.Fatalf("%s: %+v", kind, res)
		}
	}
	if saved, _ := LoadSettings(m.settings); len(saved.ModelFolders) != 0 {
		t.Fatalf("a side file added a model folder: %v", saved.ModelFolders)
	}
	if res := m.a.BrowseSideFile("twin"); res.Error == nil {
		t.Fatal("an unknown kind was accepted")
	}
	m.dialogs.file = filepath.Join(folder, "notes.txt")
	if res := m.a.BrowseSideFile("draft"); res.Error == nil || !strings.Contains(res.Error.Remedy, ".gguf") {
		t.Fatalf("a non-gguf file: %+v", res)
	}
	m.dialogs.file = ""
	if res := m.a.BrowseSideFile("image"); !res.Cancelled || res.Error != nil {
		t.Fatalf("cancel: %+v", res)
	}
}

// Start passes both side files of a plain model file; changing to a package
// model drops them, so a package model passes neither.
func TestSideFilesGoToStartAndAreDroppedForAPackage(t *testing.T) {
	registry := viewFixture(t, "switches").Value.(*Switches)
	overrides := map[string]string{
		"--mtp-model": `C:\side\mtp.gguf`,
		"--mmproj":    `C:\side\mmproj.gguf`,
		"--top-k":     "40",
	}
	args, _, err := switchCommand(overrides, registry)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, `--mtp-model C:\side\mtp.gguf`) || !strings.Contains(joined, `--mmproj C:\side\mmproj.gguf`) {
		t.Fatalf("start arguments lack a side file: %v", args)
	}
	kept := dropModelFileOverrides(overrides, registry)
	if !reflect.DeepEqual(kept, map[string]string{"--top-k": "40"}) {
		t.Fatalf("a model change must drop the side files and keep other settings: %v", kept)
	}
}

// The CLI judges a chosen side file against the saved model file; the answer
// travels with the file. Without a saved model, or from a CLI that cannot
// check, the result says unchecked.
func TestBrowseSideFileCarriesTheCLIVerdict(t *testing.T) {
	m := newModelsApp(t, "")
	folder := m.folder(t, "verdict")
	m.dialogs.file = filepath.Join(folder, "mtp-ok.gguf")
	if res := m.a.BrowseSideFile("draft"); res.Error != nil || res.Checked {
		t.Fatalf("no saved model must stay unchecked: %+v", res)
	}
	m.a.mu.Lock()
	m.a.settings.ModelPath = filepath.Join(folder, "model.gguf")
	m.a.mu.Unlock()
	if res := m.a.BrowseSideFile("draft"); res.Error != nil || !res.Checked || !res.Fits || res.Reason != "" {
		t.Fatalf("a fitting file: %+v", res)
	}
	m.dialogs.file = filepath.Join(folder, "wrong-mmproj.gguf")
	res := m.a.BrowseSideFile("image")
	if res.Error != nil || !res.Checked || res.Fits || res.Reason == "" || res.File != m.dialogs.file {
		t.Fatalf("a file for another model: %+v", res)
	}
}
