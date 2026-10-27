//go:build windows

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The package mode must follow the CLI's folder rules, or every scenario built
// on it proves nothing (E19). These tests pin the rules the real CLI has at
// engine e7af8f43 plus the E20 contract.

func packagesRig(t *testing.T) (local string) {
	t.Helper()
	oldDir, oldFixtures, oldArgs := dir, fixtures, os.Args
	t.Cleanup(func() { dir, fixtures, os.Args = oldDir, oldFixtures, oldArgs })
	dir = t.TempDir()
	abs, err := filepath.Abs(filepath.Join("..", "..", "..", "tests", "cli", "contract"))
	if err != nil {
		t.Fatal(err)
	}
	fixtures = abs
	local = filepath.Join(dir, "local")
	t.Setenv("LOCALAPPDATA", local)
	t.Setenv("BUBO_FAKE_PACKAGES", "1")
	t.Setenv("BUBO_MANIFEST", "")
	return local
}

func TestPackageStateIsPartialWhenAFileIsMissing(t *testing.T) {
	packagesRig(t)
	folder := filepath.Join(t.TempDir(), "models", "id-one")
	if packageState(folder) != "not_prepared" {
		t.Fatal("an empty folder is not prepared")
	}
	if err := writePackage(folder, "id-one", 9); err != nil {
		t.Fatal(err)
	}
	if packageState(folder) != "prepared" {
		t.Fatal("a written package is prepared")
	}
	if err := os.Remove(filepath.Join(folder, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	if packageState(folder) != "partial" {
		t.Fatal("a marker with a missing file is partial")
	}
	// Files without the marker (a download that has not finished) are not a package.
	if err := os.Remove(filepath.Join(folder, packageMarker)); err != nil {
		t.Fatal(err)
	}
	if packageState(folder) != "not_prepared" {
		t.Fatal("no marker, no package")
	}
}

func TestListReportsPackagesFoundInScanFoldersWithoutLooseFiles(t *testing.T) {
	packagesRig(t)
	scan := t.TempDir()
	catalogue := recordedCatalogue()
	id := defaultCatalogueID(catalogue)
	here := filepath.Join(scan, "models", id)
	other := filepath.Join(scan, "elsewhere", "custom")
	if err := writePackage(here, id, 5); err != nil {
		t.Fatal(err)
	}
	if err := writePackage(other, "not-a-shipped-id", 3); err != nil {
		t.Fatal(err)
	}
	os.Args = []string{"fakecli", "models", "list", "--json", "--scan-folder", scan, "--scan-folder", filepath.Join(scan, "missing")}
	_, list := packagesModelsList()
	rows := list["prepared_models"].([]any)
	if len(rows) != 2 {
		t.Fatalf("rows: %v", rows)
	}
	byID := map[string]map[string]any{}
	for _, r := range rows {
		row := r.(map[string]any)
		byID[row["id"].(string)] = row
	}
	if byID[id]["catalogue"] != true || byID["not-a-shipped-id"]["catalogue"] != false {
		t.Fatalf("catalogue flags: %v", rows)
	}
	if byID[id]["manifest_path"] != filepath.Join(here, packageMarker) {
		t.Fatalf("manifest: %v", byID[id])
	}
	if len(list["scanned"].([]any)) != 0 {
		t.Fatal("package files were listed as loose models")
	}
	warnings := list["scan_warnings"].([]any)
	if len(warnings) != 1 {
		t.Fatalf("warnings: %v", warnings)
	}
	// The catalogue state is read at the default root only (no --models-root).
	for _, r := range list["catalogue"].([]any) {
		if r.(map[string]any)["prepared_state"] != "not_prepared" {
			t.Fatalf("a package in a scan folder changed the default root's state: %v", r)
		}
	}
}

func TestListGivesTwoRowsForTheSameIdInTwoFolders(t *testing.T) {
	packagesRig(t)
	id := defaultCatalogueID(recordedCatalogue())
	a, b := t.TempDir(), t.TempDir()
	for _, root := range []string{a, b} {
		if err := writePackage(filepath.Join(root, "models", id), id, 5); err != nil {
			t.Fatal(err)
		}
	}
	os.Args = []string{"fakecli", "models", "list", "--json", "--scan-folder", a, "--scan-folder", b}
	_, list := packagesModelsList()
	if rows := list["prepared_models"].([]any); len(rows) != 2 {
		t.Fatalf("rows: %v", rows)
	}
}

func TestSelectionOrderIsFlagEnvironmentPointerStandard(t *testing.T) {
	local := packagesRig(t)
	id := defaultCatalogueID(recordedCatalogue())
	standard := filepath.Join(local, "Bubo", "models", id, packageMarker)

	os.Args = []string{"fakecli", "status", "--json"}
	if m, _, source := selectedManifest(); m != standard || source != "standard_location" {
		t.Fatalf("standard: %q %q", m, source)
	}
	pointer := filepath.Join(local, "Bubo", pointerFile)
	if err := os.MkdirAll(filepath.Dir(pointer), 0700); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(map[string]any{"schema_version": 1, "manifest_path": `D:\x\bubo-package.json`})
	if err := os.WriteFile(pointer, b, 0600); err != nil {
		t.Fatal(err)
	}
	if m, _, source := selectedManifest(); m != `D:\x\bubo-package.json` || source != "prepared_default" {
		t.Fatalf("pointer: %q %q", m, source)
	}
	t.Setenv("BUBO_MANIFEST", `E:\env\bubo-package.json`)
	if m, _, source := selectedManifest(); m != `E:\env\bubo-package.json` || source != "environment" {
		t.Fatalf("environment: %q %q", m, source)
	}
	os.Args = []string{"fakecli", "serve", "--manifest", `F:\flag\bubo-package.json`}
	if m, _, source := selectedManifest(); m != `F:\flag\bubo-package.json` || source != "explicit_manifest" {
		t.Fatalf("flag: %q %q", m, source)
	}
}

func TestStatusFollowsOnlyThePointerNotTheChosenFolder(t *testing.T) {
	// status has no folder input: a package in another folder is invisible to it
	// unless the pointer or BUBO_MANIFEST names it (the engine need in the report).
	local := packagesRig(t)
	id := defaultCatalogueID(recordedCatalogue())
	other := t.TempDir()
	if err := writePackage(filepath.Join(other, "models", id), id, 5); err != nil {
		t.Fatal(err)
	}
	os.Args = []string{"fakecli", "status", "--json"}
	p := map[string]any{"paths": map[string]any{}}
	packagesStatus(p)
	if p["prepared"] != false || p["phase"] != "not_prepared" {
		t.Fatalf("status saw a package it has no input for: %v", p)
	}
	_ = local
}
