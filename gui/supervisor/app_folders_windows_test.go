//go:build windows

package supervisor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeFolderOpener struct {
	paths []string
	err   error
}

func (f *fakeFolderOpener) OpenFolder(_ context.Context, path string) error {
	f.paths = append(f.paths, path)
	return f.err
}

func TestOpenModelFolderAllowsOnlySavedOrCLIReportedFolders(t *testing.T) {
	for _, source := range []string{"saved", "models root", "scanned folder", "catalogue package", "scanned model", "nested scanned model", "prepared package", "completed package", "status package"} {
		t.Run(source, func(t *testing.T) {
			folder := filepath.Join(t.TempDir(), "models and packages")
			if err := os.Mkdir(folder, 0700); err != nil {
				t.Fatal(err)
			}
			manifest := filepath.Join(folder, "manifest.json")
			settings := DefaultSettings()
			backend := &fakeBackend{}
			opener := &fakeFolderOpener{}
			a := NewAppService(backend, settings, nil, "", AppHooks{Folders: opener})
			a.ctx = context.Background()
			a.models = NewModelService(ModelServiceConfig{})
			t.Cleanup(a.models.Close)
			switch source {
			case "saved":
				a.settings.ModelFolders = []string{folder}
			case "models root":
				a.models.state.Listing = &ModelsListing{ModelsRoot: folder}
			case "scanned folder":
				a.models.state.Listing = &ModelsListing{ScannedFolders: []string{folder}}
			case "catalogue package":
				a.models.state.Catalogue = []CatalogueEntry{{LocalFolder: folder}}
			case "scanned model":
				a.models.state.Scanned = []ScannedModel{{Folder: folder}}
			case "nested scanned model":
				a.models.state.Scanned = []ScannedModel{{Folder: filepath.Dir(folder), Path: filepath.Join(folder, "model.gguf")}}
			case "prepared package":
				a.models.state.PreparedModels = []PreparedModel{{ManifestPath: &manifest}}
			case "completed package":
				a.models.job = &ModelJob{State: "done", ManifestPath: &manifest}
			case "status package":
				backend.view.Status = &Status{RuntimeDetail: RuntimeDetail{PreparedModels: []PreparedModel{{ManifestPath: &manifest}}}}
			}
			// Windows paths are compared case-insensitively after normalization.
			result := a.OpenModelFolder(strings.ToUpper(folder) + `\`)
			if result.Error != nil || len(opener.paths) != 1 || !SameFolder(opener.paths[0], folder) {
				t.Fatalf("authorized folder was not opened once: %+v %v", result, opener.paths)
			}
		})
	}
}

func TestOpenModelFolderRejectsUnreportedPathsBeforeNativeOpener(t *testing.T) {
	parent := t.TempDir()
	folder := filepath.Join(parent, "saved")
	child := filepath.Join(folder, "child")
	if err := os.MkdirAll(child, 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(folder, "model.gguf")
	if err := os.WriteFile(file, []byte("test"), 0600); err != nil {
		t.Fatal(err)
	}
	opener := &fakeFolderOpener{}
	settings := DefaultSettings()
	settings.ModelFolders = []string{folder}
	a := NewAppService(&fakeBackend{}, settings, nil, "", AppHooks{Folders: opener})
	a.ctx = context.Background()
	for _, test := range []struct {
		name, path, code string
	}{
		{"empty", "", "model_folder_invalid"},
		{"relative", "saved", "model_folder_invalid"},
		{"drive", filepath.VolumeName(folder) + `\`, "model_folder_invalid"},
		{"control", folder + "\nchild", "model_folder_invalid"},
		{"parent", parent, "model_folder_not_allowed"},
		{"child", child, "model_folder_not_allowed"},
		{"file", file, "model_folder_not_allowed"},
		{"prefix", folder + "-other", "model_folder_not_allowed"},
		{"escape", filepath.Join(folder, "..", "other"), "model_folder_not_allowed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			result := a.OpenModelFolder(test.path)
			if result.Error == nil || result.Error.Code != test.code || result.Error.Remedy == "" {
				t.Fatalf("wrong refusal: %+v", result)
			}
		})
	}
	if len(opener.paths) != 0 {
		t.Fatal("unauthorized path reached opener", opener.paths)
	}
}

func TestOpenModelFolderRejectsMissingOrNonFolderSavedPaths(t *testing.T) {
	for _, name := range []string{"missing", "file"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), name)
			if name == "file" {
				if err := os.WriteFile(path, []byte("test"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			settings := DefaultSettings()
			settings.ModelFolders = []string{path}
			opener := &fakeFolderOpener{}
			a := NewAppService(&fakeBackend{}, settings, nil, "", AppHooks{Folders: opener})
			a.ctx = context.Background()
			result := a.OpenModelFolder(path)
			if result.Error == nil || result.Error.Code != "model_folder_unavailable" || len(opener.paths) != 0 {
				t.Fatal("non-folder reached native opener", result, opener.paths)
			}
		})
	}
}

func TestOpenModelFolderReportsNativeFailureAndMissingAdapter(t *testing.T) {
	folder := t.TempDir()
	settings := DefaultSettings()
	settings.ModelFolders = []string{folder}
	opener := &fakeFolderOpener{err: errors.New("test shell failure")}
	a := NewAppService(&fakeBackend{}, settings, nil, "", AppHooks{Folders: opener})
	a.ctx = context.Background()
	if result := a.OpenModelFolder(folder); result.Error == nil || result.Error.Code != "folder_open_failed" || !strings.Contains(result.Error.Remedy, "test shell failure") || len(opener.paths) != 1 {
		t.Fatal("native failure was not typed", result)
	}
	a.folders = nil
	if result := a.OpenModelFolder(folder); result.Error == nil || result.Error.Code != "internal_error" {
		t.Fatal("missing adapter reported success", result)
	}
	a.folders = opener
	a.ctx = nil
	if result := a.OpenModelFolder(folder); result.Error == nil || result.Error.Code != "initializing" || len(opener.paths) != 1 {
		t.Fatal("unready window reached opener", result)
	}
}
