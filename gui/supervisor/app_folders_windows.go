//go:build windows

package supervisor

import (
	"context"
	"os"
	"path/filepath"
)

// FolderOpener executes a validated native request. AppService owns the allow
// list; the Wails adapter opens the folder and tests inject a recording fake.
type FolderOpener interface {
	OpenFolder(context.Context, string) error
}

// OpenModelFolder opens only an exact saved or CLI-reported model/package
// folder. It never accepts a descendant just because its parent is allowed.
func (a *AppService) OpenModelFolder(path string) ActionResult {
	clean, err := NormalizeModelFolderPath(path)
	if err != nil {
		return folderActionError("model_folder_invalid", err.Error())
	}
	a.mu.RLock()
	known := append([]string(nil), a.settings.ModelFolders...)
	ctx := a.ctx
	a.mu.RUnlock()
	if a.models != nil {
		s := a.models.State()
		if s.Listing != nil {
			known = append(known, s.Listing.ModelsRoot)
			known = append(known, s.Listing.ScannedFolders...)
		}
		for _, row := range s.Catalogue {
			known = append(known, row.LocalFolder)
		}
		for _, row := range s.Scanned {
			known = append(known, row.Folder)
			if filepath.IsAbs(row.Path) {
				known = append(known, filepath.Dir(row.Path))
			}
		}
		known = appendPreparedFolders(known, s.PreparedModels)
		if s.Download != nil && s.Download.ManifestPath != nil {
			known = append(known, filepath.Dir(*s.Download.ManifestPath))
		}
	}
	if status := a.backend.Snapshot().Status; status != nil {
		known = appendPreparedFolders(known, status.PreparedModels)
	}
	allowed := false
	for _, folder := range known {
		// Invalid/missing CLI fields cannot accidentally authorize "." or a root.
		candidate, normalizeErr := NormalizeModelFolderPath(folder)
		if normalizeErr == nil && SameFolder(candidate, clean) {
			allowed = true
			break
		}
	}
	if !allowed {
		return folderActionError("model_folder_not_allowed", "Choose a model or package folder reported by the CLI, or a saved model folder.")
	}
	info, err := os.Lstat(clean)
	if err != nil || !info.IsDir() || info.Mode()&(os.ModeSymlink|os.ModeIrregular) != 0 {
		return folderActionError("model_folder_unavailable", "That model folder is missing, unreadable, or is a link or junction.")
	}
	if a.folders == nil {
		return codedAction("internal_error", "The native folder opener is not attached.")
	}
	if ctx == nil {
		return codedAction("initializing", "The window is not ready to open a model folder.")
	}
	if err := a.folders.OpenFolder(ctx, clean); err != nil {
		return folderActionError("folder_open_failed", "The model folder could not be opened: "+err.Error())
	}
	return ActionResult{}
}

func appendPreparedFolders(folders []string, rows []PreparedModel) []string {
	for _, row := range rows {
		if row.ManifestPath != nil && filepath.IsAbs(*row.ManifestPath) {
			folders = append(folders, filepath.Dir(*row.ManifestPath))
		}
	}
	return folders
}

func folderActionError(code, remedy string) ActionResult {
	return ActionResult{Error: &CLIError{Code: code, Remedy: remedy}}
}
