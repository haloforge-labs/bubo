//go:build windows

package supervisor

import (
	"os"
	"slices"
)

// The download folder (preference data_folder) decides only where NEW model
// downloads go. Nothing prepared before disappears:
//
//   - every folder used is remembered (Settings.DownloadFolders) and passed to
//     `models list` as a scan folder, together with the Import folders;
//   - the CLI returns each package found there as a prepared_models row with its
//     manifest_path, and the GUI starts it with serve --manifest;
//   - the GUI keeps no list of models and derives no model path of its own.

// DownloadFolderView is what the Settings row and the model consent show. Path
// is the folder new downloads go to, as the user chose it, or the CLI's default
// folder (empty until the CLI has named it). Drive and FreeBytes describe that
// folder's drive; FreeBytes is null while the space has not been read.
type DownloadFolderView struct {
	Path      string             `json:"path"`
	IsDefault bool               `json:"is_default"`
	Drive     string             `json:"drive"`
	FreeBytes *uint64            `json:"free_bytes"`
	Change    ActionAvailability `json:"change"`
}

// folderWritable proves that the folder accepts a new file, then removes it.
// A variable only so tests can script a folder Windows would refuse.
var folderWritable = func(folder string) error {
	file, err := os.CreateTemp(folder, ".bubo-write-test-*")
	if err != nil {
		return err
	}
	name := file.Name()
	if err := file.Close(); err != nil {
		_ = os.Remove(name)
		return err
	}
	return os.Remove(name)
}

func downloadFolderView(in ViewInput) DownloadFolderView {
	v := DownloadFolderView{IsDefault: in.Saved.DataFolder == "", Change: downloadFolderAvailability(in)}
	if in.Models != nil {
		v.Path, v.Drive, v.FreeBytes = in.Models.Disk.Path, in.Models.Disk.Drive, in.Models.Disk.FreeBytes
	}
	if !v.IsDefault {
		v.Path = in.Saved.DataFolder
	}
	return v
}

// downloadFolderAvailability is the one rule for changing the folder. Both the
// Settings row and the consent dialog read it, and every write checks it again.
func downloadFolderAvailability(in ViewInput) ActionAvailability {
	s := in.Snapshot
	switch s.State {
	case Starting, Ready, Stopping, AttachedExternal:
		return blockedAction("download_folder_locked", "The server is running. Stop it to change the download folder.")
	case Preparing:
		return blockedAction("download_folder_locked", "Bubo is downloading files. Wait for it to finish to change the download folder.")
	}
	if in.Models != nil && in.Models.Download != nil {
		switch in.Models.Download.State {
		case "planning", "running", "cancelling":
			return blockedAction("download_folder_locked", "A model download is running. Wait for it or cancel it to change the download folder.")
		}
	}
	return ActionAvailability{Allowed: true}
}

// checkDownloadFolder returns the clean folder the user chose, or a typed error.
func checkDownloadFolder(path string) (string, error) {
	clean, err := NormalizeModelFolderPath(path)
	if err != nil {
		return "", appError("download_folder_invalid", err.Error())
	}
	info, statErr := os.Lstat(clean)
	switch {
	case statErr != nil:
		return "", appError("download_folder_invalid", "That folder does not exist or cannot be read.")
	case !info.IsDir() || info.Mode()&(os.ModeSymlink|os.ModeIrregular) != 0:
		return "", appError("download_folder_invalid", "Choose a normal folder. Links and junctions are not used.")
	}
	if err := folderWritable(clean); err != nil {
		return "", appError("download_folder_unwritable", "Bubo cannot write to that folder: "+err.Error())
	}
	return clean, nil
}

// applyDownloadFolderPatch enforces the folder rules on a preference patch. It
// runs inside PatchPreferences, so the Settings row, the consent dialog and any
// other caller get the same checks. It returns true when the folder changed.
func (a *AppService) applyDownloadFolderPatch(current Settings, next *Settings, patch PreferencesPatch) (bool, error) {
	touched := patch.Set.DataFolder != nil || slices.Contains(patch.Clear, "data_folder")
	if !touched {
		return false, nil
	}
	if patch.Set.DataFolder != nil {
		clean, err := checkDownloadFolder(*patch.Set.DataFolder)
		if err != nil {
			return false, err
		}
		next.DataFolder = clean
	}
	changed := !SameFolder(current.DataFolder, next.DataFolder)
	if changed {
		if change := a.GetView().DownloadFolder.Change; !change.Allowed {
			return false, &actionUnavailableError{change}
		}
	}
	if next.DataFolder != "" {
		known := false
		for _, folder := range next.DownloadFolders {
			known = known || SameFolder(folder, next.DataFolder)
		}
		if !known && len(next.DownloadFolders) < MaxModelFolders {
			next.DownloadFolders = append(append([]string(nil), next.DownloadFolders...), next.DataFolder)
		}
	}
	return changed, nil
}

// BrowseDownloadFolder opens the native folder picker and saves the folder as
// the download folder. A cancelled picker changes nothing.
func (a *AppService) BrowseDownloadFolder() BrowseResult {
	result, err := a.browseDownloadFolder()
	if err != nil {
		return BrowseResult{Error: ActionResultFromError(err).Error}
	}
	return result
}

func (a *AppService) browseDownloadFolder() (BrowseResult, error) {
	if a.dialogs == nil {
		return BrowseResult{}, appError("internal_error", "The native folder dialog is not attached.")
	}
	ctx, err := a.dialogContext()
	if err != nil {
		return BrowseResult{}, err
	}
	// Do not open a picker the write would refuse.
	if change := a.GetView().DownloadFolder.Change; !change.Allowed {
		return BrowseResult{}, &actionUnavailableError{change}
	}
	chosen, err := a.dialogs.OpenFolder(ctx, "Choose a folder for new model downloads")
	if err != nil {
		return BrowseResult{}, appError("download_folder_invalid", "The folder dialog failed: "+err.Error())
	}
	if chosen == "" {
		return BrowseResult{Cancelled: true, Message: "No folder was chosen."}, nil
	}
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{DataFolder: &chosen}}); r.Error != nil {
		return BrowseResult{}, appError(r.Error.Code, r.Error.Remedy)
	}
	a.mu.RLock()
	folder := a.settings.DataFolder
	a.mu.RUnlock()
	return BrowseResult{Folder: folder, Message: "New downloads go to " + folder}, nil
}
