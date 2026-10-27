//go:build windows

package supervisor

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// The Models page bindings. The window calls them; the supervisor package
// owns the CLI calls (ModelService). The View contains ModelsState.
//
// Download is two calls so the user sees the plan first:
//
//	Download(id)        runs `prepare model --catalogue-id ID --plan`; returns
//	                    the plan (files, sizes, free space, licence text).
//	                    Nothing is downloaded.
//	ConfirmDownload(id) the user agreed to that plan: the download starts.

// PathDialogs are the native file and folder dialogs. The real ones use the
// Wails runtime; tests use a fake because a dialog cannot be driven here.
type PathDialogs interface {
	// OpenFile asks for one .gguf file. It returns "" when the user cancels.
	OpenFile(ctx context.Context, title string) (string, error)
	// OpenFolder asks for one folder. It returns "" when the user cancels.
	OpenFolder(ctx context.Context, title string) (string, error)
}

// BrowseResult is the answer of BrowseModelFile and BrowseModelFolder.
type BrowseResult struct {
	// Cancelled is true when the user closed the dialog without a choice.
	Cancelled bool `json:"cancelled"`
	// File is the .gguf file the user chose (BrowseModelFile only).
	File string `json:"file"`
	// Folder is the folder that was added to the list (or was already in it).
	Folder string `json:"folder"`
	// Added is false when the folder was already in the list.
	Added   bool      `json:"added"`
	Message string    `json:"message"`
	Error   *CLIError `json:"error"`
}

var errModelsUnavailable = appError("internal_error", "Model service is not attached.")

// attachModels creates the model service. cliDefault is the CLI beside the
// window; cliOverride is the -cli flag (wins over the settings).
func (a *AppService) AttachModels(_ string, cliDefault, cliOverride string) {
	a.models = NewModelService(ModelServiceConfig{
		Events: a.emitStream,
		OnJobEnd: func(job ModelJob) {
			if job.State == "done" && job.Prepared {
				// The CLI confirmed a successful model operation, including a
				// non-default model that status readiness may never describe.
				if backend, ok := a.backend.(interface{ modelReady() }); ok {
					backend.modelReady()
				}
			}
		},
		Exe: func() string {
			if cliOverride != "" {
				return cliOverride
			}
			return cliDefault
		},
		Inputs: func() ModelInputs {
			a.mu.RLock()
			defer a.mu.RUnlock()
			return ModelInputs{Folders: append([]string(nil), a.settings.ModelFolders...), DataFolder: a.settings.DataFolder, DownloadFolders: append([]string(nil), a.settings.DownloadFolders...), RuntimeDir: a.settings.RuntimePath}
		},
		Keys: a.keys,
		StallTimeout: func() time.Duration {
			a.mu.RLock()
			defer a.mu.RUnlock()
			return a.settings.StallTimeout()
		},
	})
}

// modelsFrame supplies View.models. It only copies cached data; the
// first call also starts the first background read of the list.
func (a *AppService) modelsFrame() *ModelsState {
	if a.models == nil {
		return nil
	}
	if !a.models.Started() {
		a.models.Refresh()
	}
	s := a.models.State()
	if s.Download != nil && s.Download.Plan != nil {
		s.Download.FilePosition = catalogueFilePosition(s.Download.FileID, s.Download.Plan, s.Catalogue)
		// A frame is polled often; the plan (licence text) is returned once by
		// StartModelDownload and stays in the frame only while it awaits the user.
		if s.Download.State != "awaiting_confirm" {
			j := *s.Download
			j.Plan = nil
			s.Download = &j
		}
	}
	return &s
}

// RefreshModels asks for a new read of the model list. It returns at once;
// the frame key "models" shows the result.
func (a *AppService) RefreshModels() ActionResult {
	if a.models == nil {
		return ActionResultFromError(errModelsUnavailable)
	}
	a.mu.Lock()
	a.estimate.at = time.Time{}
	a.mu.Unlock()
	a.models.Refresh()
	return ActionResult{}
}

// StartModelDownload runs the plan step for a catalogue id and returns the plan.
// Nothing is downloaded until ConfirmModelDownload.
func (a *AppService) StartModelDownload(id string) (ModelPlan, error) {
	if a.models == nil {
		return ModelPlan{}, errModelsUnavailable
	}
	return a.models.StartDownload(id, "download")
}

// CheckModel verifies a model's files. repair false: an offline check (no
// network, no download). repair true: the plan step of a repair; confirm it
// with ConfirmModelDownload. The returned plan is empty for a check.
func (a *AppService) CheckModel(id string, repair bool) ModelPlanResult {
	plan, err := a.checkModel(id, repair)
	if err != nil {
		return ModelPlanResult{Error: ActionResultFromError(err).Error}
	}
	return ModelPlanResult{Plan: &plan}
}

func (a *AppService) checkModel(id string, repair bool) (ModelPlan, error) {
	if a.models == nil {
		return ModelPlan{}, errModelsUnavailable
	}
	if repair {
		return a.models.StartDownload(id, "repair")
	}
	return ModelPlan{ID: id, Kind: "check"}, a.models.Check(id)
}

// RepairPackage plans the repair of one package: the prepared row of this id
// that carries this repair_output (the --output the CLI named for it). Any other
// folder is refused. Confirm it with ConfirmModelDownload.
func (a *AppService) RepairPackage(id, output string) ModelPlanResult {
	if a.models == nil {
		return ModelPlanResult{Error: ActionResultFromError(errModelsUnavailable).Error}
	}
	known := false
	for _, row := range a.models.State().PreparedModels {
		known = known || row.ID == id && row.RepairOutput != nil && SameFolder(*row.RepairOutput, output)
	}
	if !known {
		return ModelPlanResult{Error: folderActionError("model_folder_not_allowed", "Choose a package the CLI reported.").Error}
	}
	plan, err := a.models.StartRepair(id, output)
	if err != nil {
		return ModelPlanResult{Error: ActionResultFromError(err).Error}
	}
	return ModelPlanResult{Plan: &plan}
}

// ConfirmModelDownload starts the download or repair of the plan the user saw.
func (a *AppService) ConfirmModelDownload(id string) error {
	if a.models == nil {
		return errModelsUnavailable
	}
	return a.models.ConfirmDownload(id)
}

// CancelModelDownload drops a waiting plan, or asks the running job to cancel.
func (a *AppService) CancelModelDownload() error {
	if a.models == nil {
		return errModelsUnavailable
	}
	return a.models.CancelJob()
}

// DismissModelJob clears a finished download, repair or check from the frame.
func (a *AppService) DismissModelJob() ActionResult {
	if a.models == nil {
		return ActionResultFromError(errModelsUnavailable)
	}
	a.models.DismissJob()
	return ActionResult{}
}

// EstimateModel asks the CLI for the memory estimate of a catalogue id or a
// full file path at a context and session count (0 keeps the CLI default).
func (a *AppService) EstimateModel(subject string, contextTokens uint64, sessions uint32) (ModelEstimate, error) {
	if a.models == nil {
		return ModelEstimate{}, errModelsUnavailable
	}
	return a.models.Estimate(subject, contextTokens, sessions)
}

// AddModelFolder saves a folder to scan for models and refreshes the list. The
// CLI scans it (depth 3, .gguf only). It returns the clean folder path.
func (a *AppService) AddModelFolder(path string) StringResult {
	folder, _, err := a.addModelFolder(path)
	return StringResultFrom(folder, err)
}

func (a *AppService) addModelFolder(path string) (folder string, added bool, err error) {
	if a.models == nil {
		return "", false, errModelsUnavailable
	}
	clean, err := NormalizeModelFolderPath(path)
	if err != nil {
		return "", false, err
	}
	fi, statErr := os.Lstat(clean)
	switch {
	case statErr != nil:
		return "", false, appError("model_folder_unavailable", "That folder does not exist or cannot be read.")
	case !fi.IsDir() || fi.Mode()&(os.ModeSymlink|os.ModeIrregular) != 0:
		return "", false, appError("model_folder_not_allowed", "Choose a normal folder. Links and junctions are not scanned.")
	}
	a.folderMu.Lock()
	defer a.folderMu.Unlock()
	a.mu.RLock()
	settings := a.settings
	a.mu.RUnlock()
	for _, f := range settings.ModelFolders {
		if SameFolder(f, clean) {
			return f, false, nil
		}
	}
	if len(settings.ModelFolders) >= MaxModelFolders {
		return "", false, appError("model_folder_not_allowed", "The folder list is full. Remove a folder first.")
	}
	settings.ModelFolders = append(append([]string(nil), settings.ModelFolders...), clean)
	if err := a.persistFolders(settings); err != nil {
		return "", false, err
	}
	a.models.Refresh()
	return clean, true, nil
}

// RemoveModelFolder removes a folder from the list. The folder and its files stay.
func (a *AppService) RemoveModelFolder(path string) ActionResult {
	return ActionResultFromError(a.removeModelFolder(path))
}

func (a *AppService) removeModelFolder(path string) error {
	if a.models == nil {
		return errModelsUnavailable
	}
	a.folderMu.Lock()
	defer a.folderMu.Unlock()
	a.mu.RLock()
	settings := a.settings
	a.mu.RUnlock()
	var kept []string
	found := false
	for _, f := range settings.ModelFolders {
		if SameFolder(f, path) {
			found = true
			continue
		}
		kept = append(kept, f)
	}
	if !found {
		return appError("model_folder_not_allowed", "That folder is not in the list.")
	}
	settings.ModelFolders = kept
	if err := a.persistFolders(settings); err != nil {
		return err
	}
	a.models.Refresh()
	return nil
}

// All settings writes, including folders, use the backend's one writer.
func (a *AppService) persistFolders(settings Settings) error {
	if err := a.backend.Save(settings); err != nil {
		return err
	}
	a.mu.Lock()
	a.settings = settings
	a.mu.Unlock()
	return nil
}

func (a *AppService) dialogContext() (context.Context, error) {
	a.mu.RLock()
	ctx := a.ctx
	a.mu.RUnlock()
	if ctx == nil {
		return nil, appError("initializing", "The window is not ready.")
	}
	return ctx, nil
}

// BrowseModelFolder opens the folder dialog and adds the chosen folder.
func (a *AppService) BrowseModelFolder() BrowseResult {
	result, err := a.browseModelFolder()
	if err != nil {
		return BrowseResult{Error: ActionResultFromError(err).Error}
	}
	return result
}

func (a *AppService) browseModelFolder() (BrowseResult, error) {
	if a.models == nil {
		return BrowseResult{}, errModelsUnavailable
	}
	if a.dialogs == nil {
		return BrowseResult{}, appError("internal_error", "The native folder dialog is not attached.")
	}
	ctx, err := a.dialogContext()
	if err != nil {
		return BrowseResult{}, err
	}
	chosen, err := a.dialogs.OpenFolder(ctx, "Choose a folder with model files")
	if err != nil {
		return BrowseResult{}, appError("model_folder_unavailable", "The folder dialog failed: "+err.Error())
	}
	if chosen == "" {
		return BrowseResult{Cancelled: true, Message: "No folder was chosen."}, nil
	}
	folder, added, err := a.addModelFolder(chosen)
	if err != nil {
		return BrowseResult{}, err
	}
	return BrowseResult{Folder: folder, Added: added, Message: addedMessage(folder, added)}, nil
}

// BrowseModelFile opens the file dialog for one .gguf file. The CLI lists
// models by folder, so the folder of the file is added to the list.
func (a *AppService) BrowseModelFile() BrowseResult {
	result, err := a.browseModelFile()
	if err != nil {
		return BrowseResult{Error: ActionResultFromError(err).Error}
	}
	return result
}

func (a *AppService) browseModelFile() (BrowseResult, error) {
	if a.models == nil {
		return BrowseResult{}, errModelsUnavailable
	}
	if a.dialogs == nil {
		return BrowseResult{}, appError("internal_error", "The native file dialog is not attached.")
	}
	ctx, err := a.dialogContext()
	if err != nil {
		return BrowseResult{}, err
	}
	chosen, err := a.dialogs.OpenFile(ctx, "Choose a model file")
	if err != nil {
		return BrowseResult{}, appError("model_folder_unavailable", "The file dialog failed: "+err.Error())
	}
	if chosen == "" {
		return BrowseResult{Cancelled: true, Message: "No file was chosen."}, nil
	}
	if !strings.EqualFold(filepath.Ext(chosen), ".gguf") {
		return BrowseResult{}, appError("model_file_invalid", "Choose a file that ends in .gguf.")
	}
	folder, added, err := a.addModelFolder(filepath.Dir(chosen))
	if err != nil {
		return BrowseResult{}, appError("model_folder_not_allowed", "The file is in a folder that cannot be added. "+err.Error())
	}
	return BrowseResult{File: chosen, Folder: folder, Added: added, Message: addedMessage(folder, added)}, nil
}

func addedMessage(folder string, added bool) string {
	if added {
		return "Folder added: " + folder
	}
	return "This folder is already in the list: " + folder
}
