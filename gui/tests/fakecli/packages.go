//go:build windows

package main

// Package mode (BUBO_FAKE_PACKAGES=1): the fake keeps model packages on a real
// disk and follows the folder inputs the way the real CLI does, so a test can
// move the download folder and see what the CLI would see (E19). Sources:
// src/cli/models_support.cpp and src/cli/prepare/prepare.cpp at engine
// e7af8f43, plus the E20 contract (package folders found by --scan-folder).
//
// E21: the catalogue, the package rows, the estimates and the plans are the
// recordings of the candidate exe (recorded_windows.go). Only what depends on
// the test's own disk is set here, and each such field is named in the E21
// report as hand-written: the folder paths, the prepared state of a folder, the
// drive name and the scripted free space of a plan volume, and the subject of
// an estimate. Everything else is the recorded record.
//
// Layout and rules mirrored from the CLI:
//
//   - prepare model --catalogue-id ID --output ROOT writes ROOT\models\ID
//     (default ROOT: %LOCALAPPDATA%\Bubo). bubo-package.json is written last.
//   - Preparing the default catalogue entry records the pointer
//     %LOCALAPPDATA%\Bubo\prepared-default.json, even with --output.
//   - models list reads catalogue state from ONE root (--models-root, else the
//     default root). Every --scan-folder is walked (depth 3, no links) for
//     folders that hold bubo-package.json; each becomes a prepared_models row
//     with its manifest_path (E20). The same id in two folders gives two rows.
//   - models estimate --model takes a catalogue id (read at the models root) or
//     a package folder. A shipped Flash Next folder keeps its measured table.
//   - status and serve pick the model: --manifest, --model, BUBO_MANIFEST, the
//     pointer, then the standard location. status takes no folder input.
//   - serve --manifest of a missing or incomplete package fails model_missing.
//
// Files in BUBO_FAKE_DIR:
//
//	free.json   {"C:": bytes, "D:": bytes} free space per drive (default 1 TiB for
//	            the list's available_bytes; a plan keeps its recorded numbers
//	            unless the drive is named here)

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func packagesMode() bool { return os.Getenv("BUBO_FAKE_PACKAGES") == "1" }

const (
	packageMarker = "bubo-package.json"
	pointerFile   = "prepared-default.json"
	fakeGiB       = uint64(1) << 30
	scanDepth     = 3
)

// fakeLocalAppData is where the fake CLI keeps its "Bubo" folder.
func fakeLocalAppData() string {
	if v := os.Getenv("LOCALAPPDATA"); v != "" {
		return v
	}
	return filepath.Join(dir, "local")
}
func defaultDataRoot() string   { return filepath.Join(fakeLocalAppData(), "Bubo") }
func defaultModelsRoot() string { return filepath.Join(defaultDataRoot(), "models") }

type packageMarkerFile struct {
	PackageID   string   `json:"package_id"`
	DisplayName string   `json:"display_name,omitempty"`
	SizeBytes   uint64   `json:"size_bytes"`
	Files       []string `json:"files"`
}

func readPackageMarker(folder string) (packageMarkerFile, bool) {
	var m packageMarkerFile
	b, err := os.ReadFile(filepath.Join(folder, packageMarker))
	if err != nil || json.Unmarshal(b, &m) != nil || m.PackageID == "" {
		return m, false
	}
	return m, true
}

// packageState is the readiness size check: no marker is not_prepared, a
// marker with a missing file is partial.
func packageState(folder string) string {
	m, ok := readPackageMarker(folder)
	if !ok {
		return "not_prepared"
	}
	for _, name := range m.Files {
		if fi, err := os.Stat(filepath.Join(folder, name)); err != nil || !fi.Mode().IsRegular() {
			return "partial"
		}
	}
	return "prepared"
}

// writePackage writes the files first and the marker last, as prepare does.
func writePackage(folder, id string, size uint64) error {
	if err := os.MkdirAll(folder, 0700); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(folder, "model.gguf"), []byte("fake-gguf"), 0600); err != nil {
		return err
	}
	b, err := json.Marshal(packageMarkerFile{PackageID: id, SizeBytes: size, Files: []string{"model.gguf"}})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(folder, packageMarker), b, 0600)
}

func volumeOf(path string) string { return strings.ToUpper(filepath.VolumeName(path)) }

// scriptedFree reads the scripted free space of the drive that holds path.
func scriptedFree(path string) (uint64, bool) {
	free := map[string]uint64{}
	if b, err := os.ReadFile(filepath.Join(dir, "free.json")); err == nil {
		var raw map[string]uint64
		if json.Unmarshal(b, &raw) == nil {
			for k, v := range raw {
				free[strings.ToUpper(k)] = v
			}
		}
	}
	v, ok := free[volumeOf(path)]
	return v, ok
}

// volumeFree is the scripted free space, else 1 TiB.
func volumeFree(path string) uint64 {
	if v, ok := scriptedFree(path); ok {
		return v
	}
	return 1 << 40
}

func fileExists(path string) bool { _, err := os.Stat(path); return err == nil }

// findPackageFolders walks a scan folder: depth limit 3, no links, and it does
// not descend into a package folder (E20).
func findPackageFolders(root string) []string {
	var found []string
	var walk func(folder string, depth int)
	walk = func(folder string, depth int) {
		if fileExists(filepath.Join(folder, packageMarker)) {
			found = append(found, folder)
			return
		}
		if depth >= scanDepth {
			return
		}
		entries, err := os.ReadDir(folder)
		if err != nil {
			return
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
		for _, e := range entries {
			if e.IsDir() && e.Type()&os.ModeSymlink == 0 {
				walk(filepath.Join(folder, e.Name()), depth+1)
			}
		}
	}
	walk(root, 0)
	return found
}

// repairOutput is the --output that checks and repairs a package: the folder
// above models\<id> when the package sits in prepare's layout, else null. The
// real CLI owns this rule (engine job E20); the fake mirrors it for tests.
func repairOutput(id, folder string) any {
	models := filepath.Dir(folder)
	if strings.EqualFold(filepath.Base(folder), id) && strings.EqualFold(filepath.Base(models), "models") {
		return filepath.Dir(models)
	}
	return nil
}

// preparedTemplate is the recorded package row the fake builds a row from: the
// row of the same id when the recording has one, else the imported package row.
func preparedTemplate(id string) map[string]any {
	_, list := recordedOne("models-list-gui.json")
	rows, _ := list["prepared_models"].([]any)
	var stock any
	for _, r := range rows {
		if row, ok := r.(map[string]any); ok {
			if row["id"] == id {
				stock = row
				break
			}
			if row["id"] == stockID {
				stock = row
			}
		}
	}
	b, _ := json.Marshal(stock)
	var out map[string]any
	if json.Unmarshal(b, &out) != nil || out == nil {
		missingRecording("a prepared package row")
	}
	return out
}

// preparedRow is a recorded package row with the fields that name this test's
// disk replaced (hand-written: id, name, paths, size, state, catalogue flag and
// the repair output).
func preparedRow(id, display, folder, state string, size uint64, catalogue bool) map[string]any {
	row := preparedTemplate(id)
	row["repair_output"] = repairOutput(id, folder)
	row["id"], row["display_name"] = id, display
	row["manifest_path"] = filepath.Join(folder, packageMarker)
	row["model_path"] = filepath.Join(folder, "model.gguf")
	if marker, ok := readPackageMarker(folder); ok && len(marker.Files) > 0 {
		row["model_path"] = filepath.Join(folder, marker.Files[0])
		_, listing := recordedOne("models-list-empty.json")
		if entry := catalogueRow(listing["catalogue"].([]any), id); entry != nil {
			files, _ := entry["files"].([]any)
			if len(files) > 0 && files[0].(map[string]any)["name"] == marker.Files[0] {
				caps := map[string]any{"context_max": nil, "mtp": false, "twin": false, "mmproj": false}
				for _, file := range files {
					role, _ := file.(map[string]any)["role"].(string)
					if role == "mtp" || role == "twin" || role == "mmproj" {
						caps[role] = true
					}
				}
				if estimate, ok := entry["estimate"].(map[string]any); ok {
					if inputs, ok := estimate["fit_inputs"].(map[string]any); ok {
						caps["context_max"] = inputs["native_context_tokens"]
					}
				}
				row["capabilities"] = caps
			}
		}
	}
	row["size_bytes"], row["prepared_state"], row["catalogue"] = size, state, catalogue
	return row
}

func catalogueRow(rows []any, id string) map[string]any {
	for _, r := range rows {
		if row, ok := r.(map[string]any); ok && row["id"] == id {
			return row
		}
	}
	return nil
}

// defaultCatalogueID is the entry the CLI treats as the default package.
func defaultCatalogueID(rows []any) string {
	for _, r := range rows {
		if row, ok := r.(map[string]any); ok && row["role"] == "recommended" && row["default_status"] == "default" {
			id, _ := row["id"].(string)
			return id
		}
	}
	return ""
}

// packagesModelsList answers `models list` from the disk.
func packagesModelsList() (string, map[string]any) {
	kind, payload := recordedOne("models-list-empty.json")
	root := flag("--models-root")
	if root == "" {
		root = defaultModelsRoot()
	}
	listing := payload["listing"].(map[string]any)
	listing["models_root"] = root
	if free, ok := scriptedFree(root); ok {
		listing["available_bytes"] = free
	}
	catalogue := payload["catalogue"].([]any)
	prepared := []any{}
	seen := map[string]bool{}
	for _, r := range catalogue {
		row := r.(map[string]any)
		id := row["id"].(string)
		folder := filepath.Join(root, id)
		state := packageState(folder)
		row["prepared_state"] = state
		row["local_folder"] = folder
		if state != "not_prepared" {
			m, _ := readPackageMarker(folder)
			prepared = append(prepared, preparedRow(id, row["display_name"].(string), folder, state, m.SizeBytes, true))
			seen[strings.ToLower(filepath.Join(folder, packageMarker))] = true
		}
	}
	scanned := []any{}
	if fileExists(root) {
		scanned = append(scanned, root)
	}
	warnings := []any{}
	for _, scan := range allFlags("--scan-folder") {
		scanned = append(scanned, scan)
		if !fileExists(scan) {
			warnings = append(warnings, map[string]any{"folder": scan, "reason": "The folder does not exist or cannot be read."})
			continue
		}
		for _, folder := range findPackageFolders(scan) {
			manifest := filepath.Join(folder, packageMarker)
			if seen[strings.ToLower(manifest)] {
				continue
			}
			seen[strings.ToLower(manifest)] = true
			m, _ := readPackageMarker(folder)
			display, isCatalogue := m.DisplayName, false
			if display == "" {
				display = m.PackageID
			}
			if entry := catalogueRow(catalogue, m.PackageID); entry != nil {
				display, isCatalogue = entry["display_name"].(string), true
			}
			prepared = append(prepared, preparedRow(m.PackageID, display, folder, packageState(folder), m.SizeBytes, isCatalogue))
		}
	}
	listing["scanned_folders"] = scanned
	listing["package_files_skipped"] = 0
	payload["prepared_models"] = prepared
	payload["scanned"] = []any{}
	payload["scan_warnings"] = warnings
	return kind, payload
}

// packagesEstimate answers `models estimate --model SUBJECT`: the recorded
// estimate of the package's id, with the subject of this test's folder.
func packagesEstimate() {
	subject := flag("--model")
	id, kind := "", "path"
	if catalogueRow(recordedCatalogue(), subject) != nil {
		id, kind = subject, "catalogue"
	} else {
		m, ok := readPackageMarker(subject)
		if !ok {
			printRecording("estimate-missing-folder.json")
			return
		}
		id = m.PackageID
	}
	var file string
	switch {
	case id == lowmemoryID && flag("--context") == "" && flag("--sessions") == "":
		file = "estimate-lowmemory-default.json"
	case kind == "catalogue" && id == flashID && flag("--context") == "" && flag("--sessions") == "":
		file = "estimate-flash-catalogue-id.json"
	case kind == "catalogue":
		missingRecording("models estimate for the catalogue id " + id)
	case estimatePrefix[id] == "":
		missingRecording("models estimate for a package with the id " + id)
	default:
		context, sessions := flag("--context"), flag("--sessions")
		file = "estimate-" + estimatePrefix[id] + "-default.json"
		if context != "" || sessions != "" {
			if context == "" {
				context = "200000"
			}
			if sessions == "" {
				sessions = "1"
			}
			file = "estimate-" + estimatePrefix[id] + "-c" + context + "-s" + sessions + ".json"
			if !hasRecording(file) {
				missingRecording("models estimate " + file)
			}
		}
	}
	eventKind, payload := recordedOne(file)
	// Hand-written: the subject names this test's folder, not the recording's.
	payload["subject"] = subject
	payload["subject_kind"] = kind
	event(eventKind, payload)
}

// selectedManifest is the model selection of status and serve: --manifest,
// BUBO_MANIFEST, the pointer, then the standard location of the default entry.
func selectedManifest() (manifest, pointer, source string) {
	pointer = filepath.Join(defaultDataRoot(), pointerFile)
	if m := flag("--manifest"); m != "" {
		return m, pointer, "explicit_manifest"
	}
	if flag("--model") != "" {
		return "", pointer, "explicit_model"
	}
	if m := os.Getenv("BUBO_MANIFEST"); m != "" {
		return m, pointer, "environment"
	}
	if b, err := os.ReadFile(pointer); err == nil {
		var p struct {
			ManifestPath string `json:"manifest_path"`
		}
		if json.Unmarshal(b, &p) == nil && p.ManifestPath != "" {
			return p.ManifestPath, pointer, "prepared_default"
		}
	}
	return filepath.Join(defaultModelsRoot(), defaultCatalogueID(recordedCatalogue()), packageMarker), pointer, "standard_location"
}

// packagesStatus rewrites the model fields of a status payload.
func packagesStatus(p map[string]any) {
	// status takes the same selection as serve (engine job E20, item 12): a
	// --manifest or --model reports readiness for that selection; with none it
	// reports what the environment and the pointer name.
	manifest, pointer, source := selectedManifest()
	folder := filepath.Dir(manifest)
	state := packageState(folder)
	if source == "explicit_model" {
		state = "not_prepared"
		if fi, err := os.Stat(flag("--model")); err == nil && fi.Mode().IsRegular() {
			state = "prepared"
		}
		folder = ""
	}
	ready := state == "prepared"
	p["prepared"] = ready
	if ready {
		p["phase"] = "stopped"
		p["last_error"] = nil
	} else {
		p["phase"] = "not_prepared"
		p["last_error"] = map[string]any{"code": "model_missing", "component": "model", "stage": "status", "retryable": false, "remedy": "Run bubo prepare model."}
	}
	p["readiness"] = map[string]any{"runtime": true, "model": ready}
	var manifestOut, modelOut any = manifest, nil
	if source == "explicit_model" {
		manifestOut, modelOut = nil, flag("--model")
	}
	p["model_selection"] = map[string]any{"model": modelOut, "manifest": manifestOut, "pointer": pointer, "source": source}
	if paths, ok := p["paths"].(map[string]any); ok {
		paths["manifest"] = manifestOut
		paths["prepared_default_pointer"] = pointer
	}
	p["prepared_default"] = manifestOut
	rows := []any{}
	if state != "not_prepared" && folder != "" {
		m, _ := readPackageMarker(folder)
		display := m.PackageID
		if entry := catalogueRow(recordedCatalogue(), m.PackageID); entry != nil {
			display = entry["display_name"].(string)
		}
		rows = append(rows, preparedRow(m.PackageID, display, folder, state, m.SizeBytes, true))
	}
	p["prepared_models"] = rows
}

// packagesServeCheck fails the way serve does for a missing or partial package.
func packagesServeCheck() {
	if flag("--model") != "" {
		return
	}
	manifest, _, _ := selectedManifest()
	if packageState(filepath.Dir(manifest)) != "prepared" {
		cliError("model_missing", "model", "startup", false, "Run bubo prepare model, or choose another model.", 12)
	}
}

// cataloguePackages is `prepare model --catalogue-id ID` on a real disk. The
// plan is the recorded plan of the id; hand-written: the model folder, the drive
// name and, when free.json names the drive, its free space.
func cataloguePackages() {
	recordArgv("prepare")
	id := flag("--catalogue-id")
	lines, exit := recordedPlan(id)
	if exit != 0 {
		emitRecordedPlan(lines, planDigestOf(lines), nil)
		os.Exit(exit)
	}
	rows := recordedCatalogue()
	entry := catalogueRow(rows, id)
	output := flag("--output")
	if output == "" {
		output = defaultDataRoot()
	}
	folder := filepath.Join(output, "models", id)
	size := uint64(entry["download_bytes"].(float64))
	digest := planDigestOf(lines)
	var required uint64
	adjust := func(payload map[string]any) {
		if paths, ok := payload["paths"].(map[string]any); ok {
			if _, has := paths["model"]; has {
				paths["model"] = folder
			}
		}
		volumes, _ := payload["volumes"].([]any)
		for _, v := range volumes {
			if volume, ok := v.(map[string]any); ok {
				volume["name"] = volumeOf(output)
				if free, scripted := scriptedFree(output); scripted {
					volume["free_bytes"] = free
				}
				if need, ok := volume["required_bytes"].(float64); ok {
					required = uint64(need)
				}
			}
		}
	}
	emitRecordedPlan(lines, digest, adjust)
	if hasFlag("--plan") {
		recordedPlanResult(lines)
		return
	}
	offline := hasFlag("--offline")
	if !offline {
		binding := approvalDigest(map[string]string{"model": digest})
		event("approval_required", map[string]any{"plan_digest": binding, "components": []string{"model"}})
		if !approve(binding) {
			return
		}
	}
	if !offline && required > 0 && volumeFree(output) < required {
		cliError("storage_failed", "model", "download", false, "The drive is full. Free space and start again.", 7)
	}
	phase := "download"
	if offline || hasFlag("--verify") {
		phase = "verify"
	}
	for n := 0; n <= 2; n++ {
		event("progress", map[string]any{"component": "model", "phase": phase, "file_id": "main", "bytes_done": n, "bytes_total": 2, "resume_offset": 0, "verification_method": "sha256"})
		time.Sleep(progressPause(20 * time.Millisecond))
	}
	state := packageState(folder)
	if offline {
		event("result", map[string]any{"exit_code": 0, "manifest_path": filepath.Join(folder, packageMarker), "prepared": state == "prepared"})
		return
	}
	if err := writePackage(folder, id, size); err != nil {
		cliError("storage_failed", "model", "download", false, "The folder cannot be written.", 7)
	}
	manifest := filepath.Join(folder, packageMarker)
	// The default entry records the machine-wide pointer, even with --output.
	if id == defaultCatalogueID(rows) {
		_ = os.MkdirAll(defaultDataRoot(), 0700)
		b, _ := json.Marshal(map[string]any{"schema_version": 1, "manifest_path": manifest})
		_ = os.WriteFile(filepath.Join(defaultDataRoot(), pointerFile), b, 0600)
	}
	event("result", map[string]any{"exit_code": 0, "manifest_path": manifest, "prepared": true})
}
