//go:build windows

package main

// Recorded answers (V12). Every model, estimate, plan, status and switches
// answer comes from the rebuilt merged CLI in recordings/v12. The historical
// E21/E23 captures remain as publication fixtures. metadata.json holds the commands, the
// exit codes and the exe hash). A field written by hand is a finding: the few
// that remain are named where they are set and listed in the E21 report.
//
// The recorded folders (rootA, rootB, Import) are neutral paths in the publication
// fixtures. The fake never invents an answer for a request that no recording
// covers: it fails with a message that names the missing recording.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// recordingsDir is BUBO_FAKE_RECORDINGS, else the v12 folder of this repository
// (BUBO_FAKE_FIXTURES is <repo>/tests/cli/contract), else the first v12 folder
// found above the working folder or above this executable.
func recordingsDir() string {
	if v := os.Getenv("BUBO_FAKE_RECORDINGS"); v != "" {
		return v
	}
	const tail = "gui/tests/scenarios/recordings/v12"
	if fixtures != "" {
		return filepath.Join(fixtures, "..", "..", "..", filepath.FromSlash(tail))
	}
	starts := []string{}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	for _, start := range starts {
		for folder, n := start, 0; n < 10; folder, n = filepath.Dir(folder), n+1 {
			candidate := filepath.Join(folder, filepath.FromSlash(tail))
			if _, err := os.Stat(filepath.Join(candidate, "metadata.json")); err == nil {
				return candidate
			}
		}
	}
	return ""
}

func missingRecording(what string) {
	cliError("unexpected_failure", "fake", "recording", false, "The fake CLI has no recording of "+what+". Add the command to the capture-v12.ps1 recorder and record it from the candidate exe.", 1)
}

// readRecording returns the stdout bytes of one recorded command.
func readRecording(file string) []byte {
	folder := recordingsDir()
	if folder == "" {
		missingRecording(file + " (BUBO_FAKE_FIXTURES and BUBO_FAKE_RECORDINGS are not set, and no v12 folder is above here)")
	}
	b, err := os.ReadFile(filepath.Join(folder, file))
	if err != nil {
		missingRecording(file)
	}
	return b
}

func hasRecording(file string) bool {
	_, err := os.Stat(filepath.Join(recordingsDir(), file))
	return err == nil
}

// recordedExitCode is the exit code metadata.json holds for a recorded file.
func recordedExitCode(file string) int {
	var meta struct {
		Commands []struct {
			File     string `json:"file"`
			ExitCode int    `json:"exit_code"`
		} `json:"commands"`
	}
	if json.Unmarshal(readRecording("metadata.json"), &meta) != nil {
		missingRecording("metadata.json")
	}
	for _, c := range meta.Commands {
		if c.File == file {
			return c.ExitCode
		}
	}
	return 0
}

// printRecording writes the recorded bytes unchanged and ends the process with
// the recorded exit code.
func printRecording(file string) {
	b := readRecording(file)
	os.Stdout.Write(b)
	if !bytes.HasSuffix(b, []byte("\n")) {
		os.Stdout.WriteString("\n")
	}
	if code := recordedExitCode(file); code != 0 {
		os.Exit(code)
	}
}

type recordedLine struct {
	Type    string
	Payload map[string]any
}

// recordedLines parses a recorded .json or .jsonl file into its records.
func recordedLines(file string) []recordedLine {
	var out []recordedLine
	for _, l := range bytes.Split(readRecording(file), []byte("\n")) {
		l = bytes.TrimSpace(l)
		if len(l) == 0 {
			continue
		}
		var e struct {
			Type    string         `json:"type"`
			Payload map[string]any `json:"payload"`
		}
		if err := json.Unmarshal(l, &e); err != nil {
			missingRecording(file + " (it is not JSON)")
		}
		out = append(out, recordedLine{e.Type, e.Payload})
	}
	return out
}

// recordedOne returns the first record of a recorded file.
func recordedOne(file string) (string, map[string]any) {
	lines := recordedLines(file)
	if len(lines) == 0 {
		missingRecording(file)
	}
	return lines[0].Type, lines[0].Payload
}

// Short names of the recorded packages (the files are named by them).
const (
	flashID     = "huihui-qwen38-flash-next-ud-q4-k-xl-v1"
	q27ID       = "qwen38-27b-ud-q4-k-xl"
	hauhauID    = "gemma4-26b-a4b-uncensored-hauhaucs-qat-q4-k-m"
	stockID     = "gemma4-26b-a4b-stock-unsloth-ud-q4-k-m"
	q36ID       = "qwen36-27b-ud-q4-k-xl"
	lowmemoryID = "huihui-qwen38-flash-next-lowmemory-v1"
)

// estimatePrefix names the estimate files of each recorded package.
var estimatePrefix = map[string]string{flashID: "flash", q27ID: "q27", hauhauID: "hauhau", stockID: "importpkg", q36ID: "partial"}

// recordedProfileHasModel: the fake profile holds the default package. The ui
// modes start without it until a prepare run ends (the "prepared" file).
func recordedProfileHasModel() bool {
	return !(mode == "ui" || mode == "ui-fail") || fakePrepared()
}

// recordedStatus is the recorded `status --json` of the profile with the
// default model (status-main) or of the empty profile (status-empty).
func recordedStatus(hasModel bool) map[string]any {
	file := "status-main.json"
	if !hasModel {
		file = "status-empty.json"
	}
	_, payload := recordedOne(file)
	return payload
}

// recordedListFile is the recorded `models list` that answers this profile:
// the GUI's own call (models-list-gui) or the empty profile's list.
func recordedListFile() string {
	if v := os.Getenv("BUBO_FAKE_MODELS_RECORDING"); v != "" {
		return v
	}
	if recordedProfileHasModel() {
		return "models-list-gui.json"
	}
	return "models-list-empty.json"
}

func sameFolder(a, b string) bool {
	return strings.EqualFold(strings.TrimRight(filepath.Clean(a), `\/`), strings.TrimRight(filepath.Clean(b), `\/`))
}

// recordedEstimateFile finds the recorded estimate that answers `models
// estimate --model SUBJECT [--context N] [--sessions N]`. The subject is a
// catalogue id, a package folder or a loose model file of the recorded list.
func recordedEstimateFile(subject, context, sessions string) string {
	_, list := recordedOne("models-list-gui.json")
	prefix := ""
	if catalogue, _ := list["catalogue"].([]any); catalogue != nil {
		for _, r := range catalogue {
			if row, ok := r.(map[string]any); ok && row["id"] == subject {
				if subject == lowmemoryID && context == "" && sessions == "" {
					return "estimate-lowmemory-default.json"
				}
				if subject == flashID && (context == "" || context == "200000") && (sessions == "" || sessions == "1") {
					return "estimate-flash-catalogue-id.json"
				}
				missingRecording(fmt.Sprintf("models estimate for the catalogue id %s with context %q sessions %q", subject, context, sessions))
			}
		}
	}
	rows, _ := list["prepared_models"].([]any)
	for _, r := range rows {
		row, _ := r.(map[string]any)
		manifest, _ := row["manifest_path"].(string)
		if manifest != "" && sameFolder(filepath.Dir(manifest), subject) {
			prefix = estimatePrefix[row["id"].(string)]
		}
	}
	if prefix == "" {
		scanned, _ := list["scanned"].([]any)
		for _, r := range scanned {
			if row, _ := r.(map[string]any); row != nil && sameFolder(fmt.Sprint(row["path"]), subject) {
				prefix = "loose"
			}
		}
	}
	if prefix == "" {
		// The real CLI's answer for a folder it cannot find.
		return "estimate-missing-folder.json"
	}
	if context == "" && sessions == "" {
		return "estimate-" + prefix + "-default.json"
	}
	if context == "" {
		context = "200000"
	}
	if sessions == "" {
		sessions = "1"
	}
	file := fmt.Sprintf("estimate-%s-c%s-s%s.json", prefix, context, sessions)
	if !hasRecording(file) {
		missingRecording(fmt.Sprintf("models estimate for %s with context %s and sessions %s (%s)", subject, context, sessions, file))
	}
	return file
}

// recordedPlanFile is the recorded `prepare model --catalogue-id ID --plan
// --jsonl` of an empty download folder; an id that is not in the catalogue
// gets the CLI's own answer for an unknown id.
func recordedPlanFile(id string) string {
	file := "plan-" + id + "-fresh.jsonl"
	if id != "" && hasRecording(file) {
		return file
	}
	return "plan-disabled-or-unknown.jsonl"
}

// recordedCatalogue is the catalogue of the recorded empty profile: every row
// as the CLI lists it, with nothing prepared.
func recordedCatalogue() []any {
	_, list := recordedOne("models-list-empty.json")
	rows, _ := list["catalogue"].([]any)
	return rows
}
