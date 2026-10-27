//go:build windows

package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"bubo/gui/supervisor"
)

func TestSharedScenarioRecordingsReplayThroughFakeCLI(t *testing.T) {
	previous := os.Args
	defer func() { os.Args = previous }()
	folder, err := filepath.Abs(filepath.Join("..", "scenarios", "recordings", "ready-busy"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_SCENARIO", folder)
	for _, args := range [][]string{{"status", "--json"}, {"serve", "--list-switches", "--json"}, {"models", "list", "--json"}} {
		os.Args = append([]string{"fakecli"}, args...)
		lines := capture(t, func() {
			if !recordedScenario() {
				t.Fatal("scenario not replayed")
			}
		})
		if len(lines) != 1 {
			t.Fatalf("records: %d", len(lines))
		}
		e, err := supervisor.ParseEvent([]byte(lines[0]))
		if err != nil {
			t.Fatal(err)
		}
		if e.Type == "status" {
			st := e.Value.(*supervisor.Status)
			if st.Sessions == nil || *st.Sessions != 1 || len(st.Slots) != 1 || st.Slots[0].State != "generating" {
				t.Fatal("fake replay differs from scenario")
			}
		}
	}
}

func TestSplitUTF8EventIsContractJSON(t *testing.T) {
	previous := os.Stdout
	oldMode := mode
	oldSeq := seq
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	mode = "split"
	seq = 0
	defer func() { os.Stdout = previous; mode = oldMode; seq = oldSeq; r.Close() }()
	oldInstance, oldCreation, oldExe := instance, creation, selfExe
	instance, creation, selfExe = "22222222222222222222222222222222", "134000000000000000", "C:/Bubo/bubo.exe"
	defer func() { instance, creation, selfExe = oldInstance, oldCreation, oldExe }()
	life("loading", map[string]any{"message": "é水"})
	w.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	e, err := supervisor.ParseEvent(b)
	if err != nil || e.Seq != 1 || e.Type != "loading" {
		t.Fatal(e, err)
	}
}
func TestCreationTimeUsesWindowsIdentity(t *testing.T) {
	if created() == "" || created() == "0" {
		t.Fatal("missing process creation time")
	}
}

// capture runs fn with stdout redirected and returns the JSON lines it wrote.
func capture(t *testing.T, fn func()) []string {
	t.Helper()
	previous := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	defer func() { os.Stdout = previous }()
	// A Windows pipe holds a few KB; read while fn writes.
	type read struct {
		b   []byte
		err error
	}
	done := make(chan read, 1)
	go func() { b, err := io.ReadAll(r); done <- read{b, err} }()
	fn()
	w.Close()
	got := <-done
	r.Close()
	if got.err != nil {
		t.Fatal(got.err)
	}
	b := got.b
	var lines []string
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return lines
}

func withModelEnv(t *testing.T, args ...string) {
	t.Helper()
	oldArgs, oldFixtures, oldDir, oldSeq, oldMode := os.Args, fixtures, dir, seq, mode
	abs, err := filepath.Abs(filepath.Join("..", "..", "..", "tests", "cli", "contract"))
	if err != nil {
		t.Fatal(err)
	}
	os.Args = append([]string{"bubo.exe"}, args...)
	fixtures, dir, seq, mode = abs, t.TempDir(), 0, ""
	t.Cleanup(func() { os.Args, fixtures, dir, seq, mode = oldArgs, oldFixtures, oldDir, oldSeq, oldMode })
}

// recordedFolder is the package folder of one prepared row of the recorded list.
func recordedFolder(t *testing.T, id string) string {
	t.Helper()
	_, list := recordedOne("models-list-gui.json")
	rows, _ := list["prepared_models"].([]any)
	for _, r := range rows {
		if row := r.(map[string]any); row["id"] == id {
			return filepath.Dir(row["manifest_path"].(string))
		}
	}
	t.Fatalf("the recorded list has no %s", id)
	return ""
}

func TestModelsListReplaysTheRecordedBytes(t *testing.T) {
	withModelEnv(t, "models", "list", "--json", "--scan-folder", `D:\more`)
	lines := capture(t, modelsCommand)
	if len(lines) != 1 {
		t.Fatalf("records: %d", len(lines))
	}
	if want := strings.TrimSpace(string(readRecording("models-list-gui.json"))); lines[0] != want {
		t.Fatal("the fake changed the recorded models list")
	}
	e, err := supervisor.ParseEvent([]byte(lines[0]))
	if err != nil || e.Type != "models" {
		t.Fatalf("%v %v", e.Type, err)
	}
	m := e.Value.(*supervisor.Models)
	_, list := recordedOne("models-list-gui.json")
	if len(m.Catalogue) != len(list["catalogue"].([]any)) || len(m.PreparedModels) != len(list["prepared_models"].([]any)) {
		t.Fatalf("rows: %d catalogue, %d prepared", len(m.Catalogue), len(m.PreparedModels))
	}
}

func TestModelsEstimateReplaysTheRecordedAnswers(t *testing.T) {
	withModelEnv(t, "models", "estimate") // sets the fixtures folder the recordings are found from
	flash, hauhau := recordedFolder(t, flashID), recordedFolder(t, hauhauID)
	withModelEnv(t, "models", "estimate", "--model", hauhau, "--json")
	line := capture(t, modelsCommand)[0]
	e, err := supervisor.ParseEvent([]byte(line))
	if err != nil || e.Type != "model_estimate" || e.Value.(*supervisor.ModelEstimate).State != "none_yet" {
		t.Fatalf("%v %v", e.Type, err)
	}
	withModelEnv(t, "models", "estimate", "--model", flash, "--json", "--context", "262144", "--sessions", "4")
	line = capture(t, modelsCommand)[0]
	if line != strings.TrimSpace(string(readRecording("estimate-flash-c262144-s4.json"))) {
		t.Fatal("the fake changed the recorded estimate")
	}
	e, err = supervisor.ParseEvent([]byte(line))
	if err != nil || e.Value.(*supervisor.ModelEstimate).State != "estimated" || e.Value.(*supervisor.ModelEstimate).Subject != flash {
		t.Fatalf("%v %v", e.Type, err)
	}
}

func TestCataloguePlanReplaysTheRecordedRecords(t *testing.T) {
	withModelEnv(t, "prepare", "model", "--catalogue-id", flashID, "--plan", "--jsonl")
	lines := capture(t, catalogue)
	var parts []supervisor.Plan
	var result *supervisor.Result
	for _, l := range lines {
		e, err := supervisor.ParseEvent([]byte(l))
		if err != nil {
			t.Fatalf("%s: %v", l, err)
		}
		switch v := e.Value.(type) {
		case *supervisor.Plan:
			parts = append(parts, *v)
		case *supervisor.Result:
			result = v
		}
	}
	want := 0
	for _, r := range recordedLines("plan-" + flashID + "-fresh.jsonl") {
		if r.Type == "plan" {
			want++
		}
	}
	if len(parts) != want || len(lines) != len(recordedLines("plan-"+flashID+"-fresh.jsonl")) || result == nil || !result.PlanOnly || result.Prepared {
		t.Fatalf("records: %d lines, %d plan parts, want %d", len(lines), len(parts), want)
	}
}

func TestDiffusionGemmaPublishedPlanUsesRecordedPins(t *testing.T) {
	withModelEnv(t, "prepare")
	file := "plan-diffusiongemma-26b-e38-abliterated-nvfp4-fresh.jsonl"
	if recordedExitCode(file) != 0 {
		t.Fatal("the published package must produce a successful preview")
	}
	last := recordedLines(file)[len(recordedLines(file))-1]
	if last.Type != "result" || last.Payload["plan_only"] != true || last.Payload["prepared"] != false {
		t.Fatalf("recorded preview: %v", last)
	}
}

func TestCatalogueRunStreamsProgressAndAPreparedResult(t *testing.T) {
	withModelEnv(t, "prepare", "model", "--catalogue-id", flashID, "--jsonl", "--offline", "--verify")
	lines := capture(t, catalogue)
	last, err := supervisor.ParseEvent([]byte(lines[len(lines)-1]))
	if err != nil || last.Type != "result" || !last.Value.(*supervisor.Result).Prepared {
		t.Fatalf("last: %v %v", last.Type, err)
	}
	progress := 0
	for _, l := range lines {
		if e, err := supervisor.ParseEvent([]byte(l)); err == nil && e.Type == "progress" {
			progress++
			if e.Value.(*supervisor.Progress).Phase != "verify" {
				t.Fatalf("phase: %+v", e.Value)
			}
		}
	}
	if progress != 5 {
		t.Fatalf("progress records: %d", progress)
	}
}
