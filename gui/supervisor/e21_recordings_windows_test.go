//go:build windows

package supervisor

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)





const (
	e21Flash    = "huihui-qwen38-flash-next-ud-q4-k-xl-v1"
	e21Hauhau   = "gemma4-26b-a4b-uncensored-hauhaucs-qat-q4-k-m"
	e21Disabled = "diffusiongemma-26b-e38-abliterated"
)

func e21Dir() string {
	return filepath.Join(contractDir(), "..", "..", "..", "gui", "tests", "scenarios", "recordings", "v12")
}

func e21Events(t *testing.T, file string) []Event {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(e21Dir(), file))
	if err != nil {
		t.Fatal(err)
	}
	var out []Event
	for _, line := range bytes.Split(b, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		e, err := ParseEvent(line)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		out = append(out, e)
	}
	return out
}

// e21Models is the recorded list the fake answers with.
func e21Models(t *testing.T) *Models {
	t.Helper()
	return e21Events(t, "models-list-gui.json")[0].Value.(*Models)
}

// e21Folder is the package folder of one prepared row of the recorded list.
func e21Folder(t *testing.T, id string) string {
	t.Helper()
	for _, row := range e21Models(t).PreparedModels {
		if row.ID == id && row.ManifestPath != nil {
			return filepath.Dir(*row.ManifestPath)
		}
	}
	t.Fatalf("the recorded list has no prepared package %s", id)
	return ""
}

// e21Plan is the plan the recorded plan records of one catalogue id assemble to.
func e21Plan(t *testing.T, id string) ModelPlan {
	t.Helper()
	var parts []Plan
	for _, e := range e21Events(t, "plan-"+id+"-fresh.jsonl") {
		if p, ok := e.Value.(*Plan); ok {
			parts = append(parts, *p)
		}
	}
	plan, err := assemblePlan(parts, "model")
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

// e21PromptLine is the licence prompt line of a catalogue row of the recorded list.
func e21PromptLine(t *testing.T, id string) string {
	t.Helper()
	entry, ok := e21Models(t).Entry(id)
	if !ok || entry.Licence.PromptLine == "" {
		t.Fatalf("the recorded list has no licence line for %s", id)
	}
	return entry.Licence.PromptLine
}

// The CLI writes one state per source in the plan header (part 0) and repeats
// that whole list in every other record of the plan. The ways this can go wrong:
//   - the lists of all records are joined, the count no longer matches the
//     files, and every state is dropped (the consent then says "Planned files"
//     and the job counts reused files; seen in walk W);
//   - the states are paired with the wrong files;
//   - the records arrive in another order and the pairing moves;
//   - a header list of the wrong length is paired anyway.
//
// The recording is a package with one file present and one to download.
func TestE21PlanStatesComeFromTheHeader(t *testing.T) {
	var parts []Plan
	for _, e := range e21Events(t, "plan-qwen36-rootB-partial.jsonl") {
		if p, ok := e.Value.(*Plan); ok {
			parts = append(parts, *p)
		}
	}
	var header []string
	for _, p := range parts {
		if p.Part == 0 {
			header = p.FileStates
		}
	}
	reused := 0
	for _, s := range header {
		if s == "reuse" {
			reused++
		}
	}
	if reused == 0 || reused == len(header) {
		t.Fatalf("the recording should mix reused and downloaded files, has %v", header)
	}
	// The sources as sent, in order, take the header's states in order.
	bySource := map[string]string{}
	for _, p := range parts {
		for _, s := range p.Sources {
			bySource[s.FileID] = header[len(bySource)]
		}
	}
	reversed := append([]Plan(nil), parts...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	for name, order := range map[string][]Plan{"as sent": parts, "reversed": reversed} {
		plan, err := assemblePlan(order, "model")
		if err != nil {
			t.Fatal(err)
		}
		if len(plan.Files) != len(header) {
			t.Fatalf("%s: %d files for %d states", name, len(plan.Files), len(header))
		}
		var download uint64
		var downloading string
		for _, f := range plan.Files {
			if f.State != bySource[f.FileID] {
				t.Fatalf("%s: %s has state %q, the header says %q", name, f.FileID, f.State, bySource[f.FileID])
			}
			if f.State != "reuse" {
				download += f.Size
				downloading = f.FileID
			}
		}
		// The whole-job line counts only the files the CLI transfers.
		got := jobProgress(partJobFiles(order), "model", downloading, "download", 0)
		if got == nil || got.BytesTotal != download {
			t.Fatalf("%s: job = %+v, want a total of %d (reused files not counted)", name, got, download)
		}
	}
	// A header list that does not match the sources pairs nothing.
	short := append([]Plan(nil), parts...)
	for i := range short {
		short[i].FileStates = header[:1]
	}
	plan, err := assemblePlan(short, "model")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range plan.Files {
		if f.State != "" {
			t.Fatalf("a short state list was paired: %s = %q", f.FileID, f.State)
		}
	}
}
