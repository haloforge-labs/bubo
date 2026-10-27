package supervisor

import "testing"

func TestJobProgressAddsEarlierFilesToTheCurrentOne(t *testing.T) {
	files := planJobFiles(ModelPlan{Kind: "model", Files: []ModelPlanFile{
		{FileID: "a", Size: 100},
		{FileID: "b", Size: 400},
		{FileID: "c", Size: 500},
	}})
	got := jobProgress(files, "model", "b", "download", 150)
	if got == nil || got.BytesDone != 250 || got.BytesTotal != 1000 {
		t.Fatalf("whole job = %+v, want 250 of 1000", got)
	}
	// A reported byte count past the planned size cannot pass the next file.
	got = jobProgress(files, "model", "a", "download", 900)
	if got == nil || got.BytesDone != 100 || got.BytesTotal != 1000 {
		t.Fatalf("clamped = %+v, want 100 of 1000", got)
	}
}

func TestJobProgressFromPlanPartsSkipsReusedFilesAndOrdersTheEngineFirst(t *testing.T) {
	// The CLI's shape: a header (part 0) with no sources, one source per other
	// record, and every record repeating the header's file_states.
	states := []string{"download", "reuse"}
	parts := []Plan{
		{Component: "model", Part: 2, Parts: 3, Sources: []Source{{FileID: "m2", Size: 3000}}, FileStates: states},
		{Component: "model", Part: 0, Parts: 3, FileStates: states},
		{Component: "model", Part: 1, Parts: 3, Sources: []Source{{FileID: "m1", Size: 1000}}, FileStates: states},
		{Component: "runtime", Part: 0, Parts: 2},
		{Component: "runtime", Part: 1, Parts: 2, Sources: []Source{{FileID: "r1", Size: 200}}},
	}
	got := jobProgress(partJobFiles(parts), "model", "m1", "download", 500)
	if got == nil || got.BytesDone != 700 || got.BytesTotal != 1200 {
		t.Fatalf("combined job = %+v, want 700 of 1200 (the reused file is not counted)", got)
	}
	// The engine file ends the first third of the job.
	got = jobProgress(partJobFiles(parts), "runtime", "r1", "download", 50)
	if got == nil || got.BytesDone != 50 || got.BytesTotal != 1200 {
		t.Fatalf("engine file = %+v, want 50 of 1200", got)
	}
}

func TestJobProgressUnknownFile(t *testing.T) {
	files := planJobFiles(ModelPlan{Kind: "runtime", Files: []ModelPlanFile{{FileID: "r1", Size: 200}, {FileID: "r2", Size: 300}}})
	if got := jobProgress(files, "runtime", "unpack/x.dll", "verify", 10); got != nil {
		t.Fatalf("an unknown file must give no whole-job number, got %+v", got)
	}
	// Unpacking starts after every download has finished.
	got := jobProgress(files, "runtime", "unpack/x.dll", "extract", 10)
	if got == nil || got.BytesDone != 500 || got.BytesTotal != 500 {
		t.Fatalf("extract = %+v, want 500 of 500", got)
	}
	if got := jobProgress(nil, "runtime", "r1", "download", 1); got != nil {
		t.Fatalf("no plan must give no number, got %+v", got)
	}
}
