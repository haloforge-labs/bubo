//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const s24Lowmemory = "huihui-qwen38-flash-next-lowmemory-v1"

func s24Recordings(t *testing.T) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("..", "tests", "scenarios", "recordings", "v12"))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestS24CatalogueAndEstimateFromCandidate(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(s24Recordings(t), "models-list-empty.json"))
	if err != nil {
		t.Fatal(err)
	}
	event, err := ParseEvent(data)
	if err != nil {
		t.Fatal(err)
	}
	models := event.Value.(*Models)
	if len(models.Catalogue) != 11 {
		t.Fatalf("catalogue: %d", len(models.Catalogue))
	}
	entry, ok := models.Entry(s24Lowmemory)
	if !ok || entry.DisplayName != "Qwen3.8 Flash Next · HaloForge Labs lower-memory quant (GSQ-RCO)" || !entry.DownloadEnabled || entry.DisabledReason != nil || entry.DownloadBytes == 0 || entry.Estimate.State != "estimated" || entry.Estimate.MemoryEstimate == nil {
		t.Fatalf("entry: %+v", entry)
	}
	var facts map[string]json.RawMessage
	if err := json.Unmarshal(entry.raw, &facts); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"speed", "stock_comparison"} {
		if string(facts[field]) != "null" {
			t.Fatalf("%s must be CLI null: %s", field, facts[field])
		}
	}
	data, err = os.ReadFile(filepath.Join(s24Recordings(t), "estimate-lowmemory-default.json"))
	if err != nil {
		t.Fatal(err)
	}
	event, err = ParseEvent(data)
	if err != nil {
		t.Fatal(err)
	}
	estimate := event.Value.(*ModelEstimate)
	if estimate.State != "estimated" || estimate.MemoryEstimate == nil {
		t.Fatalf("estimate: %+v", estimate)
	}
	result := ModelEstimateResultFrom(*estimate, nil)
	if result.Error != nil || result.Fit == nil || result.Fit.Class != "fits" {
		t.Fatalf("result: %+v", result)
	}
}

func TestS24PublishedDownloadReturnsRecordedPlanWithoutAcquisition(t *testing.T) {
	m := newModelsApp(t, "")
	t.Setenv("BUBO_FAKE_RECORDINGS", s24Recordings(t))
	m.waitModels(t, listed)
	result := m.a.Download(s24Lowmemory)
	if result.Plan == nil || result.Error != nil || result.Plan.ID != s24Lowmemory || len(result.Plan.Files) != 8 {
		t.Fatalf("published plan: %+v; error: %+v", result.Plan, result.Error)
	}
	time.Sleep(150 * time.Millisecond)
	calls, err := filepath.Glob(filepath.Join(m.dir, "prepare-argv-*.json"))
	if err != nil || len(calls) != 1 {
		t.Fatalf("prepare calls: %v, %v", calls, err)
	}
	if job := m.a.GetView().Models.Download; job != nil && (job.State == "running" || job.State == "planning") {
		t.Fatalf("job still active: %+v", job)
	}
}

func TestS24FakeEstimateUsesCandidateCalibration(t *testing.T) {
	m := newModelsApp(t, "")
	t.Setenv("BUBO_FAKE_RECORDINGS", s24Recordings(t))
	result := m.a.Estimate(s24Lowmemory, 0, 0)
	if result.Error != nil || result.Estimate == nil || result.Estimate.State != "estimated" || result.Estimate.MemoryEstimate == nil || result.Fit == nil || result.Fit.Class != "fits" {
		t.Fatalf("estimate: %+v, error: %+v", result.Estimate, result.Error)
	}
}

func TestS24LocalLowmemoryPackageIsReady(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(s24Recordings(t), "models-list-empty.json"))
	if err != nil {
		t.Fatal(err)
	}
	event, err := ParseEvent(data)
	if err != nil {
		t.Fatal(err)
	}
	entry, ok := event.Value.(*Models).Entry(s24Lowmemory)
	if !ok || len(entry.Files) != 8 {
		t.Fatal("real lower-memory catalogue files missing")
	}
	m := newModelsApp(t, "")
	t.Setenv("BUBO_FAKE_RECORDINGS", s24Recordings(t))
	t.Setenv("BUBO_FAKE_PACKAGES", "1")
	root := filepath.Join(m.dir, "local", "Bubo", "models", s24Lowmemory)
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	// Presence-only stubs exercise the fake transport, never model loading.
	// All displayed names, package bytes and capabilities come from the real CLI.
	files := make([]string, 0, len(entry.Files))
	for _, file := range entry.Files {
		files = append(files, file.Name)
		if err := os.WriteFile(filepath.Join(root, file.Name), nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	marker, err := json.Marshal(map[string]any{"package_id": entry.ID, "display_name": entry.DisplayName, "size_bytes": entry.DownloadBytes, "files": files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bubo-package.json"), marker, 0600); err != nil {
		t.Fatal(err)
	}
	if err := actionTestError(m.a.RefreshModels()); err != nil {
		t.Fatal(err)
	}
	s := m.waitModels(t, func(s *ViewModelsState) bool { return listed(s) && len(s.PreparedModels) > 0 })
	found := false
	for _, row := range s.PreparedModels {
		if row.ID == s24Lowmemory {
			found = true
			if row.PreparedState != "prepared" {
				t.Fatalf("row: %+v", row)
			}
			if row.SizeBytes != entry.DownloadBytes || row.DisplayName != entry.DisplayName || row.ModelPath == nil || filepath.Base(*row.ModelPath) != entry.Files[0].Name {
				t.Fatalf("local fixture lost the real CLI package identity, size or main file: %+v", row)
			}
			if row.Capabilities.ContextMax == nil || *row.Capabilities.ContextMax != 262144 || !row.Capabilities.MTP || !row.Capabilities.Twin || !row.Capabilities.MMProj {
				t.Fatalf("local fixture lost the catalogue's context and side-file capabilities: %+v", row.Capabilities)
			}
		}
	}
	if !found {
		t.Fatal("local lower-memory package missing")
	}
	result := m.a.Estimate(root, 0, 0)
	if result.Error != nil || result.Estimate == nil || result.Estimate.State != "estimated" || result.Estimate.MemoryEstimate == nil || result.Fit == nil || result.Fit.Class != "fits" {
		t.Fatalf("local estimate: %+v, error: %+v", result.Estimate, result.Error)
	}
}
