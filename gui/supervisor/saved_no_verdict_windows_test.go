//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A package without a measured memory table (Gemma 4, Qwen 27B, Cyber-Tiel,
// DiffusionGemma) has no fit verdict: `models estimate` answers none_yet with
// no memory_estimate (E11 output after-gemma31-models-estimate-json). Start
// stays allowed and the server checks memory when it starts; exit 19 reports
// a refusal. The View never says "Fits." for it.
func TestSavedCatalogueModelWithNoFitVerdictIsStartable(t *testing.T) {
	m := newModelsApp(t, "")
	m.b.mu.Lock()
	m.b.view = Snapshot{State: Stopped, Initialized: true, Prepared: true,
		Status: &Status{Readiness: Readiness{Runtime: true, Model: true}}}
	m.b.mu.Unlock()
	m.a.mu.Lock()
	// The recorded Gemma 4 package: the candidate answered none_yet for it.
	m.a.settings.ManifestPath = filepath.Join(e21Folder(t, e21Hauhau), "bubo-package.json")
	m.a.mu.Unlock()
	var last View
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		last = m.a.GetView()
		if last.SavedEstimate.Error != nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if last.SavedEstimate.Error == nil {
		t.Fatalf("the estimate never answered: %+v", last.SavedEstimate)
	}
	if last.SavedEstimate.Error.Code != "estimate_no_verdict" {
		t.Fatalf("a package with no verdict is not an estimate failure: %+v", last.SavedEstimate.Error)
	}
	if !last.Actions.Start.Allowed {
		t.Fatalf("Start is blocked for a model with no fit verdict: %+v", last.Actions.Start)
	}
	if last.Fit.Sentence == "Fits." || last.Fit.Class == "fits" {
		t.Fatalf("a verdict that does not exist: %+v", last.Fit)
	}
	if last.Actions.Start.Reason == "" {
		t.Fatal("Start needs its note: the server checks memory when it starts")
	}
}

// The CLI may answer with a memory estimate that carries no verdict (fits is
// null): the same rule. Start is allowed with the note; no "Fits." line.
func TestSavedEstimateWithNullFitsIsStartableWithTheNote(t *testing.T) {
	in := actionTestInput(Stopped)
	in.Saved.ModelPath = `D:\models\gemma4\gemma4-31b-main.gguf`
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate = &MemoryEstimate{ContextTokens: 65536, Sessions: 1}
	v := (&ViewBuilder{}).Build(in)
	if !v.Actions.Start.Allowed || v.Actions.Start.ReasonCode != "estimate_unavailable" || v.Actions.Start.Reason == "" {
		t.Fatalf("Start: %+v", v.Actions.Start)
	}
	if v.Fit.Sentence == "Fits." || v.Fit.Class == "fits" {
		t.Fatalf("a verdict that does not exist: %+v", v.Fit)
	}
}

// M2 (admission policy): a header estimate that says fits:true is a lower bound
// (fit_scope startup_lower_bound), and serve's load check can still refuse
// (exit 19). Only a conservative startup estimate may say "Fits.".
const noVerdictLine = "Not checked: you can still start the server."

func q38Estimate(t *testing.T) *MemoryEstimate {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "q38-memory-estimate.json"))
	if err != nil {
		t.Fatal(err)
	}
	var e MemoryEstimate
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	return &e
}

func TestRealQwen38LowerBoundFitsIsNoVerdictAndStartable(t *testing.T) {
	e := q38Estimate(t) // the real `models estimate` output of E11: fits true, startup_lower_bound
	if e.Fits == nil || !*e.Fits || e.FitScope != "startup_lower_bound" {
		t.Fatalf("the fixture is not the real lower-bound output: %+v", e.FitScope)
	}
	in := actionTestInput(Stopped)
	in.Saved.ModelPath = `D:\models\qwen38-27b\main.gguf`
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate = e
	v := (&ViewBuilder{}).Build(in)
	if !v.Actions.Start.Allowed || v.Actions.Start.ReasonCode != "estimate_unavailable" {
		t.Fatalf("Start: %+v", v.Actions.Start)
	}
	if v.Fit.Sentence != noVerdictLine || v.Fit.Class != "fit_unknown" || strings.Contains(v.Fit.Sentence, "Fits") {
		t.Fatalf("a lower bound is not a verdict: %+v", v.Fit)
	}
}

func TestConservativeStartupEstimateKeepsFits(t *testing.T) {
	top := 12.3
	est := fitEstimate(&top, nil, 40.0)
	est.FitScope = "conservative_startup_estimate"
	if got := buildFit(est); got.Sentence != "Fits. 12 GiB of memory would be left." || got.Class != "fits" {
		t.Fatalf("Flash Next lost its verdict: %+v", got)
	}
	in := actionTestInput(Stopped)
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate = est
	if v := (&ViewBuilder{}).Build(in); !v.Actions.Start.Allowed || v.Actions.Start.Reason != "" {
		t.Fatalf("Start: %+v", v.Actions.Start)
	}
}

func TestLowerBoundThatDoesNotFitStaysBlocked(t *testing.T) {
	e := q38Estimate(t)
	no := false
	e.Fits = &no
	in := actionTestInput(Stopped)
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate = e
	v := (&ViewBuilder{}).Build(in)
	if v.Actions.Start.Allowed || v.Actions.Start.ReasonCode == "" {
		t.Fatalf("a lower bound that does not fit cannot fit: %+v", v.Actions.Start)
	}
	if v.Fit.Sentence != doesNotFitExplanation || v.Fit.Class != "memory_does_not_fit" {
		t.Fatalf("fit: %+v", v.Fit)
	}
}

// A catalogue model selected by its manifest (the Models page "Use" route) is
// not the package the CLI reports; its memory cannot be estimated before the
// server starts. That is the plain "checked at start" line, never a failure.
// E19: a package chosen by its manifest is estimated by its folder, wherever
// that folder is. (Before E19 the estimate was refused with
// estimate_needs_restart; the folder estimate replaces that refusal.)
func TestSelectedPackageByManifestIsEstimatedByItsFolderAndStartStaysOpen(t *testing.T) {
	m := newModelsApp(t, "")
	def, other := `D:\models\default\bubo-package.json`, `D:\models\gemma4\bubo-package.json`
	m.b.mu.Lock()
	m.b.view = Snapshot{State: Stopped, Initialized: true, Prepared: true,
		Status: &Status{Readiness: Readiness{Runtime: true, Model: true}, PreparedDefault: &def}}
	m.b.mu.Unlock()
	m.a.mu.Lock()
	m.a.settings.ManifestPath = other
	m.a.mu.Unlock()
	last := waitSavedEstimate(t, m.a)
	if last.SavedEstimate.Error != nil && last.SavedEstimate.Error.Code == "estimate_needs_restart" {
		t.Fatalf("a package in another folder was refused an estimate: %+v", last.SavedEstimate.Error)
	}
	estimates := recordedEstimates(t, m.dir)
	if len(estimates) != 1 || estimates[0][3] != filepath.Dir(other) {
		t.Fatalf("estimate arguments: %v", estimates)
	}
	if !last.Actions.Start.Allowed {
		t.Fatalf("Start: %+v", last.Actions.Start)
	}
}
