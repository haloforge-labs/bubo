//go:build windows

package supervisor

import (
	"strings"
	"testing"
)

// E21w W4. The CLI answers `models estimate` for a Gemma 4 package with
// estimate_state none_yet and memory_estimate null (recording
// estimate-hauhau-default.json). That is a finished answer with no verdict, the
// same as a Qwen header estimate, but buildFit(nil) kept the in-flight line
// "Checking memory…" for ever, on the Installed row and in Server.
//
// Ways this could fail, written before the code:
//
//	G1 the Installed row (the Estimate binding) says "Checking memory…" for a
//	   finished none_yet answer.
//	G2 the Server view (saved setup) says it, when the saved estimate ended with
//	   estimate_no_verdict.
//	G3 the line is the CLI's reason, which holds developer words ("pointers",
//	   "engine code").
//	G4 the fix hides the real in-flight state: an estimate still being fetched
//	   (no answer, no error) must keep "Checking memory…".
//	G5 a Qwen row, which shows the no-verdict sentence, changes; or a row that has
//	   a verdict loses it.
//	G6 the two places disagree with each other or with the Qwen row.
//	G7 an answer with a real error (the CLI failed) is turned into a verdict.
func e21NoneYet(t *testing.T) ModelEstimate {
	t.Helper()
	est := e21Events(t, "estimate-hauhau-default.json")[0].Value.(*ModelEstimate)
	if est.State != "none_yet" || est.MemoryEstimate != nil || est.Reason == nil {
		t.Fatalf("the recording is not a none_yet answer: %+v", est)
	}
	return *est
}

func e21QwenHeader(t *testing.T) ModelEstimate {
	t.Helper()
	est := e21Events(t, "estimate-q27-default.json")[0].Value.(*ModelEstimate)
	if est.State != "estimated" || est.MemoryEstimate == nil {
		t.Fatalf("the recording is not an estimate: %+v", est)
	}
	return *est
}

func TestFinishedEstimateWithNoVerdictShowsTheNoVerdictSentence(t *testing.T) {
	qwen := ModelEstimateResultFrom(e21QwenHeader(t), nil)
	if qwen.Fit == nil || qwen.Fit.Sentence != NoVerdictSentence {
		t.Fatalf("the Qwen row is the reference and it changed: %+v", qwen.Fit)
	}
	none := e21NoneYet(t)
	got := ModelEstimateResultFrom(none, nil) // G1
	if got.Error != nil || got.Fit == nil {
		t.Fatalf("%+v", got)
	}
	if got.Fit.Sentence != qwen.Fit.Sentence || got.Fit.Class != qwen.Fit.Class { // G5 G6
		t.Fatalf("Gemma row %+v, Qwen row %+v", got.Fit, qwen.Fit)
	}
	if strings.Contains(got.Fit.Sentence, "Checking") {
		t.Fatalf("G1: %+v", got.Fit)
	}
	// G3: nothing of the CLI's reason reaches the sentence.
	for _, word := range strings.Fields(*none.Reason) {
		if len(word) > 6 && strings.Contains(got.Fit.Sentence, word) {
			t.Fatalf("G3: the sentence has %q from the CLI reason", word)
		}
	}
}

func TestServerViewOfASavedSetupWithNoVerdictShowsTheSameSentence(t *testing.T) {
	none := e21NoneYet(t)
	in := actionTestInput(Stopped)
	in.Saved.ModelPath = `D:\models\gemma4\gemma4-26b.gguf`
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate = nil
	in.SavedEstimateError = codedAction("estimate_no_verdict", *none.Reason).Error
	v := (&ViewBuilder{}).Build(in)
	if v.Fit.Sentence != NoVerdictSentence || v.Fit.Class != "fit_unknown" { // G2 G6
		t.Fatalf("%+v", v.Fit)
	}
	if strings.Contains(v.Fit.Sentence, *none.Reason) { // G3
		t.Fatal("the CLI reason is the sentence")
	}
	// The note on Start is unchanged by this fix.
	if !v.Actions.Start.Allowed {
		t.Fatalf("Start: %+v", v.Actions.Start)
	}
}

func TestEstimateStillBeingFetchedKeepsTheCheckingLine(t *testing.T) { // G4
	in := actionTestInput(Stopped)
	in.Saved.ModelPath = `D:\models\gemma4\gemma4-26b.gguf`
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimate, in.SavedEstimateError = nil, nil
	v := (&ViewBuilder{}).Build(in)
	if v.Fit.Class != "unavailable" || v.Fit.Sentence != "Checking memory…" {
		t.Fatalf("%+v", v.Fit)
	}
}

func TestEstimateErrorIsNotAVerdict(t *testing.T) { // G7
	in := actionTestInput(Stopped)
	in.Saved.ModelPath = `D:\models\gemma4\gemma4-26b.gguf`
	in.Snapshot.Status.MemoryEstimate = nil
	in.SavedEstimateError = codedAction("estimate_needs_restart", "x").Error
	v := (&ViewBuilder{}).Build(in)
	if v.Fit.Class == "fit_unknown" || v.Fit.Class == "fits" {
		t.Fatalf("an estimate that could not be asked is no verdict: %+v", v.Fit)
	}
	got := ModelEstimateResultFrom(ModelEstimate{}, protocol("boom"))
	if got.Error == nil || got.Fit != nil {
		t.Fatalf("%+v", got)
	}
}
