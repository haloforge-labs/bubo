//go:build windows

package supervisor

import "testing"

// `status --model <file>` reports the default package as missing (readiness.model
// false, last_error model_missing). A saved model file does not use that package,
// so the sample must not block a start; the default package and a real failed
// start still do.
func TestSavedModelFileIsNotBlockedByTheDefaultPackageSample(t *testing.T) {
	status := func() *Status {
		return &Status{
			Readiness: Readiness{Runtime: true},
			LastError: &CLIError{Code: "model_missing", Component: "model", Stage: "status", Remedy: "Run bubo prepare model."},
		}
	}
	input := func(saved Settings, failure *Failure) ViewInput {
		return ViewInput{Saved: saved, Snapshot: Snapshot{Initialized: true, Prepared: true, State: Stopped, Status: status(), LastError: failure}}
	}
	file := Settings{ModelPath: `S:\models\file.gguf`}

	if got := savedSetupAvailability(input(file, nil)); got.ReasonCode == "model_missing" {
		t.Errorf("a saved model file was blocked by the default package sample: %+v", got)
	}
	if got := savedSetupAvailability(input(Settings{}, nil)); got.Allowed || got.ReasonCode != "model_missing" {
		t.Errorf("no saved model must still report model_missing: %+v", got)
	}
	manifest := Settings{ModelPath: `S:\models\file.gguf`, ManifestPath: `S:\models\bubo-package.json`}
	if got := savedSetupAvailability(input(manifest, nil)); got.ReasonCode != "model_missing" {
		t.Errorf("a package manifest keeps the CLI's verdict: %+v", got)
	}
	failed := &Failure{Code: "model_missing", Source: "cli", Remedy: "The model file is missing."}
	if got := savedSetupAvailability(input(file, failed)); got.Allowed || got.ReasonCode != "model_missing" {
		t.Errorf("a real failed start must still block: %+v", got)
	}
}

// The Stopped page for a saved model file: the status sample's model_missing
// must not become the page's reason and remedy (the "No model is installed"
// banner). The default package and a saved package manifest keep it.
func TestStoppedViewOfASavedModelFileHasNoModelMissingReason(t *testing.T) {
	build := func(saved Settings) View {
		var builder ViewBuilder
		return builder.Build(ViewInput{Saved: saved, Snapshot: Snapshot{
			Initialized: true, Prepared: true, State: Stopped,
			Status: &Status{
				Readiness: Readiness{Runtime: true},
				LastError: &CLIError{Code: "model_missing", Component: "model", Stage: "status", Remedy: "Run bubo prepare model."},
			},
		}})
	}
	if v := build(Settings{ModelPath: `S:\models\file.gguf`}); v.Reason != "" || v.Remedy != "" || v.Failure != nil {
		t.Errorf("a saved model file shows the default package's error: reason %q remedy %q failure %+v", v.Reason, v.Remedy, v.Failure)
	}
	if v := build(Settings{}); v.Reason != "model_missing" || v.Remedy == "" {
		t.Errorf("no saved model keeps the CLI verdict: reason %q remedy %q", v.Reason, v.Remedy)
	}
	if v := build(Settings{ModelPath: `S:\m.gguf`, ManifestPath: `S:\p\bubo-package.json`}); v.Reason != "model_missing" {
		t.Errorf("a package manifest keeps the CLI verdict: reason %q", v.Reason)
	}
}
