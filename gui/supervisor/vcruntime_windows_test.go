//go:build windows

package supervisor

import (
	"testing"
	"time"
)

// E14 item 7: a CLI that Windows cannot start (missing Visual C++ runtime)
// exits with 0xC0000135 and prints nothing. That has its own code.
func TestMissingVCRuntimeIsNamed(t *testing.T) {
	s := supervisorFor(t, "no-vcruntime")
	v := waitFor(t, s, "the discovery failure", 10*time.Second, func(v Snapshot) bool { return v.Initialized && v.LastError != nil })
	if v.LastError.Code != "vcredist_missing" {
		t.Fatalf("want vcredist_missing, got %+v", v.LastError)
	}
}

func TestOtherSilentExitsKeepTheirCode(t *testing.T) {
	if got := errorCode(cliExitError(3)); got == "vcredist_missing" {
		t.Fatalf("exit 3 is not a missing runtime: %s", got)
	}
}

// UX10 M2: when every CLI call fails with 0xC0000135 the app knows nothing
// about the engine, so the View makes no readiness claim. It shows the
// runtime failure only, and offers no engine or model download.
func TestUnrunnableCLIMakesNoReadinessClaim(t *testing.T) {
	s := supervisorFor(t, "no-vcruntime")
	v := waitFor(t, s, "the discovery failure", 10*time.Second, func(v Snapshot) bool { return v.Initialized && v.LastError != nil })
	view := (&ViewBuilder{}).Build(ViewInput{Snapshot: v})
	if view.Failure == nil || view.Failure.Code != "vcredist_missing" || view.Phase != "failed" {
		t.Fatalf("phase %s failure %+v", view.Phase, view.Failure)
	}
	for _, code := range []string{"runtime_missing", "model_missing", "not_prepared", "source_missing"} {
		for name, a := range map[string]ActionAvailability{"start": view.Actions.Start, "restart": view.Actions.Restart, "prepare": view.Actions.Prepare, "download_default": view.Actions.DownloadDefault} {
			if a.ReasonCode == code {
				t.Fatalf("%s claims %s", name, code)
			}
		}
	}
	if view.Actions.Prepare.Allowed || view.Actions.DownloadDefault.Allowed || view.Actions.Start.Allowed {
		t.Fatalf("an unrunnable CLI offers an action: %+v", view.Actions)
	}
	if view.Fit.Class != "unavailable" {
		t.Fatalf("a fit claim without a CLI: %+v", view.Fit)
	}
}
