//go:build windows

package supervisor

import (
	"strings"
	"testing"

	"bubo/gui/updater"
)

func TestUpdateCheckDetailsRetainCauseWithoutInternalRetryInstruction(t *testing.T) {
	f := &updater.Failure{Code: updater.Code("update_information_missing"), Message: "manifest returned HTTP 404", URL: updater.ManifestURL, Retry: "retry the same action"}
	err := updateError(f)
	if err.Code != "update_information_missing" || !strings.Contains(err.Remedy, "HTTP 404") || !strings.Contains(err.Remedy, updater.ManifestURL) {
		t.Fatalf("missing update check cause: %+v", err)
	}
	if strings.Contains(err.Remedy, "retry the same action") {
		t.Fatalf("internal retry instruction leaked into details: %+v", err)
	}
}
