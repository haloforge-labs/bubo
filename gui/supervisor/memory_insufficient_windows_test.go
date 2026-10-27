//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Serve exit 19 (CLI-CONTRACT, ms-memguard e4588b7e): the error carries what
// did not fit; the View keeps the four numbers for the sentence, the CLI's
// remedy and its detail.
func TestMemoryInsufficientStartFailureCarriesTheShortfall(t *testing.T) {
	t.Setenv("BUBO_FAKE_MEMORY_INSUFFICIENT", "1")
	s := supervisorFor(t, "normal")
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600)
	reply(t, s.Start())
	v := waitFor(t, s, "failure", 8*time.Second, func(v Snapshot) bool { return v.State == Failed && v.LastError != nil })
	f := v.LastError
	if f.Code != "memory_insufficient" || f.Remedy != "Lower --context or --sessions, or free memory." || f.Detail == "" {
		t.Fatalf("failure: %+v", f)
	}
	if f.Memory == nil || f.Memory.Sessions != 2 || f.Memory.Context != 262144 || f.Memory.NeededBytes != 50784518144 || f.Memory.AvailableBytes != 33503973376 {
		t.Fatalf("shortfall lost: %+v", f.Memory)
	}
	view := (&ViewBuilder{}).Build(ViewInput{Snapshot: v})
	if view.Failure == nil || view.Failure.Memory == nil || view.Failure.Memory.Sessions != 2 {
		t.Fatalf("the View lost the shortfall: %+v", view.Failure)
	}
}
