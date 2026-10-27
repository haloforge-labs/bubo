package supervisor

import (
	"strings"
	"testing"
)

// The batched dense twin (GUFO_DENSE_TWIN_BATCH=1 with more than one session)
// is resolved by the CLI settings contract. The GUI passes only --sessions and
// the profile, never a twin or batch override of its own.
func TestStartArgsLeaveTwinBatchToCLI(t *testing.T) {
	for _, profile := range []string{"default", "exact"} {
		s := DefaultSettings()
		s.Profile = profile
		s.Sessions = 4
		args := strings.Join(startArgs(s), " ")
		if !strings.Contains(args, "--sessions 4") || !strings.Contains(args, "--profile "+profile) {
			t.Fatalf("%s: sessions and profile missing: %s", profile, args)
		}
		for _, banned := range []string{"TWIN", "twin", "BATCH"} {
			if strings.Contains(args, banned) {
				t.Fatalf("%s: GUI-only twin path in %q", profile, args)
			}
		}
	}
}
