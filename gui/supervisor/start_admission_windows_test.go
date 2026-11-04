//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The owner's walk-through: the setup shows Fits with a q8_0 cache (the fit asks the
// CLI about q8_0), then the start was refused as memory_tight because its own check
// asked about f16. The two checks must ask the same question.
func TestStartChecksMemoryAtTheSavedCacheType(t *testing.T) {
	for _, tc := range []struct {
		kv   string
		want []string
	}{
		{"q8_0", []string{"--switch", "--kv-cache-type=q8_0", "--switch", "--prefill-mode=fast"}},
		{"f16", []string{"--switch", "--kv-cache-type=f16", "--switch", "--prefill-mode=fast"}},
		{"", []string{"--switch", "--prefill-mode=fast"}},
	} {
		t.Run("kv "+tc.kv, func(t *testing.T) {
			cfg := configFor(t, "normal")
			cfg.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
			s, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { reply(t, s.Exit()) }()
			folder := e21Folder(t, e21Flash)
			saved := DefaultSettings()
			saved.ManifestPath = filepath.Join(folder, "bubo-package.json")
			saved.Context, saved.Sessions = 262144, 4
			saved.SwitchOverrides = map[string]string{"--prefill-mode": "fast"}
			if tc.kv != "" {
				saved.SwitchOverrides["--kv-cache-type"] = tc.kv
			}
			_ = s.admitSavedStart(saved, &Status{}) // the answer is the recording's; the request is the point
			want := append([]string{"models", "estimate", "--model", folder, "--json", "--context", "262144", "--sessions", "4"}, tc.want...)
			asked := recordedEstimates(t, os.Getenv("BUBO_FAKE_DIR"))
			if len(asked) != 1 || !reflect.DeepEqual(asked[0], want) {
				t.Fatalf("the start asked %v, want %v", asked, want)
			}
			// The fit shown before the start builds its request the same way.
			if shown := estimateArguments(folder, 262144, 4, "", saved.SwitchOverrides); !reflect.DeepEqual(shown, want) {
				t.Fatalf("the shown fit asks %v, the start asks %v", shown, want)
			}
		})
	}
}

// The owner's final7 walk-through runs with an MTP file and a vision file. Both add
// to the memory the server needs, so the start check sends them to the estimate.
func TestStartChecksMemoryWithTheSavedSideFiles(t *testing.T) {
	cfg := configFor(t, "normal")
	cfg.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { reply(t, s.Exit()) }()
	folder := e21Folder(t, e21Flash)
	saved := DefaultSettings()
	saved.ManifestPath = filepath.Join(folder, "bubo-package.json")
	saved.Context, saved.Sessions = 100000, 4
	saved.SwitchOverrides = map[string]string{
		"--kv-cache-type":     "q8_0",
		"--mmproj":            `C:\m\mmproj.gguf`,
		"--mtp-model":         `C:\m\mtp.gguf`,
		"--prefill-mode":      "fast",
		"--refusal-direction": "on",
	}
	_ = s.admitSavedStart(saved, &Status{})
	want := []string{"models", "estimate", "--model", folder, "--json", "--context", "100000", "--sessions", "4",
		"--switch", "--kv-cache-type=q8_0", "--switch", `--mmproj=C:\m\mmproj.gguf`, "--switch", `--mtp-model=C:\m\mtp.gguf`,
		"--switch", "--prefill-mode=fast", "--switch", "--refusal-direction=on"}
	if asked := recordedEstimates(t, os.Getenv("BUBO_FAKE_DIR")); len(asked) != 1 || !reflect.DeepEqual(asked[0], want) {
		t.Fatalf("the start asked %v, want %v", asked, want)
	}
}
