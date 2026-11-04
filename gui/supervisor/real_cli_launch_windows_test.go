//go:build windows

package supervisor

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// An opt-in launch check with the real CLI (it starts no server). It reproduces
// the owner's walk-through: a fresh GUI start with the saved setup, a q8 cache and
// fast prefill, on a machine whose instances folder holds thousands of records of
// servers that are gone. The fit must show without any input, soon after the start.
//
//	BUBO_REAL_CLI_EXE       the CLI of the package
//	BUBO_REAL_CLI_RUNTIME   its runtime folder
//	BUBO_REAL_CLI_MODEL     a GGUF file (only its header is read)
//	BUBO_REAL_CLI_PREPARED  a prepared-default.json to start with (optional)
//	BUBO_REAL_CLI_INSTANCES a copy of an instances folder to start with (optional)
func TestRealCLILaunchShowsTheFitWithoutInput(t *testing.T) {
	exe, runtime, model := os.Getenv("BUBO_REAL_CLI_EXE"), os.Getenv("BUBO_REAL_CLI_RUNTIME"), os.Getenv("BUBO_REAL_CLI_MODEL")
	if exe == "" || runtime == "" || model == "" {
		t.Skip("set BUBO_REAL_CLI_EXE, BUBO_REAL_CLI_RUNTIME and BUBO_REAL_CLI_MODEL")
	}
	scratch := t.TempDir()
	local := filepath.Join(scratch, "local")
	instances := filepath.Join(local, "Bubo", "instances")
	if err := os.MkdirAll(filepath.Join(local, "Bubo"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LOCALAPPDATA", local)
	records := 0
	if source := os.Getenv("BUBO_REAL_CLI_INSTANCES"); source != "" {
		// The CLI refuses an instances folder that inherits its permissions, so the
		// copy gets the owner-only folder the CLI itself makes.
		if err := os.MkdirAll(instances, 0o700); err != nil {
			t.Fatal(err)
		}
		user := os.Getenv("USERDOMAIN") + `\` + os.Getenv("USERNAME")
		if out, err := exec.Command("icacls", instances, "/inheritance:r", "/grant:r", user+":(OI)(CI)F").CombinedOutput(); err != nil {
			t.Fatalf("icacls: %v\n%s", err, out)
		}
		entries, err := os.ReadDir(source)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			to := filepath.Join(instances, entry.Name())
			if err := copyTestFile(filepath.Join(source, entry.Name()), to); err != nil {
				t.Fatal(err)
			}
			// The CLI drops old records of servers that are gone: keep their age.
			if info, err := entry.Info(); err == nil {
				_ = os.Chtimes(to, info.ModTime(), info.ModTime())
			}
			records++
		}
	}
	if pointer := os.Getenv("BUBO_REAL_CLI_PREPARED"); pointer != "" {
		if err := copyTestFile(pointer, filepath.Join(local, "Bubo", "prepared-default.json")); err != nil {
			t.Fatal(err)
		}
	}
	saved := DefaultSettings()
	saved.ModelPath, saved.RuntimePath = model, runtime
	saved.Context, saved.Sessions = 57344, 1
	saved.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"}
	settingsPath := filepath.Join(scratch, "settings.json")
	if err := SaveSettings(settingsPath, saved); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Exe: exe, RuntimeDir: runtime, Port: 19097, SettingsPath: settingsPath, LogDir: filepath.Join(scratch, "logs"), PollInterval: time.Second}
	b, err := NewBackend(cfg, saved, exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, b.s.Exit()) })
	a := newTestApp(b, saved, nil, "")
	a.AttachModels(settingsPath, exe, "")
	t.Cleanup(func() { a.models.Close() })
	began := time.Now()
	var initialized, fitted time.Duration
	var v View
	for time.Since(began) < 90*time.Second {
		v = a.GetView()
		if initialized == 0 && v.Initialized {
			initialized = time.Since(began)
		}
		if v.Initialized && (v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error != nil) {
			fitted = time.Since(began)
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	left, _ := os.ReadDir(instances)
	snap := b.Snapshot()
	t.Logf("label %q; log: %v", snap.Label, b.s.logs.Ring())
	t.Logf("%d records at the start, %d after; initialized after %s, fit after %s; state %s, error %+v; fit %+v; estimate error %+v", records, len(left), initialized, fitted, snap.State, snap.LastError, v.Fit, v.SavedEstimate.Error)
	if initialized == 0 || initialized > 15*time.Second {
		t.Errorf("the CLI answered the first call after %s", initialized)
	}
	if v.SavedEstimate.Error != nil || v.Fit.Class != "fits" {
		t.Errorf("a fresh launch with the saved setup shows no fit: %+v, error %+v", v.Fit, v.SavedEstimate.Error)
	}
}

// When the setup shows Fits, its Start must not be refused for memory. The owner's
// final6 walk-through showed Fits for a saved IQ3_XXS setup (262144 x 4, q8_0 cache,
// fast prefill) and the Start was refused as memory_tight, because the start check
// asked the CLI about an f16 cache. This runs both checks against the real CLI; no
// server starts. Pick a context and session count that fit only with a q8_0 cache
// (BUBO_REAL_CLI_CONTEXT, BUBO_REAL_CLI_SESSIONS: tokens per session, default
// 262144 and 4). BUBO_REAL_CLI_SWITCHES sets the saved overrides (default a q8_0
// cache and fast prefill). Set BUBO_REAL_CLI_REQUIRE_Q8_ONLY=1 to fail when f16 would fit too,
// as the run then proves nothing about the cache type.
func TestRealCLIStartCheckAgreesWithTheShownFit(t *testing.T) {
	exe, runtime, model := os.Getenv("BUBO_REAL_CLI_EXE"), os.Getenv("BUBO_REAL_CLI_RUNTIME"), os.Getenv("BUBO_REAL_CLI_MODEL")
	if exe == "" || runtime == "" || model == "" {
		t.Skip("set BUBO_REAL_CLI_EXE, BUBO_REAL_CLI_RUNTIME and BUBO_REAL_CLI_MODEL")
	}
	context, sessions := uint64(262144), uint32(4)
	if v, err := strconv.ParseUint(os.Getenv("BUBO_REAL_CLI_CONTEXT"), 10, 64); err == nil {
		context = v
	}
	if v, err := strconv.ParseUint(os.Getenv("BUBO_REAL_CLI_SESSIONS"), 10, 32); err == nil {
		sessions = uint32(v)
	}
	scratch := t.TempDir()
	local := filepath.Join(scratch, "local")
	if err := os.MkdirAll(filepath.Join(local, "Bubo"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LOCALAPPDATA", local)
	if pointer := os.Getenv("BUBO_REAL_CLI_PREPARED"); pointer != "" {
		if err := copyTestFile(pointer, filepath.Join(local, "Bubo", "prepared-default.json")); err != nil {
			t.Fatal(err)
		}
	}
	saved := DefaultSettings()
	saved.ModelPath, saved.RuntimePath = model, runtime
	saved.Context, saved.Sessions = context, sessions
	saved.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"}
	// BUBO_REAL_CLI_SWITCHES replaces them: NAME=VALUE pairs joined by "|", for the
	// side files, a bare flag (NAME=) or an environment row.
	if spec := os.Getenv("BUBO_REAL_CLI_SWITCHES"); spec != "" {
		saved.SwitchOverrides = map[string]string{}
		for _, pair := range strings.Split(spec, "|") {
			name, value, _ := strings.Cut(pair, "=")
			saved.SwitchOverrides[name] = value
		}
	}
	settingsPath := filepath.Join(scratch, "settings.json")
	if err := SaveSettings(settingsPath, saved); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Exe: exe, RuntimeDir: runtime, Port: 19098, SettingsPath: settingsPath, LogDir: filepath.Join(scratch, "logs"), PollInterval: time.Second}
	b, err := NewBackend(cfg, saved, exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, b.s.Exit()) })
	a := newTestApp(b, saved, nil, "")
	a.AttachModels(settingsPath, exe, "")
	t.Cleanup(func() { a.models.Close() })
	began := time.Now()
	var v View
	for time.Since(began) < 90*time.Second {
		v = a.GetView()
		if v.Initialized && (v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error != nil) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if v.SavedEstimate.Error != nil || v.SavedEstimate.MemoryEstimate == nil {
		t.Fatalf("no estimate for the saved setup: %+v", v.SavedEstimate.Error)
	}
	// The start check, as the supervisor runs it for the launch arguments.
	shown := v.Fit.Class
	admission := b.s.admitSavedStart(saved, b.Snapshot().Status)
	var f16 string
	if estimate, err := a.models.EstimateWithCache(filepath.Clean(model), context, sessions, "f16"); err == nil && estimate.MemoryEstimate != nil {
		f16 = buildFit(estimate.MemoryEstimate).Class
	}
	t.Logf("%d x %d: the fit shown is %q; the start check says %v; at f16 the fit would be %q", context, sessions, shown, admission, f16)
	if shown == "fits" && admission != nil {
		t.Errorf("the setup shows Fits and its Start is refused: %v", admission)
	}
	if os.Getenv("BUBO_REAL_CLI_REQUIRE_Q8_ONLY") == "1" && f16 == "fits" {
		t.Errorf("f16 fits at %d x %d, so this run does not test the cache type", context, sessions)
	}
}

func copyTestFile(from, to string) error {
	in, err := os.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
