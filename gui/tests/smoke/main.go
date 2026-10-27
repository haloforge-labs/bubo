//go:build windows

// bubo-smoke drives the real bubo.exe through the same gui/supervisor code the
// app uses, with no window. It uses one loopback port (default 18120), never
// 8081-8086. It starts only processes it records, and it stops only those, by
// exact PID. Run it only from the user queue (see SMOKE-PLAN.md).
//
// Exit codes: 0 every step passed; 1 a step failed; 2 usage error; 3 refused
// (forbidden port, GPU lock, busy port); 4 no failure but a step did not run;
// 130 interrupted (Ctrl+C or Ctrl+Break).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"bubo/gui/supervisor"
)

const (
	ExitPass        = 0
	ExitFail        = 1
	ExitUsage       = 2
	ExitRefused     = 3
	ExitNotRun      = 4
	ExitInterrupted = 130
)

var defaultLock = filepath.Join(os.TempDir(), "bubo-test-guards", "gpu-harness.lock")
var defaultBuildLock = filepath.Join(os.TempDir(), "bubo-test-guards", "build.lock")

// Run modes. Only "full" can produce the PASS verdict that proves the smoke.
const (
	ModeFull          = "full"
	ModePreflightOnly = "preflight-only"
	ModeOnlyStep9     = "only-step9"
)

type Config struct {
	MVP            bool
	MVPStep1Only   bool
	Fake           bool
	Package        string
	Exe            string
	Manifest       string
	Out            string
	Port           int
	LockPath       string
	BuildLockPath  string
	AllowLockOwner string
	RequireLocks   bool
	PreflightOnly  bool
	OnlyStep9      bool
	ProductionUp   bool
	ProcPatterns   []string
	Pins           string
	StopFile       string
	LoadTimeout    time.Duration
	StopTimeout    time.Duration
	RequestTimeout time.Duration
	QueryTimeout   time.Duration
	PollInterval   time.Duration
	IdleWatch      time.Duration
	SlotIdleWait   time.Duration
	MinFreeRAMGiB  float64
	AllowBattery   bool
	KeepEnv        []string
}

func main() {
	var c Config
	var keep, procs string
	fs := flag.NewFlagSet("bubo-smoke", flag.ContinueOnError)
	fs.BoolVar(&c.MVP, "mvp", false, "run the v0.1.1 View smoke (user entry script selects this)")
	fs.BoolVar(&c.MVPStep1Only, "mvp-step1-only", false, "run only MVP step 1 (no model load, no GPU work, no locks needed) and the evidence step")
	fs.BoolVar(&c.Fake, "fake", false, "fake CLI rehearsal: nullable live rates are structural checks only; never a real-hardware PASS")
	fs.StringVar(&c.Package, "package", "", "K-b package root; --exe must be its bin/bubo.exe in real MVP mode")
	fs.StringVar(&c.Exe, "exe", "", "absolute path of bubo.exe (required)")
	fs.StringVar(&c.Manifest, "manifest", "", "absolute path of the prepared bubo-package.json (required)")
	fs.StringVar(&c.Out, "out", "", "output folder for report.json, transcript.txt, pids.jsonl (required, new or empty)")
	fs.IntVar(&c.Port, "port", 18120, "loopback port; 8081-8086 are refused")
	fs.StringVar(&c.LockPath, "lock-path", defaultLock, "GPU harness lock file to respect")
	fs.StringVar(&c.BuildLockPath, "build-lock-path", defaultBuildLock, "build lock file to respect (same rules as the GPU lock)")
	fs.BoolVar(&c.RequireLocks, "require-locks", false, "refuse unless both lock files exist and carry the --allow-lock-owner line (the user waiter writes them)")
	fs.BoolVar(&c.PreflightOnly, "preflight-only", false, "run step 1 only (no server is started), write the report, exit 0 or 1")
	fs.BoolVar(&c.OnlyStep9, "only-step9", false, "run step 9 (log export leak check) only; no model is loaded")
	fs.BoolVar(&c.ProductionUp, "production-up", false, "with --preflight-only: production is still running, so the process list, live instance and RAM floor are recorded but do not fail")
	fs.StringVar(&procs, "proc-patterns", "bubo.exe,gufo.exe,llama-*.exe", "process image names that must not be running (read-only check)")
	fs.StringVar(&c.StopFile, "stop-file", "", "when this file appears the run is interrupted exactly like Ctrl+C (the user waiter uses it for its outer time limit)")
	fs.StringVar(&c.Pins, "pins", "", "comma list name=sha256 of files in the --exe folder that must match (case-insensitive hex)")
	fs.StringVar(&c.AllowLockOwner, "allow-lock-owner", "", "exact owner line of the lock this run may proceed under")
	fs.DurationVar(&c.LoadTimeout, "load-timeout", 900*time.Second, "model load deadline")
	fs.DurationVar(&c.StopTimeout, "stop-timeout", 45*time.Second, "graceful stop deadline")
	fs.DurationVar(&c.RequestTimeout, "request-timeout", 180*time.Second, "one chat request deadline")
	fs.DurationVar(&c.QueryTimeout, "query-timeout", 30*time.Second, "one status query deadline")
	fs.DurationVar(&c.PollInterval, "poll-interval", 2*time.Second, "supervisor status sampling interval")
	fs.DurationVar(&c.IdleWatch, "idle-watch", 20*time.Second, "idle time watched for a stall warning")
	fs.DurationVar(&c.SlotIdleWait, "slot-idle-wait", 30*time.Second, "how long step 4 waits for the slots to go idle")
	fs.Float64Var(&c.MinFreeRAMGiB, "min-free-ram-gib", 0, "preflight fails below this much free host RAM (0 records only)")
	fs.BoolVar(&c.AllowBattery, "allow-battery", false, "do not fail preflight on battery power")
	fs.StringVar(&keep, "keep-env-prefix", "", "comma list of environment prefixes kept when the production scrub runs (tests only)")
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(ExitUsage)
	}
	if keep != "" {
		c.KeepEnv = strings.Split(keep, ",")
	}
	c.ProcPatterns = strings.Split(procs, ",")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if c.StopFile != "" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		go watchStopFile(ctx, c.StopFile, cancel)
	}
	os.Exit(run(ctx, c, os.Stdout))
}

// watchStopFile cancels the run when the file appears. It only reads.
func watchStopFile(ctx context.Context, path string, cancel context.CancelFunc) {
	t := time.NewTicker(500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if _, err := os.Stat(path); err == nil {
				cancel()
				return
			}
		}
	}
}

func (c *Config) validate() error {
	if c.Fake && !c.MVP {
		return errors.New("--fake requires --mvp")
	}
	if c.MVPStep1Only && !c.MVP {
		return errors.New("--mvp-step1-only requires --mvp")
	}
	if c.MVP && (c.PreflightOnly || c.OnlyStep9 || c.ProductionUp) {
		return errors.New("--mvp cannot be combined with legacy partial modes")
	}
	if c.MVP && !c.Fake && (!filepath.IsAbs(c.Package) || normPath(c.Exe) != normPath(filepath.Join(c.Package, "bin", "bubo.exe"))) {
		return errors.New("real --mvp requires --package <K-b folder> and its bin/bubo.exe")
	}
	if c.MVP && !c.Fake && c.LoadTimeout > 900*time.Second {
		return errors.New("MVP readiness deadline cannot exceed 900s")
	}
	for name, p := range map[string]string{"--exe": c.Exe, "--manifest": c.Manifest, "--out": c.Out} {
		if p == "" {
			return fmt.Errorf("%s is required", name)
		}
		if !filepath.IsAbs(p) {
			return fmt.Errorf("%s must be an absolute path", name)
		}
	}
	if c.LoadTimeout <= 0 || c.StopTimeout <= 0 || c.RequestTimeout <= 0 || c.QueryTimeout <= 0 || c.PollInterval <= 0 || c.IdleWatch <= 0 || c.SlotIdleWait <= 0 {
		return errors.New("every duration must be positive")
	}
	if c.PreflightOnly && c.OnlyStep9 {
		return errors.New("--preflight-only and --only-step9 cannot be combined")
	}
	if c.ProductionUp && !c.PreflightOnly {
		return errors.New("--production-up is allowed only with --preflight-only")
	}
	return nil
}

// refusal returns the reason this run must not start, or "". It runs before the
// tool creates, starts or touches anything except the lock file (read only).
func refusal(c Config) string {
	if c.Port >= 8081 && c.Port <= 8086 {
		return fmt.Sprintf("port %d is a production port (8081-8086 are never used by the smoke)", c.Port)
	}
	if err := supervisor.ValidatePort(c.Port); err != nil {
		return fmt.Sprintf("port %d is not permitted", c.Port)
	}
	// The locks matter only when a model will load. The two read-only modes
	// record the lock state and never refuse on it.
	if c.mode() == ModeFull {
		for _, l := range []struct{ name, path string }{{"GPU lock", c.LockPath}, {"build lock", c.BuildLockPath}} {
			if l.path == "" {
				continue
			}
			if why := lockRefusal(l.name, l.path, c.AllowLockOwner, c.RequireLocks); why != "" {
				return why
			}
		}
	}
	if c.mode() == ModeOnlyStep9 {
		return ""
	}
	if err := portFree(c.Port); err != nil {
		return fmt.Sprintf("port %d is occupied: %v", c.Port, err)
	}
	return ""
}

func (c Config) mode() string {
	switch {
	case c.PreflightOnly:
		return ModePreflightOnly
	case c.OnlyStep9:
		return ModeOnlyStep9
	}
	return ModeFull
}

// lockOwnerLines returns the trimmed non-empty lines of a lock file, or nil
// with exists=false when there is no lock.
func lockOwnerLines(path string) (lines []string, exists bool, err error) {
	b, err := readSmall(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, true, err
	}
	for _, l := range strings.Split(strings.ReplaceAll(string(b), "\r", ""), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			lines = append(lines, l)
		}
	}
	return lines, true, nil
}

// lockRefusal applies the whole-line owner match to one lock file. A missing
// lock is accepted unless require is set (the user waiter must hold it).
func lockRefusal(name, path, allow string, require bool) string {
	lines, exists, err := lockOwnerLines(path)
	switch {
	case err != nil:
		return fmt.Sprintf("%s %s exists but cannot be read: %v", name, path, err)
	case !exists:
		if require {
			return fmt.Sprintf("%s %s does not exist; the user waiter must hold it for the run (--require-locks)", name, path)
		}
		return ""
	}
	owner := ""
	if len(lines) > 0 {
		owner = lines[0]
	}
	for _, l := range lines {
		if allow != "" && l == allow {
			return ""
		}
	}
	return fmt.Sprintf("%s %s exists (owner line %q); pass it with --allow-lock-owner only if this run belongs to that owner", name, path, owner)
}

func readSmall(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(io.LimitReader(f, 4096))
}

// portFree binds and releases 127.0.0.1:port. It touches only the given port.
func portFree(port int) error {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	return l.Close()
}

var scrubPrefixes = []string{"GUFO_", "HIP_", "HSA_", "ROC", "AMD_", "GPU_", "HCC_", "MIOPEN", "LLAMA_", "GGML_", "BUBO_"}

// scrubEnv removes the experiment and timing variables that serve-bubo.ps1
// removes, so the child sees the production environment. It returns the removed
// names (names only) and a restore function.
func scrubEnv(keep []string) ([]string, func()) {
	orig := os.Environ()
	var removed []string
	for _, kv := range orig {
		name, _, _ := strings.Cut(kv, "=")
		if name == "" {
			continue
		}
		up := strings.ToUpper(name)
		skip := false
		for _, k := range keep {
			if k != "" && strings.HasPrefix(up, strings.ToUpper(k)) {
				skip = true
			}
		}
		if skip {
			continue
		}
		for _, p := range scrubPrefixes {
			if strings.HasPrefix(up, p) {
				os.Unsetenv(name)
				removed = append(removed, name)
				break
			}
		}
	}
	sort.Strings(removed)
	return removed, func() {
		now := map[string]bool{}
		for _, kv := range os.Environ() {
			n, _, _ := strings.Cut(kv, "=")
			now[n] = true
		}
		for n := range now {
			if n != "" {
				os.Unsetenv(n)
			}
		}
		for _, kv := range orig {
			if n, v, ok := strings.Cut(kv, "="); ok && n != "" {
				os.Setenv(n, v)
			}
		}
	}
}

// run is the whole program. It returns the exit code. --out is created before
// the refusal checks (a REFUSED try still writes its report), so every try
// needs its own --out.
func run(ctx context.Context, c Config, console io.Writer) int {
	if err := c.validate(); err != nil {
		fmt.Fprintln(console, "USAGE:", err)
		return ExitUsage
	}
	if entries, err := os.ReadDir(c.Out); err == nil && len(entries) > 0 {
		fmt.Fprintln(console, "USAGE: --out must be new or empty:", c.Out)
		return ExitUsage
	}
	rec, err := NewRecorder(c.Out, console)
	if err != nil {
		fmt.Fprintln(console, "USAGE: cannot create --out:", err)
		return ExitUsage
	}
	defer rec.Close()
	rec.report.Exe, rec.report.Manifest, rec.report.Port, rec.report.Mode = c.Exe, c.Manifest, c.Port, c.mode()
	if why := refusal(c); why != "" {
		rec.Logf("REFUSED: %s", why)
		rec.report.Overall, rec.report.ExitCode = "REFUSED", ExitRefused
		rec.report.FinishedAt = stamp()
		return ExitRefused
	}
	removed, restore := scrubEnv(c.KeepEnv)
	defer restore()
	// The status helper reads the model selection from this variable.
	os.Setenv("BUBO_MANIFEST", c.Manifest)
	r := &Runner{cfg: c, rec: rec, led: NewLedger(rec, c.Out), scrubbed: removed}
	if c.MVP {
		return r.runMVP(ctx)
	}
	return r.runAll(ctx)
}
