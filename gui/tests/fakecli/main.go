//go:build windows

// fakecli is a test-only bubo.exe. It never binds a network port or loads an
// engine. It speaks CLI contract schema_version 1: lifecycle events with
// identity, `status --json` snapshots, `stop --json`, `serve --list-switches
// --json` and prepare JSON Lines. It can replay real contract fixtures.
//
// Environment:
//
//	BUBO_FAKE_DIR       state folder (required)
//	BUBO_FAKE_MODE      failure or scenario mode (see the switch statements)
//	BUBO_FAKE_FIXTURES  tests/cli/contract folder for replay and the registry
//	BUBO_FAKE_REPLAY    prepare: comma list of valid/ fixture names to stream
//	BUBO_FAKE_STATUS_DELAYS comma-separated Go durations for successive one-shot
//	                    status calls (e.g. "200ms,0s"); survives child restarts
//	BUBO_FAKE_RECORDINGS the v12 recordings folder (default: found from
//	                    BUBO_FAKE_FIXTURES); models, status, estimates, plans and
//	                    the switch registry are replayed from it (recorded_windows.go)
//	status.json        (in BUBO_FAKE_DIR) top-level keys merged into the next
//	                    status payload, so a test can script slots and counters
package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/windows"
)

var seq uint64
var eventMu sync.Mutex
var op = "11111111111111111111111111111111"
var instance string
var creation string
var selfExe string
var dir = os.Getenv("BUBO_FAKE_DIR")
var mode = os.Getenv("BUBO_FAKE_MODE")
var fixtures = os.Getenv("BUBO_FAKE_FIXTURES")
var product map[string]any
var capabilities = recordedStatus(true)["capabilities"].(map[string]any)

func init() { product = recordedStatus(true)["versions"].(map[string]any) }

func flag(name string) string {
	for i, a := range os.Args {
		if a == name && i+1 < len(os.Args) {
			return os.Args[i+1]
		}
	}
	return ""
}
func now() string { return time.Now().UTC().Format("2006-01-02T15:04:05Z") }

func progressPause(fallback time.Duration) time.Duration {
	value := os.Getenv("BUBO_FAKE_PROGRESS_PAUSE_MS")
	if value == "" {
		return fallback
	}
	ms, err := strconv.Atoi(value)
	if err != nil || ms < 0 || ms > 10000 {
		panic("BUBO_FAKE_PROGRESS_PAUSE_MS must be 0..10000")
	}
	return time.Duration(ms) * time.Millisecond
}

// eventPID is the pid the lifecycle events claim. Mode spoof-pid lies.
func eventPID() int {
	if mode == "spoof-pid" {
		return os.Getpid() + 1
	}
	return os.Getpid()
}
func identity() map[string]any {
	return map[string]any{"instance_id": instance, "pid": eventPID(), "process_creation_time": creation, "exe_path": selfExe}
}

// event writes one envelope. Lifecycle events from serve also carry the
// identity at the top level, as the contract requires.
func event(kind string, payload any) {
	eventMu.Lock()
	defer eventMu.Unlock()
	seq++
	version := 1
	if mode == "major" {
		version = 2 // a CLI from a future major version
	}
	e := map[string]any{"schema_version": version, "type": kind, "operation_id": op, "seq": seq, "time": now(), "product": product, "payload": payload}
	switch kind {
	case "starting", "loading", "ready", "warning", "stopping", "stopped":
		if instance != "" {
			e["instance_id"] = instance
			e["pid"] = eventPID()
			e["process_creation_time"] = creation
			e["exe_path"] = selfExe
		}
	}
	b, _ := json.Marshal(e)
	if mode == "split" {
		for _, v := range append(b, '\r', '\n') {
			os.Stdout.Write([]byte{v})
		}
	} else {
		fmt.Println(string(b))
	}
}

// life emits a lifecycle event whose payload starts with the identity.
func life(kind string, extra map[string]any) {
	p := map[string]any{"identity": identity()}
	for k, v := range extra {
		p[k] = v
	}
	event(kind, p)
}
func cliError(code, component, stage string, retryable bool, remedy string, exit int) {
	payload := map[string]any{"code": code, "component": component, "stage": stage, "retryable": retryable, "remedy": remedy}
	if detail := os.Getenv("BUBO_FAKE_ERROR_DETAIL"); detail != "" {
		payload["detail"] = detail // the optional cause field of CLI-CONTRACT
	}
	event("error", payload)
	os.Exit(exit)
}
func randomID() string {
	var b [16]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
func created() string {
	h, _ := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(os.Getpid()))
	defer windows.CloseHandle(h)
	var c, e, k, u windows.Filetime
	windows.GetProcessTimes(h, &c, &e, &k, &u)
	return strconv.FormatUint(uint64(c.HighDateTime)<<32|uint64(c.LowDateTime), 10)
}
func recordPID() {
	f, _ := os.OpenFile(filepath.Join(dir, "pids.txt"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	fmt.Fprintln(f, os.Getpid())
	f.Close()
}
func live(i map[string]any) bool {
	p, _ := i["pid"].(float64)
	h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(p))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	s, _ := windows.WaitForSingleObject(h, 0)
	return s == uint32(windows.WAIT_TIMEOUT)
}
func readInstance() map[string]any {
	b, err := os.ReadFile(filepath.Join(dir, "instance.json"))
	if err != nil {
		return nil
	}
	var i map[string]any
	if json.Unmarshal(b, &i) != nil || !live(i) {
		return nil
	}
	return i
}
func save(i map[string]any) {
	b, _ := json.Marshal(i)
	tmp := filepath.Join(dir, "instance.tmp")
	os.WriteFile(tmp, b, 0600)
	from, _ := windows.UTF16PtrFromString(tmp)
	to, _ := windows.UTF16PtrFromString(filepath.Join(dir, "instance.json"))
	var err error
	for n := 0; n < 100; n++ {
		err = windows.MoveFileEx(from, to, windows.MOVEFILE_REPLACE_EXISTING)
		if err == nil {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	fmt.Fprintln(os.Stderr, "fake identity publication failed:", err)
	os.Exit(70)
}

// row is one effective_settings row in the contract shape.
func row(requested, effective any, source string, reason any) map[string]any {
	return map[string]any{"requested": requested, "effective": effective, "source": source, "reason": reason}
}

// effectiveFromArgs reads the flags the supervisor passes to serve and reports
// them as CLI rows (source "flag" for what was requested).
func effectiveFromArgs() map[string]any {
	out := map[string]any{"profile": row("default", "default", "default", nil), "context": row(nil, 200000, "default", nil), "sessions": row(nil, 1, "default", nil)}
	if v := flag("--profile"); v != "" {
		effective := v
		if v == "stock" {
			effective = "vanilla"
		}
		out["profile"] = row(v, effective, "flag", nil)
	}
	for key, name := range map[string]string{"context": "--context", "sessions": "--sessions"} {
		if v := flag(name); v != "" {
			n, _ := strconv.Atoi(v)
			out[key] = row(n, n, "flag", nil)
		}
	}
	if httpMode == "mvp" {
		mvpEffective(out)
	}
	if v := flag("--npu-prefill"); v != "" {
		out["--npu-prefill"] = row(v, "off", "flag", "Test CLI has no NPU support.")
	}
	if mode == "ui" || mode == "ui-fail" {
		out["--npu-prefill"] = row("on", "off", "profile", "Test CLI has no NPU support.")
		out["--reasoning-effort"] = row(nil, "medium", "default", nil)
	}
	return out
}

func readFixture(name string) (map[string]any, bool) {
	if fixtures == "" {
		return nil, false
	}
	b, err := os.ReadFile(filepath.Join(fixtures, "valid", name+".json"))
	if err != nil {
		return nil, false
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return nil, false
	}
	return m, true
}
func replay(name string) (string, map[string]any) {
	m, ok := readFixture(name)
	if !ok {
		cliError("unexpected_failure", "fake", "replay", false, "Fixture "+name+" is missing.", 1)
	}
	return m["type"].(string), m["payload"].(map[string]any)
}

func instanceSummary(i map[string]any, ready bool) map[string]any {
	id := map[string]any{"instance_id": i["instance_id"], "pid": i["pid"], "process_creation_time": i["process_creation_time"], "exe_path": i["exe_path"]}
	return map[string]any{"schema_version": 1, "control_version": 1, "identity": id, "endpoint": i["endpoint"], "phase": i["phase"], "log_path": nil, "last_error": nil,
		"instance_id": i["instance_id"], "pid": i["pid"], "process_creation_time": i["process_creation_time"], "exe_path": i["exe_path"],
		"ready": ready, "effective_settings": i["effective_settings"], "served_model": i["served_model"]}
}

func statusPayload() map[string]any {
	prepared := true
	runtimeReady := true
	if mode == "ui" || mode == "ui-fail" {
		prepared = fakePrepared()
		runtimeReady = fakeRuntimePrepared()
	}
	// The recorded status of a profile with or without the default model (E21).
	// What follows adds only what belongs to this fake process: its version, its
	// state folder, the test's switches and a runtime that is not there (no
	// recording of that state exists for the candidate exe: hand-written fields).
	p := recordedStatus(prepared)
	p["versions"] = product
	p["sample_time"] = now()
	if paths, ok := p["paths"].(map[string]any); ok {
		paths["local_app_data"] = dir
		if runtimePath := os.Getenv("BUBO_FAKE_RUNTIME_PATH"); runtimePath != "" {
			paths["runtime"] = runtimePath
		}
	}
	if caps, ok := p["capabilities"].(map[string]any); ok {
		if mode == "unsupported" {
			caps["instance_control"] = false
		}
		if mode == "no-status-watch" {
			caps["status_watch"] = false
		}
	}
	if !runtimeReady {
		p["phase"] = "not_prepared"
		p["readiness"] = map[string]any{"runtime": false, "model": false}
		p["last_error"] = map[string]any{"code": "runtime_missing", "component": "runtime", "stage": "status", "retryable": false, "remedy": "Prepare the runtime."}
	}
	if httpMode != "" {
		httpAlways(p)
	}
	if packagesMode() {
		packagesStatus(p)
	}
	i := readInstance()
	want := flag("--instance-id")
	if want != "" && (i == nil || i["instance_id"] != want) {
		cliError("control_access", "control", "status", false, "Select a running instance.", 16)
	}
	if i != nil {
		ready := i["phase"] == "ready"
		if mode == "disagreement" {
			ready = false
		}
		id := map[string]any{"instance_id": i["instance_id"], "pid": i["pid"], "process_creation_time": i["process_creation_time"], "exe_path": i["exe_path"]}
		p["phase"] = i["phase"]
		p["instances"] = []any{instanceSummary(i, ready)}
		p["identity"] = id
		p["endpoint"] = i["endpoint"]
		p["served_model"] = i["served_model"]
		p["effective_settings"] = i["effective_settings"]
		p["last_transition"] = i["last_transition"]
		p["memory"] = map[string]any{"rss_bytes": 10661888, "host_available_bytes": 13970321408, "unit": "bytes", "evidence": "measured"}
		p["counters"] = map[string]any{"prefill_tokens_total": 0, "decode_tokens_total": 0, "cached_tokens_total": 0, "cache_disk_hits_total": 0, "cache_memory_hits_total": 0, "cache_misses_total": 0, "max_sequence_tokens": 0, "draft_rounds_total": 0, "draft_tokens_total": 0, "draft_accepted_tokens_total": 0, "device_lost_total": 0, "prefill_seconds_total": 0.0, "decode_seconds_total": 0.0}
		p["last_completed_request"] = map[string]any{"prefill_tokens_per_second": nil, "decode_tokens_per_second": nil, "age_ms": nil, "rate_unit": "tokens_per_second", "age_unit": "milliseconds"}
		p["sessions"] = 0
		p["queue"] = 0
		p["slots"] = []any{idleSlot(0)}
		p["requests"] = []any{}
		if ep, ok := i["endpoint"].(string); ok {
			p["connection"] = map[string]any{"endpoint": ep, "model_id": i["served_model"], "context_tokens": 200000, "authenticated": i["authenticated"] == true}
		}
		p["readiness"] = map[string]any{"runtime": true, "model": true, "server": ready}
		if mode == "ui" || mode == "ui-fail" {
			uiStatus(p, i)
		}
		if httpMode != "" {
			httpInstance(p, i)
		}
	}
	if b, err := os.ReadFile(filepath.Join(dir, "status.json")); err == nil && i != nil {
		var over map[string]any
		if json.Unmarshal(b, &over) == nil {
			for k, v := range over {
				p[k] = v
			}
		}
	}
	if httpMode == "mvp" {
		mvpStatus(p, i)
	}
	return p
}

func fakePrepared() bool {
	_, err := os.Stat(filepath.Join(dir, "prepared"))
	return err == nil
}
func fakeRuntimePrepared() bool {
	if fakePrepared() {
		return true
	}
	_, err := os.Stat(filepath.Join(dir, "runtime-prepared"))
	return err == nil
}
func idleSlot(id int) map[string]any {
	return map[string]any{"id": id, "processing": false, "state": "idle", "tokens_used": 0, "speculative": false, "request_id": nil, "context_tokens": 200000, "prompt_tokens": 0, "cached_prompt_tokens": 0, "prefilled_tokens": 0, "generated_tokens": 0, "remaining_tokens": nil}
}

func hasFlag(name string) bool {
	for _, arg := range os.Args {
		if arg == name {
			return true
		}
	}
	return false
}

func main() {
	if os.Getenv("BUBO_FAKE_PACKAGE_VERSION_FILE") == "1" {
		exe, err := os.Executable()
		if err != nil {
			panic(err)
		}
		value, err := os.ReadFile(filepath.Join(filepath.Dir(filepath.Dir(exe)), "fixture-version"))
		if err != nil {
			panic(err)
		}
		product["version"] = strings.TrimSpace(string(value))
	}
	if os.Getenv("BUBO_FAKE_RECORD_RUNTIME") == "1" {
		// Test-only child-side observation; never records credentials.
		runtimeDir, present := os.LookupEnv("BUBO_RUNTIME_DIR")
		exe, _ := os.Executable()
		f, err := os.OpenFile(filepath.Join(dir, "runtime-children.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			panic(err)
		}
		_ = json.NewEncoder(f).Encode(struct {
			Args    []string
			Runtime string
			Present bool
			Exe     string
		}{os.Args[1:], runtimeDir, present, exe})
		_ = f.Close()
	}
	if len(os.Args) < 2 {
		os.Exit(2)
	}
	if recordedScenario() {
		return
	}
	selfExe, _ = os.Executable()
	if mode == "no-vcruntime" {
		os.Exit(0xC0000135) // Windows could not load a DLL: no output at all
	}
	if os.Args[1] == "echo" {
		b, _ := json.Marshal(os.Args[2:])
		fmt.Println(string(b))
		return
	}
	if os.Args[1] == "mutex-abandon" {
		name, _ := windows.UTF16PtrFromString(os.Args[2])
		windows.CreateMutex(nil, true, name)
		os.WriteFile(filepath.Join(dir, "mutex-ready"), []byte("ready"), 0600)
		for {
			time.Sleep(time.Second)
		}
	}
	if os.Args[1] == "child" || os.Args[1] == "grandchild" {
		recordPID()
		if os.Args[1] == "child" {
			exe, _ := os.Executable()
			cmd := exec.Command(exe, "grandchild")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Start()
		}
		for {
			time.Sleep(time.Second)
		}
	}
	switch os.Args[1] {
	case "diagnose":
		if httpMode != "" && flag("--bundle") != "" {
			if err := writeSmokeBundle(flag("--bundle")); err != nil {
				cliError("storage_failed", "diagnose", "bundle", false, "Choose a new path.", 7)
			}
			event("support_bundle", uiBundlePreview(true))
			return
		}
		if mode == "ui" || mode == "ui-fail" {
			if flag("--bundle") != "" || hasFlag("--preview") {
				saved := false
				if path := flag("--bundle"); path != "" {
					if err := writeUIBundle(path); err != nil {
						os.Exit(1)
					}
					saved = true
				}
				event("support_bundle", uiBundlePreview(saved))
				return
			}
		}
		event("diagnose", map[string]any{"schema_version": 1, "schemaVersion": "1.0.0", "engineRevision": "fake", "timestamp": now(), "status": "PASS", "checks": []any{}, "warnings": []any{}, "errors": []any{}})
		return
	case "status":
		if hasFlag("--watch") {
			statusWatch()
			return
		}
		if os.Getenv("BUBO_FAKE_RECORD_STATUS") == "1" {
			f, err := os.OpenFile(filepath.Join(dir, "status-oneshots.txt"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
			if err != nil {
				panic(err)
			}
			fmt.Fprintln(f, os.Getpid())
			f.Close()
		}
		delayStatus()
		if _, err := os.Stat(filepath.Join(dir, "lose-status")); err == nil {
			os.Exit(1)
		}
		if mode == "bad-product" {
			product["name"] = "another-product"
		}
		if name := os.Getenv("BUBO_FAKE_STATUS_FIXTURE"); name != "" {
			kind, payload := replay(name)
			event(kind, payload)
			return
		}
		event("status", statusPayload())
		return
	case "stop":
		i := readInstance()
		if i == nil || i["instance_id"] != flag("--instance-id") {
			cliError("control_access", "control", "stop", false, "Select a running instance.", 16)
		}
		wait := 45000
		if v := flag("--wait-ms"); v != "" {
			wait, _ = strconv.Atoi(v)
		}
		os.WriteFile(filepath.Join(dir, "stop"), []byte("stop"), 0600)
		id := map[string]any{"instance_id": i["instance_id"], "pid": i["pid"], "process_creation_time": i["process_creation_time"], "exe_path": i["exe_path"]}
		event("stopping", map[string]any{"schema_version": 1, "control_version": 1, "action": "stop", "identity": id, "acknowledged": true, "stopped": false, "process_exited": false})
		deadline := time.Now().Add(time.Duration(wait) * time.Millisecond)
		for live(i) {
			if time.Now().After(deadline) {
				cliError("stop_timeout", "control", "stop", true, "The stop deadline expired.", 17)
			}
			time.Sleep(10 * time.Millisecond)
		}
		event("stopped", map[string]any{"identity": id, "process_exited": true, "already_stopped": false})
		return
	case "models":
		modelsCommand()
		return
	case "prepare":
		prepare()
		return
	case "serve":
		serve()
	default:
		os.Exit(2)
	}
}

func prepare() {
	if hasFlag("--catalogue-id") {
		catalogue()
		return
	}
	runtimeOnly := len(os.Args) > 2 && os.Args[2] == "runtime"
	digest := strings.Repeat("a", 64)
	events := os.Getenv("BUBO_FAKE_REPLAY")
	if events != "" {
		// Stream real fixtures. The fixture operation id equals this process's.
		for _, name := range strings.Split(events, ",") {
			kind, payload := replay(name)
			event(kind, payload)
			if kind == "progress" {
				time.Sleep(progressPause(0))
			}
			if kind == "approval_required" {
				digest = payload["plan_digest"].(string)
				if !approve(digest) {
					return
				}
			}
		}
		return
	}
	component := "model"
	if runtimeOnly || os.Getenv("BUBO_FAKE_PREPARE_COMPONENT") == "runtime" {
		component = "runtime"
	}
	event("preflight", map[string]any{"component": component, "phase": "preflight"})
	bindings := map[string]string{component: digest}
	if mode == "ui" || mode == "ui-fail" {
		if !fakeRuntimePrepared() && !runtimeOnly && os.Getenv("BUBO_FAKE_PREPARE_COMPONENT") == "" {
			bindings = map[string]string{"model": digest, "runtime": strings.Repeat("b", 64)}
			uiPlan("model", digest)
			uiPlan("runtime", bindings["runtime"])
		} else {
			uiPlan(component, digest)
		}
	} else {
		event("plan", map[string]any{"component": component, "ready": false, "sources": []any{}, "licenses": []any{}, "paths": map[string]any{}, "volumes": []any{}, "plan_digest": digest, "record_kind": "plan", "part": 0, "parts": 1})
	}
	binding := approvalDigest(bindings)
	components := []string{component}
	if len(bindings) == 2 {
		components = []string{"model", "runtime"}
	}
	approval := map[string]any{"plan_digest": binding, "components": components}
	event("approval_required", approval)
	if !approve(binding) {
		return
	}
	if countText := os.Getenv("BUBO_FAKE_PROGRESS_REPLAY_COUNT"); countText != "" {
		count, err := strconv.Atoi(countText)
		if err != nil || count < 1 || count > 200 {
			panic("BUBO_FAKE_PROGRESS_REPLAY_COUNT must be 1..200")
		}
		for n := 0; n < count; n++ {
			name := "progress"
			if n%2 == 0 {
				name = "progress-unknown"
			}
			kind, payload := replay(name)
			event(kind, payload)
			time.Sleep(progressPause(25 * time.Millisecond))
		}
		os.WriteFile(filepath.Join(dir, "prepared"), []byte("true"), 0600)
		event("result", map[string]any{"exit_code": 0, "manifest_path": "C:/fake/manifest.json", "prepared": true})
		return
	}
	for n := 0; n <= 5; n++ {
		var total any = 5
		if (mode == "ui" || mode == "ui-fail") && n == 0 {
			total = nil
		}
		event("progress", map[string]any{"component": component, "phase": "verify", "file_id": "fixture", "bytes_done": n, "bytes_total": total, "resume_offset": 0, "verification_method": "sha256"})
		if os.Getenv("BUBO_FAKE_ENGINE_STALL") == "1" && n == 0 {
			// No later progress record: the supervisor must enforce the idle deadline.
			// A cancel line ends the run as the real CLI does, and leaves a mark.
			lines := make(chan string, 4)
			go func() {
				scanner := bufio.NewScanner(os.Stdin)
				for scanner.Scan() {
					lines <- scanner.Text()
				}
			}()
			select {
			case line := <-lines:
				if strings.Contains(line, "cancel") {
					os.WriteFile(filepath.Join(dir, "cancel-received"), []byte(line), 0600)
					cliError("cancelled", "prepare", "download", false, "Cancelled.", 3)
				}
			case <-time.After(60 * time.Second):
			}
			return
		}
		pause := 10 * time.Millisecond
		if mode == "ui" || mode == "ui-fail" {
			pause = 80 * time.Millisecond
			if n == 0 {
				pause = 500 * time.Millisecond
			}
		}
		time.Sleep(progressPause(pause))
	}
	if runtimeOnly {
		os.WriteFile(filepath.Join(dir, "runtime-prepared"), []byte("true"), 0600)
		event("result", map[string]any{"exit_code": 0, "manifest_path": nil, "prepared": true})
	} else {
		os.WriteFile(filepath.Join(dir, "prepared"), []byte("true"), 0600)
		event("result", map[string]any{"exit_code": 0, "manifest_path": "C:/fake/manifest.json", "prepared": true})
	}
}

// approve reads one input line and checks it as the real CLI does.
func approve(digest string) bool {
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return false
	}
	if os.Getenv("BUBO_FAKE_RECORD_RUNTIME") == "1" {
		f, err := os.OpenFile(filepath.Join(dir, "prepare-approval.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			panic(err)
		}
		_ = json.NewEncoder(f).Encode(map[string]any{"received": scanner.Text() != ""})
		_ = f.Close()
	}
	var input map[string]any
	json.Unmarshal(scanner.Bytes(), &input)
	if input["action"] != "approve" || input["operation_id"] != op || input["plan_digest"] != digest {
		// The real CLI ends a declined plan like a cancel: error "cancelled", exit 3.
		if input["action"] == "cancel" || input["action"] == "decline" {
			cliError("cancelled", "prepare", "approval", false, "Run Prepare when needed.", 3)
		}
		event("result", map[string]any{"exit_code": 4, "manifest_path": nil, "prepared": false})
		return false
	}
	return true
}

// servedModelName follows the merged CLI: an explicit name, a prepared package
// id, or the loose GGUF stem. The app passes no name of its own.
func servedModelName() string {
	if v := flag("--served-model-name"); v != "" {
		return v
	}
	model := flag("--model")
	_, listing := recordedOne("models-list-gui.json")
	for _, raw := range listing["prepared_models"].([]any) {
		prepared := raw.(map[string]any)
		if model == "" && prepared["manifest_path"] == recordedStatus(true)["prepared_default"] {
			return prepared["id"].(string)
		}
		if path, ok := prepared["model_path"].(string); ok && model != "" && sameFolder(path, model) {
			return prepared["id"].(string)
		}
	}
	return strings.TrimSuffix(filepath.Base(model), filepath.Ext(model))
}

func serve() {
	if hasFlag("--list-switches") {
		if path := os.Getenv("BUBO_FAKE_REGISTRY"); path != "" {
			b, err := os.ReadFile(path)
			if err != nil {
				cliError("unexpected_failure", "fake", "registry", false, "Cannot read test registry.", 1)
			}
			fmt.Println(string(b))
			return
		}
		// The recorded registry of the candidate exe, byte for byte.
		printRecording("switches.json")
		return
	}
	if hasFlag("--instance-id") {
		cliError("invalid_arguments", "cli", "serve", false, "serve creates its own instance id.", 2)
	}
	if mode == "reject-port" && flag("--port") != "19091" {
		cliError("invalid_arguments", "cli", "serve", false, "Choose a supported port.", 2)
	}
	os.Remove(filepath.Join(dir, "stop"))
	// Every serve start leaves its command line, so a test can prove that a
	// restart used the same setup.
	if f, err := os.OpenFile(filepath.Join(dir, "serve-argv.txt"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600); err == nil {
		fmt.Fprintln(f, strings.Join(os.Args[1:], " "))
		f.Close()
	}
	recordPID()
	instance = randomID()
	creation = created()
	exe := selfExe
	host := flag("--host")
	if host == "" {
		host = "127.0.0.1"
	}
	port := flag("--port")
	if port == "" {
		port = "19091"
	} // independent fake default, no real port is bound
	endpoint := "http://" + host + ":" + port + "/v1"
	// Record the received command line (the child side of the key hand-over).
	if b, err := json.Marshal(os.Args[1:]); err == nil {
		os.WriteFile(filepath.Join(dir, "argv.json"), b, 0600)
	}
	recordEnv()
	if names := os.Getenv("BUBO_FAKE_RECORD_ENV"); names != "" {
		received := map[string][]string{}
		for _, name := range strings.Split(names, ",") {
			for _, entry := range os.Environ() {
				key, value, _ := strings.Cut(entry, "=")
				if strings.EqualFold(name, key) {
					received[name] = append(received[name], value)
				}
			}
		}
		b, _ := json.Marshal(received)
		os.WriteFile(filepath.Join(dir, "override-env.json"), b, 0600)
	}
	if mode == "reject-override" && flag("--g4-value") == "reject" {
		event("error", map[string]any{"code": "invalid_arguments", "component": "serve", "stage": "startup", "retryable": false, "remedy": "Check the settings and model files.", "switch": "--g4-value"})
		os.Exit(2)
	}
	if mode == "leak-key" {
		// A misbehaving CLI that echoes its command line and its key
		// variable everywhere.
		fmt.Fprintln(os.Stderr, "argv:", strings.Join(os.Args[1:], " "))
		defer fmt.Fprintln(os.Stderr, "argv again:", strings.Join(os.Args[1:], " "))
		fmt.Fprintln(os.Stderr, "env:", leakEnvValue())
	}
	effective := effectiveFromArgs()
	if mode == "ui" || mode == "ui-fail" {
		effective = uiEffective()
	}
	i := map[string]any{"instance_id": instance, "pid": os.Getpid(), "process_creation_time": creation, "phase": "starting", "endpoint": endpoint, "effective_settings": effective, "exe_path": exe, "served_model": servedModelName(), "last_transition": now(), "ready": false, "authenticated": hasFlag("--api-key") || len(keyFromEnv()) > 0}
	save(i)
	// A slow start: the process exists but has sent no lifecycle record yet.
	if ms, _ := strconv.Atoi(os.Getenv("BUBO_FAKE_START_DELAY_MS")); ms > 0 {
		time.Sleep(time.Duration(ms) * time.Millisecond)
	}
	life("starting", nil)
	if packagesMode() {
		packagesServeCheck()
	}
	if mode == "serve-runtime-missing" {
		// _source fixture: replay the contract's recorded runtime_missing
		// payload, with this process's live operation identity.
		kind, payload := replay("error-runtime_missing")
		event(kind, payload)
		os.Exit(11)
	}
	i["phase"] = "loading"
	save(i)
	loadMessage := "Ã©æ°´"
	if mode == "leak-key" {
		loadMessage = "argv " + strings.Join(os.Args[1:], " ") + " " + leakEnvValue()
	}
	life("loading", map[string]any{"message": loadMessage})
	if os.Getenv("BUBO_FAKE_PORT_OCCUPIED") == "1" {
		// _source: fixture, tests/cli/contract/valid/error-port_occupied.json.
		kind, payload := replay("error-port_occupied")
		event(kind, payload)
		os.Exit(11)
	}
	if os.Getenv("BUBO_FAKE_MEMORY_INSUFFICIENT") == "1" {
		// Serve exit 19 (CLI-CONTRACT, ms-memguard e4588b7e): the sessions at this
		// context do not fit.
		event("error", map[string]any{"code": "memory_insufficient", "component": "serve", "stage": "startup", "retryable": false,
			"remedy": "Lower --context or --sessions, or free memory.", "detail": "2 sessions at 262144 tokens need 47.3 GiB; 31.2 GiB is available.",
			"needed_bytes": 50784518144, "available_bytes": 33503973376, "sessions": 2, "context": 262144, "per_session_bytes": 25392259072, "floor_bytes": 0})
		os.Exit(19)
	}
	if mode == "before-ready" {
		os.Exit(23)
	}
	if mode == "invalid-json" {
		fmt.Println("this is not JSON")
		for {
			time.Sleep(time.Second)
		}
	}
	if mode == "truncated" {
		fmt.Print(`{"schema_version":1`)
		return
	}
	if mode == "spawn" || mode == "inherited" {
		cmd := exec.Command(exe, "child")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Start()
		if mode == "inherited" {
			return
		}
	}
	if mode == "hang-load" {
		for {
			life("warning", map[string]any{"code": "still_loading", "remedy": "Wait for the load."})
			time.Sleep(30 * time.Millisecond)
		}
	}
	if httpMode != "" {
		startFakeHTTP(host, flag("--port"))
	}
	if ms, _ := strconv.Atoi(os.Getenv("BUBO_FAKE_LOAD_MS")); ms > 0 {
		time.Sleep(time.Duration(ms) * time.Millisecond)
	}
	i["phase"] = "ready"
	i["ready"] = true
	i["last_transition"] = now()
	save(i)
	life("ready", map[string]any{"endpoint": endpoint, "served_model": servedModelName(), "effective_settings": effective, "log_path": nil, "readiness": map[string]any{"runtime": true, "model": true, "server": true}})
	if code := os.Getenv("BUBO_FAKE_WARNING"); code != "" {
		// A lifecycle warning the server sends while it runs (CLI-CONTRACT).
		life("warning", map[string]any{"code": code, "remedy": "Wait for a free session or lower the load."})
		life("warning", map[string]any{"code": code, "remedy": "Wait for a free session or lower the load."})
	}
	if source := os.Getenv("BUBO_FAKE_SERVER_LOG_FIXTURE"); source != "" {
		// The file contains only path-free [INFO] lines from the recorded
		// 2026-10-15 07:50 GPU smoke server log (W17).
		lines, err := os.ReadFile(source)
		if err != nil {
			panic(err)
		}
		if _, err := os.Stderr.Write(lines); err != nil {
			panic(err)
		}
	}
	if mode == "lose-pipe" {
		os.Stdout.Close()
		for {
			time.Sleep(time.Second)
		}
	}
	if mode == "flood" {
		go func() {
			for n := 0; n < 40000; n++ {
				fmt.Fprintln(os.Stderr, strings.Repeat("x", 1024))
				life("warning", map[string]any{"code": "flood", "remedy": "Ã©æ°´"})
			}
		}()
	}
	for {
		scriptedExit(false)
		if mode == "ui-fail" {
			if _, err := os.Stat(filepath.Join(dir, "fail")); err == nil {
				cliError("device_lost", "serve", "decode", true, "Restart on request.", 17)
			}
		}
		if b, err := os.ReadFile(filepath.Join(dir, "emit-request")); err == nil {
			os.Remove(filepath.Join(dir, "emit-request"))
			var row map[string]any
			if json.Unmarshal(b, &row) == nil {
				event("request_completed", row)
			}
		}
		if _, err := os.Stat(filepath.Join(dir, "stop")); err == nil && mode != "hang-stop" && mode != "spawn" {
			life("stopping", nil)
			scriptedExit(true)
			life("stopped", nil)
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// scriptedExit ends serve with a chosen status. The file "exit-serve" in the
// state folder holds "<status> [once] [on-stop] [late]": "once" is consumed by
// the first serve that obeys it; "on-stop" applies only while a stop is
// running (a server that loses its GPU while the user stops it); "late" closes
// stdout 300 ms before the exit (a busy machine, or a large process tearing
// down). Status 75 is the engine's device_lost exit (EX_TEMPFAIL); it logs the
// shutdown line first.
func scriptedExit(stopping bool) {
	path := filepath.Join(dir, "exit-serve")
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	fields := strings.Fields(string(b))
	if len(fields) == 0 {
		return
	}
	code, err := strconv.Atoi(fields[0])
	if err != nil {
		return
	}
	options := map[string]bool{}
	for _, option := range fields[1:] {
		options[option] = true
	}
	if options["on-stop"] != stopping {
		return
	}
	if options["once"] {
		os.Remove(path)
	}
	if code == 75 {
		fmt.Fprintln(os.Stderr, "[ERROR] event=device_lost_shutdown")
	}
	if options["late"] {
		os.Stdout.Close()
		time.Sleep(300 * time.Millisecond)
	}
	os.Exit(code)
}
