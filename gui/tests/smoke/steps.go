//go:build windows

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/windows"

	"bubo/gui/supervisor"
)

// ---- step 1: preflight ----

var requiredCapabilities = []string{"status_json", "instance_control", "machine_readings", "request_history", "connection_helpers", "memory_estimate", "support_bundle"}

func (r *Runner) preflight(ctx context.Context, s *Step) {
	cfg := r.cfg
	if !s.Check("exe", fileExists(cfg.Exe), "%s", cfg.Exe) || !s.Check("manifest", fileExists(cfg.Manifest), "%s", cfg.Manifest) {
		return
	}
	s.Set("mode", cfg.mode())
	s.Set("production_up_flag", cfg.ProductionUp)
	// Binary hashes (B1): every exe and dll next to bubo.exe, plus this tool.
	hashes, herr := binaryHashes(filepath.Dir(cfg.Exe))
	s.Check("binary hashes", herr == nil && len(hashes) > 0, "%d exe/dll files hashed in %s (%v)", len(hashes), filepath.Dir(cfg.Exe), herr)
	s.Set("binary_sha256", hashes)
	if self, serr := os.Executable(); serr == nil {
		if h, e := hashFile(self); e == nil {
			s.Set("bubo_smoke_sha256", h)
		}
	}
	pins, perr := parsePins(cfg.Pins)
	if !s.Check("pins syntax", perr == nil, "%v", perr) {
		return
	}
	for _, name := range sortedKeys(pins) {
		got, have := hashes[name]
		s.Check("pin "+name, have && got == pins[name], "file %s sha256 %s (pinned %s)", name, got, pins[name])
	}
	s.Check("port allowed", cfg.Port < 8081 || cfg.Port > 8086, "port %d", cfg.Port)
	s.Check("port free", portFree(cfg.Port) == nil, "127.0.0.1:%d binds", cfg.Port)
	if len(r.scrubbed) > 0 {
		s.Set("env_scrubbed_names", r.scrubbed)
	}
	ev, err := r.cli(ctx, cfg.QueryTimeout, "status", "--json")
	if !s.Check("status handshake", err == nil, "%v", err) {
		return
	}
	st := ev.Value.(*supervisor.Status)
	s.Check("product", ev.Product.Name == "bubo" && versionPattern.MatchString(ev.Product.Version), "%s %s revision %s engine %s schema %d", ev.Product.Name, ev.Product.Version, ev.Product.Revision, ev.Product.EngineVersion, ev.SchemaVersion)
	s.Set("versions", ev.Product)
	var payload map[string]json.RawMessage
	_ = json.Unmarshal(ev.Payload, &payload)
	caps, _ := decodeAny(payload["capabilities"]).(map[string]any)
	var missing []string
	for _, k := range requiredCapabilities {
		if v, _ := caps[k].(bool); !v {
			missing = append(missing, k)
		}
	}
	s.Check("capabilities", len(missing) == 0, "required %v; missing %v", requiredCapabilities, missing)
	s.Set("capabilities", caps)
	lastCode := ""
	if st.LastError != nil {
		lastCode = st.LastError.Code
	}
	s.Check("prepared", st.Prepared, "prepared=%t phase=%s last_error=%q", st.Prepared, st.Phase, lastCode)
	sel := st.ModelSelection
	s.Check("manifest resolves", sel != nil && sel.Manifest != nil && sel.Source == "environment" && normPath(*sel.Manifest) == normPath(cfg.Manifest),
		"selection source %q, resolved manifest matches --manifest", func() string {
			if sel == nil {
				return "absent"
			}
			return sel.Source
		}())
	if st.PreparedDefault != nil {
		s.Set("prepared_default", *st.PreparedDefault)
	}
	// Gate that production being up is allowed to fail (--preflight-only --production-up
	// before the production window): the value is recorded, the check passes with a label.
	gate := func(name string, ok bool, format string, args ...any) {
		if cfg.ProductionUp && !ok {
			s.Check(name, true, "RECORDED ONLY (--production-up): "+format, args...)
			return
		}
		s.Check(name, ok, format, args...)
	}
	gate("no live instance", len(st.Instances) == 0, "%d live bubo instance(s) in discovery; stop them first (the smoke never touches them)", len(st.Instances))
	procs, lerr := listProcs(cfg.ProcPatterns)
	if s.Check("process list readable", lerr == nil, "%v", lerr) {
		var fixtures []ProcInfo
		procs, fixtures = splitTestFixtures(procs)
		s.Set("matching_processes", procs)
		s.Set("ignored_test_fixture_processes", fixtures)
		gate("no bubo/gufo/llama process", len(procs) == 0, "%d process(es) match %v: %s", len(procs), cfg.ProcPatterns, procText(procs))
	}
	s.Set("lock_state", map[string]any{"gpu": lockSummary(cfg.LockPath), "build": lockSummary(cfg.BuildLockPath)})
	total, avail, merr := hostMemory()
	s.Check("host memory read", merr == nil, "%v", merr)
	s.Set("host_ram_total_gib", gib(total))
	s.Set("host_ram_available_gib", gib(avail))
	if cfg.MinFreeRAMGiB > 0 {
		gate("free RAM floor", gib(avail) >= cfg.MinFreeRAMGiB, "%.1f GiB available, floor %.1f GiB", gib(avail), cfg.MinFreeRAMGiB)
	}
	for name, p := range map[string]string{"out": cfg.Out, "manifest": filepath.Dir(cfg.Manifest)} {
		f, derr := freeDisk(p)
		s.Check("disk free "+name, derr == nil, "%.1f GiB free", gib(f))
		s.Set("disk_free_gib_"+name, gib(f))
	}
	// Power: read only. Battery fails (the plan needs AC); unknown is recorded.
	power := "unknown"
	var machine any
	if raw, ok := payload["machine"]; ok {
		machine = decodeAny(raw)
	}
	if v, ok := dig(machine, "windows", "power_source", "value").(string); ok {
		power = v
	}
	mode := dig(machine, "windows", "effective_power_mode", "windows_enum_name")
	gpuW := dig(machine, "gpu", "power", "value")
	s.Set("power_source", power)
	s.Set("effective_power_mode", mode)
	s.Set("gpu_power_w", gpuW)
	if m, ok := mode.(string); ok {
		s.Set("power_mode_is_performance", strings.Contains(strings.ToLower(m), "performance")) // recorded warning only
	}
	switch {
	case cfg.AllowBattery:
		s.Check("AC power", true, "power source %s (--allow-battery)", power)
	case power == "battery":
		s.Check("AC power", false, "the laptop is on battery; connect AC")
	case power == "unknown":
		s.Check("AC power", false, "the CLI cannot read the power source (unknown); fix that or pass --allow-battery")
	default:
		s.Check("AC power", true, "power source %s", power)
	}
	s.Summary("bubo %s rev %s; capabilities complete; prepared; port %d free; %.0f GiB RAM available; power %s", ev.Product.Version, ev.Product.Revision, cfg.Port, gib(avail), power)
}

// ---- step 2: start through the supervisor ----

type settingRow struct {
	Requested any    `json:"requested"`
	Effective any    `json:"effective"`
	Source    string `json:"source"`
	Reason    any    `json:"reason"`
}

func rowText(raw json.RawMessage) string {
	var row settingRow
	if json.Unmarshal(raw, &row) != nil {
		return "unreadable"
	}
	t := fmt.Sprintf("%v", row.Effective)
	if row.Reason != nil {
		t += fmt.Sprintf(" (reason: %v)", row.Reason)
	}
	return t
}

func (r *Runner) stepStart(ctx context.Context, s *Step) {
	sup, err := r.newSup("run1", true)
	if !s.Check("supervisor", err == nil, "%v", err) {
		return
	}
	t0 := time.Now()
	if !s.Check("start accepted", reply(ctx, sup.Start("--manifest", r.cfg.Manifest), time.Minute) == nil, "Start returned without an error") {
		return
	}
	v, err := r.waitView(ctx, sup, r.cfg.LoadTimeout+90*time.Second, func(v supervisor.View) (bool, error) {
		switch v.State {
		case supervisor.Ready:
			return v.Instance != nil && v.Instance.ID != "", nil
		case supervisor.Failed:
			return false, fmt.Errorf("supervisor state Failed: %s", lastErr(v))
		case supervisor.Stopped:
			return false, fmt.Errorf("supervisor returned to Stopped: %s", lastErr(v))
		}
		return false, nil
	})
	if !s.Check("verified Ready", err == nil, "%v", err) {
		return
	}
	ready := time.Since(t0)
	in := v.Instance
	s.Set("ready_seconds", ready.Seconds())
	s.Set("instance_id", in.ID)
	s.Set("pid", in.PID)
	s.Set("process_creation_time", in.CreationTime)
	s.Check("process alive", isLive(in.PID, in.CreationTime), "pid %d, created %s", in.PID, in.CreationTime)
	recorded := false
	for _, p := range r.led.Snapshot() {
		if p.PID == in.PID && p.CreationTime == in.CreationTime {
			recorded = true
		}
	}
	s.Check("pid in ledger", recorded, "pid %d is recorded for cleanup", in.PID)
	want := fmt.Sprintf("http://127.0.0.1:%d/v1", r.cfg.Port)
	ep := ""
	if in.Endpoint != nil {
		ep = *in.Endpoint
	}
	s.Check("endpoint", ep == want, "endpoint %s (want %s)", ep, want)
	s.Check("served model", in.Model != "", "served model %q", in.Model)
	s.Check("no error", v.Failure == nil && !viewForced(v), "last error: %s", lastErr(v))
	r.endpoint, r.model, r.instanceID = ep, in.Model, in.ID
	s.Set("endpoint", ep)
	s.Set("served_model", in.Model)
	eff := map[string]string{}
	for k, raw := range in.Effective {
		eff[k] = rowText(viewRaw(raw))
	}
	s.Set("effective_settings", eff)
	for _, k := range []string{"profile", "context", "sessions"} {
		raw, ok := in.Effective[k]
		s.Check("setting "+k, ok, "%s", func() string {
			if ok {
				return rowText(viewRaw(raw))
			}
			return "missing"
		}())
	}
	npu := "not reported"
	for k, raw := range in.Effective {
		if strings.Contains(strings.ToLower(k), "npu") {
			npu = k + " = " + rowText(viewRaw(raw))
		}
	}
	s.Set("npu", npu)
	cacheRows, cacheOn := diskCacheState(viewRawSettings(in.Effective))
	s.Set("cache_rows", cacheRows)
	s.Check("disk cache off", len(cacheOn) == 0, "cache rows %v; not off: %v (the smoke passes no --cache-disk)", cacheRows, cacheOn)
	if len(cacheRows) == 0 {
		s.Set("disk_cache_proof", "NOT PROVEN by the CLI (no cache row reported); the smoke passes no --cache-disk and the user folder diff must show no cache file written")
	}
	r.ramAfterReady(s, "host_ram_available_gib_after_ready")
	// Baseline sample for steps 3 and 4.
	st, _, err := r.statusOf(ctx, in.ID)
	if !s.Check("status sample", err == nil, "%v", err) {
		return
	}
	s.Check("status identity", st.Identity != nil && st.Identity.ID == in.ID && st.Phase == "ready", "phase %s", st.Phase)
	r.base = st
	r.baseRing = len(st.Requests)
	mem := "no memory estimate"
	if st.MemoryEstimate != nil {
		me := st.MemoryEstimate
		mem = fmt.Sprintf("memory estimate %.1f-%.1f GiB (%s, fit_verified=%t)", me.TotalRange.Min.GiB, me.TotalRange.Max.GiB, me.Evidence, me.FitVerified)
		s.Set("memory_estimate_gib", map[string]any{"min": me.TotalRange.Min.GiB, "max": me.TotalRange.Max.GiB, "evidence": me.Evidence, "fit_verified": me.FitVerified})
	} else if st.MemoryEstimateReason != nil {
		mem = "memory estimate unavailable: " + *st.MemoryEstimateReason
		s.Set("memory_estimate_reason", *st.MemoryEstimateReason)
	}
	s.Set("memory_estimate", mem)
	s.Summary("Ready in %.1f s; pid %d; %s; model %s; profile %s; context %s; sessions %s; npu %s; %s", ready.Seconds(), in.PID, ep, in.Model,
		rowText(viewRaw(in.Effective["profile"])), rowText(viewRaw(in.Effective["context"])), rowText(viewRaw(in.Effective["sessions"])), npu, mem)
}

// ---- step 3: one bounded request, then the same prefix again ----

// ramAfterReady records the available host RAM right after a verified Ready.
func (r *Runner) ramAfterReady(s *Step, key string) {
	if total, avail, err := hostMemory(); err == nil {
		s.Set(key, gib(avail))
		r.rec.Logf("host RAM after Ready: %.1f of %.1f GiB available (%s)", gib(avail), gib(total), key)
	}
}

func (r *Runner) chatOnce(ctx context.Context, msgs []chatMessage, maxTokens int) (chatResult, error) {
	cctx, cancel := context.WithTimeout(ctx, r.cfg.RequestTimeout)
	defer cancel()
	return chat(cctx, r.client, r.endpoint, r.model, msgs, maxTokens)
}

func (r *Runner) checkChat(s *Step, label string, res chatResult, err error, maxTokens int) bool {
	ok := s.Check(label+" sent", err == nil, "%v", err)
	ok = s.Check(label+" HTTP status", res.HTTPStatus == 200, "HTTP %d (error code %q)", res.HTTPStatus, res.ErrorCode) && ok
	if res.HTTPStatus != 200 {
		return false
	}
	ok = s.Check(label+" tokens", res.PromptTokens > 0 && res.CompletionTokens >= 1 && res.CompletionTokens <= int64(maxTokens), "%d in / %d out (max %d)", res.PromptTokens, res.CompletionTokens, maxTokens) && ok
	ok = s.Check(label+" finish", res.FinishReason == "stop" || res.FinishReason == "length", "finish_reason %q", res.FinishReason) && ok
	ok = s.Check(label+" content", res.ContentBytes > 0, "%d content bytes (text not logged)", res.ContentBytes) && ok
	ok = s.Check(label+" timings", res.TTFTms >= 0 && res.DecodeTPS >= 0, "ttft %.1f ms, decode %.2f tok/s, prefill %.1f tok/s", res.TTFTms, res.DecodeTPS, res.PrefillTPS) && ok
	return ok
}

func (r *Runner) stepRequest(ctx context.Context, s *Step) {
	r.rec.Logf("request: %s", promptLabel())
	first := chatMessage{"user", promptFirstTurn()}
	r1, err1 := r.chatOnce(ctx, []chatMessage{first}, 64)
	ok1 := r.checkChat(s, "request 1", r1, err1, 64)
	if err1 != nil || r1.HTTPStatus != 200 {
		return
	}
	s.Set("request_1", chatSummary(r1))
	reuse := r1.Content
	if reuse == "" {
		reuse = "ready"
	}
	r2, err2 := r.chatOnce(ctx, []chatMessage{first, {"assistant", reuse}, {"user", promptAsk2}}, 64)
	ok2 := r.checkChat(s, "request 2", r2, err2, 64)
	if err2 != nil || r2.HTTPStatus != 200 {
		return
	}
	s.Set("request_2", chatSummary(r2))
	// Both requests returned HTTP 200: step 4 may run (see runAll).
	r.chats = []chatResult{r1, r2}
	if !ok1 || !ok2 {
		return
	}
	s.Check("cache reuse in usage", r2.CachedTokens > 0, "request 2 reused %d of %d prompt tokens (request 1 reused %d)", r2.CachedTokens, r2.PromptTokens, r1.CachedTokens)
	// The typed request ring must show both rows.
	var rows []supervisor.RequestRow
	deadline := time.Now().Add(20 * time.Second)
	for {
		st, _, serr := r.statusOf(ctx, r.instanceID)
		if serr == nil {
			rows = st.Requests
			if len(rows) >= r.baseRing+2 {
				break
			}
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if !s.Check("request ring rows", len(rows) >= r.baseRing+2, "ring has %d rows, expected at least %d", len(rows), r.baseRing+2) {
		return
	}
	row1, row2 := rows[len(rows)-2], rows[len(rows)-1]
	u := func(p *uint64) int64 {
		if p == nil {
			return -1
		}
		return int64(*p)
	}
	s.Check("ring row 1", row1.Outcome == "completed" && u(row1.PromptTokens) == r1.PromptTokens, "outcome %s, prompt tokens %d (HTTP said %d)", row1.Outcome, u(row1.PromptTokens), r1.PromptTokens)
	kind := ""
	if row2.CacheKind != nil {
		kind = *row2.CacheKind
	}
	s.Check("ring row 2", row2.Outcome == "completed" && u(row2.PromptTokens) == r2.PromptTokens && u(row2.CompletionTokens) == r2.CompletionTokens,
		"outcome %s, %d in / %d out (HTTP said %d / %d)", row2.Outcome, u(row2.PromptTokens), u(row2.CompletionTokens), r2.PromptTokens, r2.CompletionTokens)
	s.Check("cache reuse in ring", u(row2.CachedTokens) > 0 && (kind == "memory" || kind == "disk") && u(row2.CachedTokens) == r2.CachedTokens,
		"cached tokens %d kind %q (HTTP said %d)", u(row2.CachedTokens), kind, r2.CachedTokens)
	s.Set("ring_row_2", map[string]any{"cached_tokens": u(row2.CachedTokens), "cache_kind": kind, "ttft_ms": row2.TTFTMS, "decode_tps": row2.DecodeTPS, "prefill_tps": row2.PrefillTPS})
	s.Summary("req1 %d in/%d out, ttft %.0f ms, %.1f tok/s; req2 %d in/%d out, cached %d (%s), ttft %.0f ms, %.1f tok/s",
		r1.PromptTokens, r1.CompletionTokens, r1.TTFTms, r1.DecodeTPS, r2.PromptTokens, r2.CompletionTokens, u(row2.CachedTokens), kind, r2.TTFTms, r2.DecodeTPS)
}

func chatSummary(c chatResult) map[string]any {
	return map[string]any{"http_status": c.HTTPStatus, "prompt_tokens": c.PromptTokens, "completion_tokens": c.CompletionTokens, "cached_tokens": c.CachedTokens,
		"ttft_ms": c.TTFTms, "decode_tokens_per_second": c.DecodeTPS, "prefill_tokens_per_second": c.PrefillTPS, "wall_ms": c.WallMS, "finish_reason": c.FinishReason, "content_bytes": c.ContentBytes}
}

// ---- step 4: status counters and the Pi snippet ----

func cnt(c *supervisor.Counters, f func(*supervisor.Counters) *uint64) int64 {
	if c == nil || f(c) == nil {
		return -1
	}
	return int64(*f(c))
}

func (r *Runner) stepCounters(ctx context.Context, s *Step) {
	// Wait for the slots to go idle again.
	var st *supervisor.Status
	var idle bool
	var why string
	deadline := time.Now().Add(r.cfg.SlotIdleWait)
	for {
		var err error
		st, _, err = r.statusOf(ctx, r.instanceID)
		if err != nil {
			why = err.Error()
		} else {
			idle, why = slotsIdle(st)
			if idle {
				break
			}
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if !s.Check("status sample", st != nil, "%s", why) {
		return
	}
	s.Check("slots idle again", idle, "%s", why)
	b, a := r.base.Counters, st.Counters
	s.Check("counters present", a != nil && r.base.Counters != nil, "counters object present")
	if a == nil || b == nil {
		return
	}
	dPre := cnt(a, func(c *supervisor.Counters) *uint64 { return c.PrefillTokens }) - cnt(b, func(c *supervisor.Counters) *uint64 { return c.PrefillTokens })
	dDec := cnt(a, func(c *supervisor.Counters) *uint64 { return c.DecodeTokens }) - cnt(b, func(c *supervisor.Counters) *uint64 { return c.DecodeTokens })
	dCache := cnt(a, func(c *supervisor.Counters) *uint64 { return c.CachedTokens }) - cnt(b, func(c *supervisor.Counters) *uint64 { return c.CachedTokens })
	dHits := cnt(a, func(c *supervisor.Counters) *uint64 { return c.CacheMemoryHits }) + cnt(a, func(c *supervisor.Counters) *uint64 { return c.CacheDiskHits }) -
		cnt(b, func(c *supervisor.Counters) *uint64 { return c.CacheMemoryHits }) - cnt(b, func(c *supervisor.Counters) *uint64 { return c.CacheDiskHits })
	s.Check("prefill counter advanced", dPre > 0, "prefill_tokens_total +%d", dPre)
	s.Check("decode counter advanced", dDec > 0, "decode_tokens_total +%d", dDec)
	s.Check("cached counter advanced", dCache > 0, "cached_tokens_total +%d", dCache)
	s.Check("cache hit counted", dHits >= 1, "memory+disk hits +%d", dHits)
	if !s.Check("completed counter present", a.CompletedSuccessful != nil && b.CompletedSuccessful != nil, "completed_successful_requests_total present in both samples (NOT PROVEN when absent)") {
		s.Set("completed_requests_proof", "NOT PROVEN: the CLI did not report completed_successful_requests_total")
	} else {
		d := int64(*a.CompletedSuccessful) - int64(*b.CompletedSuccessful)
		s.Check("completed requests", d == 2, "completed_successful_requests_total +%d (expected +2)", d)
	}
	// Prediction error (informational): what the two HTTP replies imply.
	if len(r.chats) == 2 {
		expDec := r.chats[0].CompletionTokens + r.chats[1].CompletionTokens
		expPre := r.chats[0].PromptTokens - r.chats[0].CachedTokens + r.chats[1].PromptTokens - r.chats[1].CachedTokens
		s.Set("decode_delta", map[string]any{"expected": expDec, "observed": dDec, "error": dDec - expDec})
		s.Set("prefill_delta", map[string]any{"expected": expPre, "observed": dPre, "error": dPre - expPre})
		r.rec.Logf("prediction error: decode expected %d observed %d; prefill expected %d observed %d", expDec, dDec, expPre, dPre)
	}
	s.Check("request ring", len(st.Requests) >= r.baseRing+2, "%d rows", len(st.Requests))
	// Pi snippet: print the structure and the placeholder key only.
	pi := st.PiSnippet
	if !s.Check("pi snippet present", pi != nil, "pi_snippet object") {
		return
	}
	p := pi.Providers.Bubo
	placeholder := p.APIKey == "bubo-loopback" || p.APIKey == "BUBO_API_KEY"
	shown := "<not a placeholder; value withheld>"
	if placeholder {
		shown = p.APIKey
	}
	s.Check("pi snippet key is a placeholder", placeholder, "apiKey = %s", shown)
	s.Check("pi snippet fields", p.BaseURL == r.endpoint && p.API == "openai-completions" && len(p.Models) == 1 && p.Models[0].ID == r.model && p.Models[0].ContextWindow > 0,
		"baseUrl matches endpoint, api %q, %d model(s)", p.API, len(p.Models))
	s.Check("connection object", st.Connection != nil && st.Connection.Endpoint == r.endpoint && !st.Connection.Authenticated, "connection present, unauthenticated loopback")
	structure := fmt.Sprintf("providers.bubo{baseUrl=%s api=%s apiKey=%s models=[%d]{id name contextWindow}}", p.BaseURL, p.API, shown, len(p.Models))
	s.Set("pi_snippet_structure", structure)
	r.rec.Logf("pi snippet: %s", structure)
	s.Summary("prefill +%d, decode +%d, cached +%d, hits +%d; slots idle; ring %d rows; pi snippet key %s", dPre, dDec, dCache, dHits, len(st.Requests), shown)
}

func slotsIdle(st *supervisor.Status) (bool, string) {
	if len(st.Slots) == 0 {
		return false, "status has no slots"
	}
	for _, sl := range st.Slots {
		if sl.Processing || sl.State != "idle" {
			return false, fmt.Sprintf("slot %d is %s", sl.ID, sl.State)
		}
	}
	if st.Sessions == nil || st.Queue == nil {
		return false, "sessions or queue is null"
	}
	if *st.Sessions != 0 || *st.Queue != 0 {
		return false, fmt.Sprintf("sessions %d, queue %d", *st.Sessions, *st.Queue)
	}
	return true, fmt.Sprintf("%d slot(s) idle, sessions 0, queue 0", len(st.Slots))
}

// ---- step 5: idle watch ----

func (r *Runner) stepIdle(ctx context.Context, s *Step) {
	before, _, err := r.statusOf(ctx, r.instanceID)
	if !s.Check("status sample", err == nil, "%v", err) {
		return
	}
	end := time.Now().Add(r.cfg.IdleWatch)
	var sawStall string
	polls := 0
	for time.Now().Before(end) && ctx.Err() == nil {
		v := r.view(r.sup)
		polls++
		if v.Stall != nil || strings.HasPrefix(v.Reason, "No token progress for ") {
			sawStall = v.Reason
			break
		}
		if v.State != supervisor.Ready {
			sawStall = "state left Ready: " + string(v.State)
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	s.Check("no stall warning", sawStall == "", "watched %s, %d snapshots %s", r.cfg.IdleWatch, polls, sawStall)
	v := r.view(r.sup)
	s.Check("watchdog had samples", v.Product != nil && !v.Stale, "supervisor holds a fresh status sample")
	after, _, err := r.statusOf(ctx, r.instanceID)
	if s.Check("status sample after", err == nil, "%v", err) {
		dec := func(c *supervisor.Counters) *uint64 { return c.DecodeTokens }
		b4, af := cnt(before.Counters, dec), cnt(after.Counters, dec)
		s.Check("idle counters read", b4 >= 0 && af >= 0, "decode_tokens_total read before %d and after %d (-1 = counter absent, NOT PROVEN)", b4, af)
		s.Check("idle counters still", b4 >= 0 && af >= 0 && af-b4 == 0, "decode_tokens_total changed by %d while idle", af-b4)
		idle, why := slotsIdle(after)
		s.Check("slots idle", idle, "%s", why)
	}
	s.Summary("no stall warning in %s idle; state Ready; counters unchanged", r.cfg.IdleWatch)
}

// ---- step 6: stop through the supervisor, then a restart cycle ----

func (r *Runner) stopOwned(ctx context.Context, s *Step, label string) (time.Duration, bool) {
	before := r.view(r.sup).Instance
	if before == nil {
		s.Check(label+" instance", false, "no owned instance to stop")
		return 0, false
	}
	pid, created := before.PID, before.CreationTime
	t0 := time.Now()
	if !s.Check(label+" stop accepted", reply(ctx, r.sup.Stop(), time.Minute) == nil, "Stop returned without an error") {
		return 0, false
	}
	v, err := r.waitView(ctx, r.sup, r.cfg.StopTimeout+45*time.Second, func(v supervisor.View) (bool, error) {
		switch v.State {
		case supervisor.Stopped:
			return true, nil
		case supervisor.Failed:
			return false, fmt.Errorf("supervisor state Failed (forced=%t): %s", viewForced(v), lastErr(v))
		}
		return false, nil
	})
	took := time.Since(t0)
	ok := s.Check(label+" reached Stopped", err == nil, "%v", err)
	if err == nil {
		ok = s.Check(label+" graceful", !viewForced(v) && v.Failure == nil, "forced=%t, last error %s", viewForced(v), lastErr(v)) && ok
	}
	ok = s.Check(label+" process gone", !isLive(pid, created), "pid %d", pid) && ok
	if st, _, serr := r.statusOf(ctx, ""); s.Check(label+" status after stop", serr == nil, "%v", serr) {
		ok = s.Check(label+" no instance left", len(st.Instances) == 0, "%d instance(s) in discovery", len(st.Instances)) && ok
	}
	return took, ok
}

func (r *Runner) stepStopRestart(ctx context.Context, s *Step) {
	stop1, ok := r.stopOwned(ctx, s, "stop")
	s.Set("stop_seconds", stop1.Seconds())
	if !ok {
		s.Check("restart cycle", false, "skipped because the first stop did not pass")
		return
	}
	t0 := time.Now()
	if !s.Check("restart: start accepted", reply(ctx, r.sup.Start("--manifest", r.cfg.Manifest), time.Minute) == nil, "Start returned without an error") {
		return
	}
	v, err := r.waitView(ctx, r.sup, r.cfg.LoadTimeout+90*time.Second, func(v supervisor.View) (bool, error) {
		switch v.State {
		case supervisor.Ready:
			return v.Instance != nil && v.Instance.ID != "", nil
		case supervisor.Failed, supervisor.Stopped:
			return false, fmt.Errorf("supervisor state %s: %s", v.State, lastErr(v))
		}
		return false, nil
	})
	if !s.Check("restart: verified Ready", err == nil, "%v", err) {
		return
	}
	readyAfter := time.Since(t0)
	in := v.Instance
	s.Check("restart: new identity", in.ID != r.instanceID, "instance id changed")
	if in.Endpoint != nil {
		r.endpoint = *in.Endpoint
	}
	r.model, r.instanceID = in.Model, in.ID
	r.ramAfterReady(s, "restart_host_ram_available_gib_after_ready")
	res, err := r.chatOnce(ctx, []chatMessage{{"user", promptAsk1}}, 8)
	if r.checkChat(s, "restart: tiny request", res, err, 8) {
		s.Set("restart_request", chatSummary(res))
	}
	stop2, _ := r.stopOwned(ctx, s, "restart: stop")
	s.Set("restart_ready_seconds", readyAfter.Seconds())
	s.Set("restart_stop_seconds", stop2.Seconds())
	s.Summary("stop %.1f s graceful; restart Ready in %.1f s, tiny request %d out, stop %.1f s graceful", stop1.Seconds(), readyAfter.Seconds(), res.CompletionTokens, stop2.Seconds())
}

// ---- step 7: reconnect to a server started outside the supervisor ----

type external struct {
	cmd      *exec.Cmd
	pid      uint32
	creation string
	done     chan struct{}
	exitCode int
	mu       sync.Mutex
	events   []string
	ready    chan struct{}
	id       string
}

func (e *external) seen(typ string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, t := range e.events {
		if t == typ {
			return true
		}
	}
	return false
}

func (r *Runner) startExternal() (*external, error) {
	args := []string{"serve", "--jsonl", "--host", "127.0.0.1", "--port", fmt.Sprint(r.cfg.Port), "--manifest", r.cfg.Manifest}
	cmd := exec.Command(r.cfg.Exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
	// M6: the planted fake key is placed in this server's environment only (not
	// BUBO_API_KEY, which would switch key mode on). If the server echoes its
	// environment into stderr, status, history or logs, step 9 finds it.
	cmd.Env = append(os.Environ(), "GUFO_EVAL_API_KEY="+r.plantedKey(), "OPENAI_API_KEY="+r.plantedKey())
	errFile, err := os.Create(filepath.Join(r.cfg.Out, "external-serve.stderr.log"))
	if err != nil {
		return nil, err
	}
	cmd.Stderr = errFile
	// An own pipe, not StdoutPipe: Wait must not close the reader before the
	// last lifecycle events (stopping, stopped) are read.
	out, pipeW, err := os.Pipe()
	if err != nil {
		errFile.Close()
		return nil, err
	}
	cmd.Stdout = pipeW
	if err = cmd.Start(); err != nil {
		errFile.Close()
		out.Close()
		pipeW.Close()
		return nil, err
	}
	pipeW.Close()
	pid := uint32(cmd.Process.Pid)
	created, cerr := creationTimeOf(pid)
	e := &external{cmd: cmd, pid: pid, creation: created, done: make(chan struct{}), ready: make(chan struct{})}
	if cerr != nil {
		_ = cmd.Process.Kill() // our own just-started child, by handle
		errFile.Close()
		return nil, fmt.Errorf("cannot read the creation time of pid %d: %v", pid, cerr)
	}
	r.led.Add(pid, created, "external bubo serve (started by the smoke)")
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		sc := bufio.NewScanner(out)
		sc.Buffer(make([]byte, 64*1024), 1<<20)
		var once sync.Once
		for sc.Scan() {
			ev, perr := supervisor.ParseEvent(sc.Bytes())
			if perr != nil {
				continue
			}
			e.mu.Lock()
			e.events = append(e.events, ev.Type)
			if ev.InstanceID != "" {
				e.id = ev.InstanceID
			}
			e.mu.Unlock()
			if ev.Type == "ready" {
				once.Do(func() { close(e.ready) })
			}
		}
	}()
	go func() {
		werr := cmd.Wait()
		select {
		case <-readerDone:
		case <-time.After(3 * time.Second): // a child that kept the pipe open
		}
		out.Close()
		code := 0
		if werr != nil {
			code = -1
			if ee, ok := werr.(*exec.ExitError); ok {
				code = ee.ExitCode()
			}
		}
		e.mu.Lock()
		e.exitCode = code
		e.mu.Unlock()
		errFile.Close()
		close(e.done)
	}()
	return e, nil
}

func (r *Runner) stepReconnect(ctx context.Context, s *Step) {
	ext, err := r.startExternal()
	if !s.Check("external started", err == nil, "%v", err) {
		return
	}
	s.Check("external pid recorded", true, "pid %d created %s", ext.pid, ext.creation)
	select {
	case <-ext.ready:
	case <-ext.done:
	case <-time.After(r.cfg.LoadTimeout + time.Minute):
	case <-ctx.Done():
	}
	if !s.Check("external ready event", ext.seen("ready"), "the plain `bubo serve --jsonl` reported ready (exited=%t)", isClosed(ext.done)) {
		return
	}
	sup, err := r.newSup("attach", false)
	if !s.Check("attach supervisor", err == nil, "%v", err) {
		return
	}
	v, err := r.waitView(ctx, sup, time.Minute, func(v supervisor.View) (bool, error) {
		return v.Initialized && v.State == supervisor.AttachedExternal && v.Instance != nil, nil
	})
	if !s.Check("attached as external", err == nil, "state %s, label %q, err %v", v.State, v.Reason, err) {
		return
	}
	r.ramAfterReady(s, "external_host_ram_available_gib_after_ready")
	// N16: one status call for the external instance; the raw payload is saved under --out so the step 9 walk scans it.
	if st, _, serr := r.statusOf(ctx, v.Instance.ID); s.Check("external status sample", serr == nil, "%v", serr) {
		s.Check("external status identity", st.Identity != nil && st.Identity.ID == v.Instance.ID, "instance %s", v.Instance.ID)
	}
	s.Check("attached identity", v.Instance.PID == ext.pid && v.Instance.CreationTime == ext.creation, "attached pid %d (external pid %d)", v.Instance.PID, ext.pid)
	s.Check("read-only first", !v.Managed && v.ManagementSupported, "managed=%t, management supported=%t", v.Managed, v.ManagementSupported)
	t0 := time.Now()
	stopErr := reply(ctx, sup.Stop(), 30*time.Second)
	refused, why := manageRefusal(stopErr, time.Since(t0))
	s.Check("stop refused before Manage", refused, "%s", why)
	s.Check("still not managed after the refusal", !r.view(sup).Managed, "snapshot Managed=%t", r.view(sup).Managed)
	s.Check("external survived refusal", isLive(ext.pid, ext.creation), "pid %d still alive", ext.pid)
	s.Check("manage accepted", reply(ctx, sup.Manage(), 30*time.Second) == nil, "Manage returned without an error")
	s.Check("managed now", r.view(sup).Managed, "snapshot Managed")
	stopErr = reply(ctx, sup.Stop(), r.cfg.StopTimeout+45*time.Second)
	s.Check("managed stop through the control pipe", stopErr == nil, "Stop returned: %v", stopErr)
	select {
	case <-ext.done:
	case <-time.After(5 * time.Second):
	case <-ctx.Done():
	}
	exited := isClosed(ext.done)
	s.Check("external exited", exited, "pid %d", ext.pid)
	ext.mu.Lock()
	code, events := ext.exitCode, append([]string(nil), ext.events...)
	ext.mu.Unlock()
	s.Check("clean exit code", exited && code == 0, "exit code %d; lifecycle events %v", code, events)
	s.Check("stopped event seen", ext.seen("stopped"), "the external reported stopped")
	s.Check("supervisor state", r.view(sup).State == supervisor.Stopped, "state %s", r.view(sup).State)
	r.closeSup("after attach")
	s.Check("no orphan", !isLive(ext.pid, ext.creation), "pid %d not alive", ext.pid)
	s.Summary("external pid %d attached read-only, Manage + Stop through the control pipe, exit code %d, no orphan", ext.pid, code)
}

func isClosed(c chan struct{}) bool {
	select {
	case <-c:
		return true
	default:
		return false
	}
}

// ---- step 9: log export with a planted key ----

var bundleAllowList = map[string]bool{"versions.json": true, "settings.json": true, "status.json": true, "diagnose.json": true, "history.json": true, "logs.json": true, "manifest.json": true}

// plantedKey returns the fake key of this run (one value for step 7 and step 9).
func (r *Runner) plantedKey() string {
	if r.planted == "" {
		var raw [12]byte
		_, _ = rand.Read(raw[:])
		r.planted = "sk-SMOKE-PLANTED-" + hex.EncodeToString(raw[:])
		r.rec.AddSecret(r.planted)
	}
	return r.planted
}

func (r *Runner) stepLogExport(ctx context.Context, s *Step) {
	planted := r.plantedKey()
	s.Check("exe", fileExists(r.cfg.Exe), "%s", r.cfg.Exe)
	for _, name := range []string{"BUBO_API_KEY", "GUFO_EVAL_API_KEY", "OPENAI_API_KEY"} {
		old, had := os.LookupEnv(name)
		os.Setenv(name, planted)
		defer func(n, o string, h bool) {
			if h {
				os.Setenv(n, o)
			} else {
				os.Unsetenv(n)
			}
		}(name, old, had)
	}
	// Route 1: the supervisor helper, as the app calls it. The supervisor drops
	// BUBO_API_KEY from a child it starts, so this route carries the other two names.
	sup, err := r.newSup("diag", false)
	if !s.Check("diagnose supervisor", err == nil, "%v", err) {
		return
	}
	defer r.closeSup("after log export")
	zip1 := filepath.Join(r.cfg.Out, "support-bundle-supervisor.zip")
	var rep supervisor.DiagnosticReply
	select {
	case rep = <-sup.Diagnose("--bundle", zip1):
	case <-time.After(3 * time.Minute):
		s.Check("supervisor route reply", false, "no reply in 3 minutes")
		return
	case <-ctx.Done():
		s.Check("supervisor route reply", false, "%v", ctx.Err())
		return
	}
	if s.Check("supervisor route: diagnose --bundle", rep.Err == nil, "%v", rep.Err) {
		sb, ok := rep.Event.Value.(*supervisor.SupportBundle)
		s.Check("supervisor route: support_bundle event", rep.Event.Type == "support_bundle" && ok && sb.Saved, "event %s saved=%t", rep.Event.Type, ok && sb.Saved)
		r.checkBundle(s, "supervisor route", zip1, planted)
	}
	// Route 2: the CLI called directly with the full environment (all three names).
	zip2 := filepath.Join(r.cfg.Out, "support-bundle-direct.zip")
	ev, err := r.cli(ctx, 3*time.Minute, "diagnose", "--bundle", zip2, "--json")
	if s.Check("direct route: diagnose --bundle", err == nil, "%v", err) {
		sb, ok := ev.Value.(*supervisor.SupportBundle)
		s.Check("direct route: support_bundle event", ev.Type == "support_bundle" && ok && sb.Saved, "event %s saved=%t", ev.Type, ok && sb.Saved)
		r.checkBundle(s, "direct route", zip2, planted)
	}
	// The same key must not appear in anything else this run wrote.
	files, other := 0, 0
	_ = filepath.Walk(r.cfg.Out, func(p string, fi os.FileInfo, werr error) error {
		if werr != nil || fi.IsDir() || strings.HasSuffix(p, ".zip") {
			return nil
		}
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			return nil
		}
		files++
		other += bytes.Count(b, []byte(planted))
		return nil
	})
	s.Check("planted key absent from run output", other == 0, "%d hit(s) in %d files under --out", other, files)
	s.Summary("2 bundles (supervisor route and direct route) hold only allow-list members and 0 hits for the planted key; 0 hits in %d output files", files)
}

// checkBundle opens one saved ZIP, checks the member list and sizes, and greps
// every member for the planted key.
func (r *Runner) checkBundle(s *Step, label, zipPath, planted string) {
	info, err := os.Stat(zipPath)
	if !s.Check(label+": zip written", err == nil, "%v", err) {
		return
	}
	s.Check(label+": zip size", info.Size() <= 163840, "%d bytes (limit 163840)", info.Size())
	zr, err := zip.OpenReader(zipPath)
	if !s.Check(label+": zip opens", err == nil, "%v", err) {
		return
	}
	defer zr.Close()
	extract := strings.TrimSuffix(zipPath, ".zip") + "-files"
	_ = os.MkdirAll(extract, 0o700)
	hits, total, members := 0, 0, 0
	var unexpected []string
	for _, f := range zr.File {
		members++
		if !bundleAllowList[f.Name] {
			unexpected = append(unexpected, f.Name)
			continue
		}
		rc, oerr := f.Open()
		if oerr != nil {
			s.Check(label+": member "+f.Name, false, "%v", oerr)
			continue
		}
		b, _ := io.ReadAll(io.LimitReader(rc, 1<<20))
		rc.Close()
		total += len(b)
		if len(b) > 32768 {
			s.Check(label+": member size "+f.Name, false, "%d bytes (limit 32768)", len(b))
		}
		hits += bytes.Count(b, []byte(planted)) + bytes.Count(bytes.ToLower(b), bytes.ToLower([]byte(planted)))
		// A leaked key must not be copied into the run output.
		_ = os.WriteFile(filepath.Join(extract, f.Name), bytes.ReplaceAll(b, []byte(planted), []byte("[PLANTED-KEY-FOUND]")), 0o600)
	}
	s.Check(label+": allow-list members only", len(unexpected) == 0 && members > 0, "%d member(s); unexpected %v", members, unexpected)
	s.Check(label+": planted key absent from the bundle", hits == 0, "%d hit(s) for the planted key in %d bytes of members", hits, total)
}

// ---- step 8: cleanup proof ----

func (r *Runner) stepCleanup() {
	list := r.led.Snapshot()
	if len(list) == 0 {
		r.rec.NotRun(8, "cleanup proof", "no process was started")
		return
	}
	s := r.rec.Begin(8, "cleanup proof")
	stopped, stuck := r.led.Cleanup()
	for _, p := range r.led.Snapshot() {
		alive := p.AliveAtEnd != nil && *p.AliveAtEnd
		s.Check(fmt.Sprintf("pid %d", p.PID), !alive, "%s; created %s; end: %s", p.Role, p.CreationTime, p.EndAction)
	}
	s.Check("no forced cleanup", len(stopped) == 0 && len(r.forced) == 0, "stopped by exact PID outside a normal stop: %v %v", r.forced, pidList(stopped))
	s.Check("nothing stuck", len(stuck) == 0, "%d process(es) still alive", len(stuck))
	s.Check("port free again", portFree(r.cfg.Port) == nil, "127.0.0.1:%d binds (no unrecorded listener)", r.cfg.Port)
	cctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if st, _, err := r.statusOf(cctx, ""); s.Check("discovery empty", err == nil, "%v", err) {
		s.Check("no live instance", len(st.Instances) == 0, "%d instance(s) in discovery", len(st.Instances))
	}
	s.Set("pids", r.led.Snapshot())
	s.Summary("%d process(es) recorded, none alive, none needed a forced stop, port %d free", len(list), r.cfg.Port)
	s.End()
}

func pidList(p []PIDRecord) []uint32 {
	var out []uint32
	for _, x := range p {
		out = append(out, x.PID)
	}
	return out
}
