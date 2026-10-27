//go:build windows

package main

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"bubo/gui/supervisor"
)

// coldStartLimit bounds the one cold `status --json` before step 1's
// supervisor checks (real runs only).
const coldStartLimit = 5 * time.Minute

const storyPrompt = "Write a complete short story of about 200 tokens about an owl discovering a hidden garden. Begin the story immediately."

func (r *Runner) runMVP(ctx context.Context) int {
	r.ctx, r.client = ctx, &http.Client{Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(*http.Request, []*http.Request) error { return fmt.Errorf("smoke refuses HTTP redirects") }}
	r.rec.report.Mode = "mvp-real"
	if r.cfg.Fake {
		r.rec.report.Mode = "mvp-fake"
	}
	r.rec.Logf("MVP mode=%s package=%s; no --profile or --context override; no production stop/restore", r.rec.report.Mode, r.cfg.Package)
	budget := r.cfg.QueryTimeout * 3
	if !r.cfg.Fake {
		budget += coldStartLimit
	}
	pre := r.exec(1, "prepared View", budget, r.mvpPrepared)
	if r.cfg.MVPStep1Only {
		for n, name := range map[int]string{2: "one-session Ready", 3: "story with live status", 4: "one-session normal stop", 5: "two sessions with recommended context"} {
			r.rec.NotRun(n, name, "step-1-only run (no model load)")
		}
	} else if pre.Status == Pass {
		one := r.exec(2, "one-session Ready", r.cfg.LoadTimeout, r.mvpStartOne)
		if one.Status == Pass {
			r.exec(3, "story with live status", r.cfg.RequestTimeout+r.cfg.QueryTimeout, r.mvpStory)
		} else {
			r.rec.NotRun(3, "story with live status", "one-session start failed")
		}
		if r.sup != nil && r.view(r.sup).Instance != nil {
			r.exec(4, "one-session normal stop", r.cfg.StopTimeout+r.cfg.QueryTimeout+time.Minute, r.mvpStop)
		} else {
			r.rec.NotRun(4, "one-session normal stop", "no owned server")
		}
		r.closeSup("before two-session run")
		if len(r.led.Alive()) == 0 && portFree(r.cfg.Port) == nil {
			r.exec(5, "two sessions with recommended context", r.cfg.LoadTimeout+r.cfg.RequestTimeout+r.cfg.StopTimeout+r.cfg.QueryTimeout*3+time.Minute, r.mvpTwo)
		} else {
			r.rec.NotRun(5, "two sessions with recommended context", "first server not fully stopped")
		}
	} else {
		for n, name := range map[int]string{2: "one-session Ready", 3: "story with live status", 4: "one-session normal stop", 5: "two sessions with recommended context"} {
			r.rec.NotRun(n, name, "prepared View failed")
		}
	}
	r.closeSup("MVP final")
	// Interruption never skips cleanup/evidence. Keep the original cancellation
	// for the final verdict, but give owned-process cleanup its own deadline.
	r.ctx = context.WithoutCancel(ctx)
	r.exec(6, "saved Views, error scan and cleanup", time.Minute, r.mvpEvidence)
	r.ctx = ctx
	pass, fail, notrun, checks := 0, 0, 0, 0
	for _, s := range r.rec.report.Steps {
		checks += len(s.Checks)
		switch s.Status {
		case Pass:
			pass++
		case Fail:
			fail++
		default:
			notrun++
		}
	}
	code, overall := ExitPass, "REAL PASS"
	if r.cfg.Fake {
		overall = "DRY RUN PASS"
	}
	if r.cfg.MVPStep1Only {
		overall = "STEP 1 PASS"
	}
	if fail > 0 {
		code, overall = ExitFail, "FAIL"
	} else if r.cfg.MVPStep1Only {
		if pass != 2 {
			code, overall = ExitNotRun, "NOT RUN"
		}
	} else if pass != 6 {
		code, overall = ExitNotRun, "NOT RUN"
	}
	if ctx.Err() != nil {
		code, overall = ExitInterrupted, "INTERRUPTED"
		r.rec.report.Interrupt = true
	}
	r.rec.report.Overall, r.rec.report.ExitCode, r.rec.report.FinishedAt = overall, code, stamp()
	r.rec.Logf("OVERALL: %s (%d pass, %d fail, %d not run; %d checks); exit %d", overall, pass, fail, notrun, checks, code)
	return code
}

func (r *Runner) mvpPrepared(ctx context.Context, s *Step) {
	if !s.Check("CLI file", fileExists(r.cfg.Exe), "%s", r.cfg.Exe) || !s.Check("manifest file", fileExists(r.cfg.Manifest), "%s", r.cfg.Manifest) {
		return
	}
	h, err := hashFile(r.cfg.Exe)
	s.Check("CLI hash", err == nil, "sha256=%s error=%v", h, err)
	s.Set("cli_sha256", h)
	if self, e := os.Executable(); e == nil {
		if h, e = hashFile(self); e == nil {
			s.Set("driver_sha256", h)
		}
	}
	procs, err := listProcs(r.cfg.ProcPatterns)
	procs, fixtures := splitTestFixtures(procs)
	s.Set("ignored_test_fixture_processes", fixtures)
	if !s.Check("production already stopped by user", err == nil && len(procs) == 0, "%s error=%v; ignored %d test fake(s) under the temp folder (read-only; nothing stopped)", procText(procs), err, len(fixtures)) {
		return
	}
	if !r.cfg.Fake {




		wctx, cancel := context.WithTimeout(ctx, coldStartLimit)
		start := time.Now()
		err := hiddenCommand(wctx, r.cfg.Exe, "status", "--json").Run()
		cancel()
		took := time.Since(start)
		s.Set("cli_cold_status_seconds", took.Seconds())
		if !s.Check("CLI cold-start status", err == nil, "status --json took %.2f s (limit %s) error=%v", took.Seconds(), coldStartLimit, err) {
			return
		}
	}
	sup, err := r.newSup("mvp-one", true)
	if !s.Check("supervisor created", err == nil, "%v", err) {
		return
	}
	v, err := r.waitView(ctx, sup, r.cfg.QueryTimeout*2, func(v supervisor.View) (bool, error) { return v.Initialized && v.Product != nil, nil })
	if !s.Check("bootstrap View initialized", err == nil, "%v", err) {
		return
	}
	s.Check("prepared from CLI", v.Prepared && v.Runtime.Readiness.Runtime && v.Runtime.Readiness.Model, "prepared=%t readiness=%+v", v.Prepared, v.Runtime.Readiness)
	s.Check("no discovered instance", v.Instance == nil && len(v.Discovered) == 0, "%d discovered", len(v.Discovered))
	// A runtime folder with its AMD files beside the exe is "legacy"; the CLI
	// reports it only while the shared folder is not installed. The shared
	// layout itself is K-b's check.
	shared := v.Runtime.Paths["runtime_shared"].Text
	_, sharedErr := os.Stat(shared)
	kindOK := v.Runtime.Kind == "shared" || (v.Runtime.Kind == "legacy" && shared != "" && os.IsNotExist(sharedErr))
	s.Set("runtime_kind", v.Runtime.Kind)
	s.Check("runtime kind and path from CLI", kindOK && v.Runtime.Paths["runtime"].Kind == "string" && v.Runtime.Paths["runtime"].Text != "", "kind=%s path=%s shared=%s shared_present=%t", v.Runtime.Kind, v.Runtime.Paths["runtime"].Text, shared, sharedErr == nil)
	// Before any server runs the View has no memory estimate (real status:
	// null with a reason); the GUI setup page asks `models estimate`.
	s.Set("view_memory_estimate_reason", v.EstimateReason)
	est, subject, err := r.prestartEstimate(v, 0)
	s.Set("prestart_estimate_subject", subject)
	s.Set("memory_estimate", est)
	if !s.Check("CLI memory line fields (models estimate)", err == nil && estimatePresent(est), "subject=%s error=%v; GPU/host budgets, fit, recommendation table", subject, err) {
		return
	}
	reason := ""
	if est.FitReason != nil {
		reason = *est.FitReason
	}
	s.Check("default setup fits (CLI admission)", *est.Fits, "fits=%t reason=%q (a start would be refused; free host RAM or GPU memory first)", *est.Fits, reason)
	s.Summary("prepared with the CLI runtime; memory comes from the CLI estimate; no GUI")
}

func memoryPresent(v supervisor.View) bool {
	m := v.MemoryEstimate
	return m != nil && m.GPU != nil && m.GPU.Available != nil && m.Host != nil && m.Host.Available != nil && m.Fits != nil && m.RecommendedContext != nil && len(m.RecommendedContext.BySessions) > 0
}

func estimatePresent(m *supervisor.MemoryEstimate) bool {
	return m != nil && m.GPU != nil && m.GPU.Available != nil && m.Host != nil && m.Host.Available != nil && m.Fits != nil && m.RecommendedContext != nil && len(m.RecommendedContext.BySessions) > 0
}

// prestartEstimate asks the CLI for the prepared model's memory estimate the
// way the GUI setup page does (`models estimate` through ModelService). With
// no server running, the real `status` has memory_estimate null plus a reason
// (04:29 real run), so the View cannot carry it before the first start.
func (r *Runner) prestartEstimate(v supervisor.View, sessions uint32) (*supervisor.MemoryEstimate, string, error) {
	subject := ""
	for _, m := range v.PreparedModels {
		if m.ManifestPath != nil && normPath(*m.ManifestPath) == normPath(r.cfg.Manifest) {
			subject = m.ID
		}
	}
	if subject == "" {
		return nil, "", fmt.Errorf("no prepared model has manifest %s", r.cfg.Manifest)
	}
	svc := supervisor.NewModelService(supervisor.ModelServiceConfig{Exe: func() string { return r.cfg.Exe }, CallTimeout: r.cfg.QueryTimeout})
	defer svc.Close()
	est, err := svc.Estimate(subject, 0, sessions)
	if err != nil {
		return nil, subject, err
	}
	if est.MemoryEstimate == nil {
		reason := ""
		if est.Reason != nil {
			reason = *est.Reason
		}
		return nil, subject, fmt.Errorf("estimate_state=%s reason=%s", est.State, reason)
	}
	return est.MemoryEstimate, subject, nil
}

func recommendation(v supervisor.View, sessions int) (supervisor.RecommendedContextRow, bool) {
	if v.MemoryEstimate != nil {
		return recommendationIn(v.MemoryEstimate.RecommendedContext, sessions)
	}
	return supervisor.RecommendedContextRow{}, false
}

func recommendationIn(table *supervisor.RecommendedContext, sessions int) (supervisor.RecommendedContextRow, bool) {
	if table != nil {
		for _, row := range table.BySessions {
			if uint64(row.Sessions) == uint64(sessions) {
				return row, row.Fits && row.ContextPerSession > 0 && (row.Recommended == nil || *row.Recommended)
			}
		}
	}
	return supervisor.RecommendedContextRow{}, false
}

func setting(v supervisor.View, key string) (supervisor.ViewSetting, bool) {
	for _, row := range v.Settings {
		if row.Name == key || row.Name == "--"+key {
			return row.ViewSetting, true
		}
	}
	return supervisor.ViewSetting{}, false
}

func (r *Runner) mvpStart(ctx context.Context, s *Step, sessions int) bool {
	args := []string{"--manifest", r.cfg.Manifest, "--sessions", strconv.Itoa(sessions)}
	s.Set("serve_extra_args", args)
	t0 := time.Now()
	if !s.Check("Start accepted", reply(ctx, r.sup.Start(args...), r.cfg.QueryTimeout) == nil, "sessions=%d; profile/context omitted", sessions) {
		return false
	}
	v, err := r.waitView(ctx, r.sup, r.cfg.LoadTimeout, func(v supervisor.View) (bool, error) {
		if v.State == supervisor.Failed {
			return false, fmt.Errorf("%s: %s", lastErr(v), v.Remedy)
		}
		return v.State == supervisor.Ready && v.Instance != nil && v.StatusFeed.State == supervisor.StatusUpdating, nil
	})
	s.Set("ready_seconds", time.Since(t0).Seconds())
	if !s.Check("verified Ready within deadline", err == nil && time.Since(t0) <= r.cfg.LoadTimeout, "elapsed=%s limit=%s error=%v", time.Since(t0).Round(time.Millisecond), r.cfg.LoadTimeout, err) {
		return false
	}
	r.led.Add(v.Instance.PID, v.Instance.CreationTime, "MVP supervisor-owned serve")
	s.Check("server identity alive", isLive(v.Instance.PID, v.Instance.CreationTime), "pid=%d creation=%s", v.Instance.PID, v.Instance.CreationTime)
	s.Check("clean lifecycle", v.Failure == nil, "%s", lastErr(v))
	epOK := v.Endpoints != nil && v.Endpoints.OpenAIBaseURL != "" && v.Endpoints.AnthropicBaseURL != ""
	if !s.Check("CLI OpenAI and Anthropic endpoints", epOK, "%+v", v.Endpoints) {
		return false
	}
	for name, ep := range map[string]string{"OpenAI": v.Endpoints.OpenAIBaseURL, "Anthropic": v.Endpoints.AnthropicBaseURL} {
		if !s.Check(name+" endpoint on smoke loopback port", smokeEndpoint(ep, r.cfg.Port), "%s", ep) {
			return false
		}
	}
	r.endpoint, r.model, r.instanceID = v.Endpoints.OpenAIBaseURL, v.Instance.Model, v.Instance.ID
	s.Check("CLI served model", r.model != "", "%s", r.model)
	s.Check("memory fits and recommendation present", memoryPresent(v) && *v.MemoryEstimate.Fits, "CLI memory fields (no fit recomputation)")
	if v.MemoryEstimate != nil {
		s.Set("memory_estimate", v.MemoryEstimate)
		s.Check("CLI recommends selected allocation", v.MemoryEstimate.Recommended != nil && *v.MemoryEstimate.Recommended, "recommended=%v reason=%v", v.MemoryEstimate.Recommended, v.MemoryEstimate.RecommendationReason)
	}
	if r.cfg.Fake {
		s.Set("gpu_check", "not exercised: fake mode does not construct a GPU probe")
	} else {
		v, err = r.waitView(ctx, r.sup, r.cfg.QueryTimeout, func(v supervisor.View) (bool, error) {
			return v.GPU != nil && v.GPU.TemperatureC != nil && v.GPU.ChipPowerW != nil, nil
		})
		s.Check("GPU temperature and chip power", err == nil, "GPU=%+v error=%v", v.GPU, err)
	}
	// Absence in saved preferences and the actual argument list proves the
	// default profile request. The name itself remains a CLI-owned fact.
	s.Check("no saved profile/context override", v.Saved.Profile == nil && v.Saved.Context == nil, "CLI resolves both")
	profile, ok := setting(v, "profile")
	s.Check("CLI effective profile", ok && profile.Effective.Kind == "string" && profile.Effective.Text != "", "%+v", profile)
	return true
}

// Validate only transport isolation, never derive or append a CLI base URL.
func smokeEndpoint(ep string, port int) bool {
	u, err := url.Parse(ep)
	if err != nil || u.Scheme != "http" || u.User != nil || u.Fragment != "" || u.RawQuery != "" {
		return false
	}
	h := u.Hostname()
	return (h == "127.0.0.1" || h == "localhost" || h == "::1") && u.Port() == strconv.Itoa(port)
}

func (r *Runner) mvpStartOne(ctx context.Context, s *Step) {
	if r.mvpStart(ctx, s, 1) {
		s.Summary("one session; CLI defaults; shared runtime; CLI endpoints and memory")
	}
}
func (r *Runner) mvpStory(ctx context.Context, s *Step) { r.mvpBurst(ctx, s, 1) }

type burstReply struct {
	index  int
	result chatResult
	err    error
}

// Gaps include request-start -> first arrival and final arrival -> last HTTP
// completion. Never average gaps or count repeated GetView calls as records.
type cadence struct {
	last    time.Time
	gaps    []float64
	records int
	bad     bool
}

func (c *cadence) observe(received, now time.Time) {
	if received.After(c.last) {
		gap := received.Sub(c.last)
		c.gaps = append(c.gaps, gap.Seconds())
		c.bad = c.bad || gap > 2*time.Second
		c.last = received
		c.records++
	}
	if now.Sub(c.last) > 2*time.Second {
		c.bad = true
	}
}

func hasLiveRates(v supervisor.View) bool {
	return v.Rates != nil && v.Rates.Source == "cli" && v.Rates.DecodeTPS != nil && v.Rates.PrefillTPS != nil
}
func busySlots(v supervisor.View) int {
	n := 0
	for _, slot := range v.Slots {
		if slot.Processing {
			n++
		}
	}
	return n
}

func (r *Runner) mvpBurst(ctx context.Context, s *Step, n int) {
	before := r.view(r.sup)
	seenIDs := map[string]bool{}
	for _, row := range before.Requests {
		seenIDs[row.ID] = true
	}
	t0 := time.Now()
	cad := cadence{last: t0}
	lastReceipt := r.receipt()
	requestCtx, cancel := context.WithTimeout(ctx, r.cfg.RequestTimeout)
	defer cancel()
	done := make(chan burstReply, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			res, err := chat(requestCtx, r.client, r.endpoint, r.model, []chatMessage{{"user", fmt.Sprintf("%s Story number %d.", storyPrompt, i+1)}}, 240)
			done <- burstReply{i, res, err}
		}(i)
	}
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	completed, maxBusy, live := 0, 0, false
	for completed < n {
		select {
		case res := <-done:
			completed++
			r.checkChat(s, fmt.Sprintf("story %d", res.index+1), res.result, res.err, 240)
			s.Set(fmt.Sprintf("story_%d", res.index+1), chatSummary(res.result))
		case <-tick.C:
			v := r.view(r.sup)
			now := time.Now()
			if r.receipt().After(lastReceipt) && v.StatusFeed.State == supervisor.StatusUpdating {
				lastReceipt = r.receipt()
				cad.observe(lastReceipt, now)
			} else {
				cad.observe(cad.last, now)
			}
			if v.Sessions != nil && *v.Sessions >= uint64(busySlots(v)) {
				maxBusy = max(maxBusy, busySlots(v))
			}
			live = live || (busySlots(v) > 0 && hasLiveRates(v))
			if v.State != supervisor.Ready || v.StatusFeed.Error != nil || v.Stale || v.StatusFeed.State != supervisor.StatusUpdating {
				cad.bad = true
			}
		case <-requestCtx.Done():
			s.Check("request deadline", false, "%v", requestCtx.Err())
			completed = n
		}
	}
	cad.observe(cad.last, time.Now())
	cad.gaps = append(cad.gaps, time.Since(cad.last).Seconds())
	s.Set("status_gaps_seconds", cad.gaps)
	s.Set("status_records_during_burst", cad.records)
	s.Check("status cadence every gap <=2s", !cad.bad && cad.records >= 2, "%d record arrivals; gaps=%v", cad.records, cad.gaps)
	s.Check("View shows busy sessions", maxBusy >= n, "max concurrent processing slots=%d expected=%d", maxBusy, n)


	if r.cfg.Fake {
		s.Check("live rate structure (fake only)", r.view(r.sup).Rates != nil, "nullable rates allowed; non-null assertion NOT RUN")
	} else {
		s.Check("live CLI prefill/decode rates during request", live, "both non-null in a busy View, source=cli")
	}
	v, err := r.waitView(ctx, r.sup, r.cfg.QueryTimeout, func(v supervisor.View) (bool, error) {
		fresh := 0
		for _, row := range v.Requests {
			if !seenIDs[row.ID] && row.Outcome == "completed" && row.CompletionTokens != nil && *row.CompletionTokens > 0 {
				fresh++
			}
		}
		return fresh >= n, nil
	})
	s.Check("completed requests in View log", err == nil, "%d rows, error=%v", len(v.Requests), err)
	s.Summary("%d requests while watch runs; %d arrivals; max busy=%d", n, cad.records, maxBusy)
}

func (r *Runner) mvpStop(ctx context.Context, s *Step) {
	_, ok := r.stopOwned(ctx, s, "normal stop")
	s.Check("smoke port free", portFree(r.cfg.Port) == nil, "127.0.0.1:%d", r.cfg.Port)
	s.Check("recorded server identities gone", ok && len(r.led.Alive()) == 0, "%d alive", len(r.led.Alive()))
	s.Summary("normal supervisor Stop; server PIDs gone; smoke port free")
}

func (r *Runner) mvpTwo(ctx context.Context, s *Step) {
	sup, err := r.newSup("mvp-two", true)
	if !s.Check("second supervisor", err == nil, "%v", err) {
		return
	}
	v, err := r.waitView(ctx, sup, r.cfg.QueryTimeout*2, func(v supervisor.View) (bool, error) {
		return v.Initialized && v.Product != nil && v.Instance == nil, nil
	})
	if !s.Check("fresh stopped View", err == nil, "%v", err) {
		return
	}
	// Fresh pre-start recommendation from the CLI estimate, as the setup page
	// gets it (the View has no estimate before a server runs).
	est, subject, err := r.prestartEstimate(v, 2)
	var row supervisor.RecommendedContextRow
	ok := false
	if err == nil && est != nil {
		row, ok = recommendationIn(est.RecommendedContext, 2)
	}
	if !s.Check("CLI two-session recommendation", ok, "subject=%s error=%v row=%+v", subject, err, row) {
		return
	}
	s.Set("prestart_cli_recommendation", row)
	if !r.mvpStart(ctx, s, 2) {
		return
	}
	v = r.view(sup)
	contextSetting, found := setting(v, "context")
	want := strconv.FormatUint(uint64(row.ContextPerSession), 10)
	s.Check("CLI resolved context without override", found && contextSetting.Effective.Text == want && contextSetting.Source == "recommended_context", "expected=%s effective=%+v", want, contextSetting)
	if v.MemoryEstimate != nil {
		s.Check("CLI memory uses two-session context", v.MemoryEstimate.Sessions == 2 && uint64(v.MemoryEstimate.ContextTokens) == uint64(row.ContextPerSession), "sessions=%d context=%d", v.MemoryEstimate.Sessions, v.MemoryEstimate.ContextTokens)
	}
	r.mvpBurst(ctx, s, 2)
	r.mvpStop(ctx, s)
	s.Summary("two-session recommendation %s; concurrent stories; normal stop", want)
}

var smokeErrors = regexp.MustCompile(`(?i)\berror\b|\bfatal\b|\bpanic\b|state capacity|protocol_mismatch`)

func (r *Runner) mvpEvidence(_ context.Context, s *Step) {
	s.Check("supervisor exits completed", len(r.shutdownFailures) == 0, "%v", r.shutdownFailures)
	stopped, stuck := r.led.Cleanup()
	s.Check("no forced cleanup", len(stopped) == 0 && len(r.forced) == 0, "forced PIDs=%v", pidList(stopped))
	s.Check("exact server PIDs gone", len(stuck) == 0, "%d stuck of %d recorded", len(stuck), len(r.led.Snapshot()))
	s.Check("port released", portFree(r.cfg.Port) == nil, "port %d", r.cfg.Port)
	if r.views.file != nil {
		if err := r.views.file.Close(); r.views.err == nil {
			r.views.err = err
		}
	}
	s.Check("every constructed View saved", r.views.err == nil && r.views.count > 0, "%d JSON Views; error=%v", r.views.count, r.views.err)
	s.Set("view_count", r.views.count)
	var hits []string
	files := 0
	err := filepath.WalkDir(r.cfg.Out, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".log") {
			return nil
		}
		files++
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 4096), 4<<20)
		for line := 1; scanner.Scan(); line++ {
			if smokeErrors.MatchString(scanner.Text()) {
				hits = append(hits, fmt.Sprintf("%s:%d: %s", path, line, scanner.Text()))
			}
		}
		return scanner.Err()
	})
	s.Set("log_error_hits", hits)
	s.Check("all supervisor disk logs scanned for errors", err == nil && files > 0 && len(hits) == 0, "%d files, %d hits, error=%v", files, len(hits), err)
	s.Summary("%d saved Views; %d logs; %d error hits; all exact server identities gone", r.views.count, files, len(hits))
}
