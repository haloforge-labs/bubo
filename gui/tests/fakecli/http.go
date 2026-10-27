//go:build windows

package main

// J12 smoke support. BUBO_FAKE_HTTP turns on a loopback chat endpoint inside
// the fake serve process, and the matching status objects. Values:
//
//	ok       normal: 200 replies, prompt-prefix cache reuse, idle slots
//	fail     every chat request returns HTTP 500
//	nocache  replies work but never report cached tokens
//	stuck    replies work but the slot never returns to idle in status
//	refuse   every chat request is refused by the memory guard (503
//	         insufficient_memory; outcome refused with the refusal object)
//	badmodel every chat request fails with HTTP 404 model_not_found
//
// BUBO_FAKE_POWER=battery makes the machine object report battery power.
// BUBO_FAKE_BUNDLE_LEAK=1 makes diagnose --bundle copy BUBO_API_KEY into the ZIP
// (a misbehaving CLI, to prove the smoke grep catches it).
// BUBO_FAKE_POWER=unknown reports an unreadable power source.
// BUBO_FAKE_NULL_COUNTERS=a,b reports those counters as null.
// BUBO_FAKE_ENV_LEAK=1 makes serve echo GUFO_EVAL_API_KEY to stderr (a server
// that dumps its environment, to prove the step 9 walk finds a planted key).
// BUBO_FAKE_LOAD_MS delays the ready event.
// The serve process owns http-state.json in BUBO_FAKE_DIR; status reads it.

import (
	"archive/zip"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/windows"
)

var httpMode = os.Getenv("BUBO_FAKE_HTTP")

type httpState struct {
	Counters map[string]any `json:"counters"`
	Requests []any          `json:"requests"`
	Stuck    bool           `json:"stuck"`
	Active   map[int]bool   `json:"active,omitempty"`
}

var (
	hmu        sync.Mutex
	hstate     = httpState{Counters: baseCounters(), Requests: []any{}}
	lastPrompt []string
	reqSeq     int
)

func baseCounters() map[string]any {
	return map[string]any{"prefill_tokens_total": 0, "decode_tokens_total": 0, "cached_tokens_total": 0, "cache_disk_hits_total": 0, "cache_memory_hits_total": 0, "cache_misses_total": 0, "max_sequence_tokens": 0, "draft_rounds_total": 0, "draft_tokens_total": 0, "draft_accepted_tokens_total": 0, "device_lost_total": 0, "prefill_seconds_total": 0.0, "decode_seconds_total": 0.0, "completed_successful_requests_total": 0}
}
func bump(key string, by int) { hstate.Counters[key] = hstate.Counters[key].(int) + by }

func saveHTTPState() {
	b, _ := json.Marshal(hstate)
	tmp := filepath.Join(dir, "http-state.tmp")
	os.WriteFile(tmp, b, 0600)
	from, _ := windows.UTF16PtrFromString(tmp)
	to, _ := windows.UTF16PtrFromString(filepath.Join(dir, "http-state.json"))
	// A status process can hold the target open for a moment: retry the replace.
	for n := 0; n < 400; n++ {
		if windows.MoveFileEx(from, to, windows.MOVEFILE_REPLACE_EXISTING) == nil {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// startFakeHTTP binds the loopback endpoint or exits with port_occupied/13.
func startFakeHTTP(host, port string) {
	// The GUI exercises LAN settings and key handoff, but this test server
	// must never expose a socket on the owner's network.
	l, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", port))
	if err != nil {
		cliError("port_occupied", "serve", "bind", false, "Choose a free port.", 13)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", chat)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	hmu.Lock()
	saveHTTPState()
	hmu.Unlock()
	go http.Serve(l, mux)
}

type fakeChatBody struct {
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	MaxTokens int `json:"max_tokens"`
}

func chat(w http.ResponseWriter, r *http.Request) {
	var body fakeChatBody
	if httpMode == "fail" {
		http.Error(w, `{"error":{"message":"fake failure"}}`, 500)
		return
	}
	if httpMode == "refuse" || httpMode == "badmodel" {
		refuseOrFail(w)
		return
	}
	if json.NewDecoder(r.Body).Decode(&body) != nil || len(body.Messages) == 0 {
		http.Error(w, `{"error":{"message":"bad request"}}`, 400)
		return
	}
	if httpMode == "mvp" {
		mvpChat(w, r, body)
		return
	}
	var prompt []string
	for _, m := range body.Messages {
		prompt = append(prompt, strings.Fields(m.Role+" "+m.Content)...)
	}
	hmu.Lock()
	defer hmu.Unlock()
	cached := 0
	for cached < len(prompt) && cached < len(lastPrompt) && prompt[cached] == lastPrompt[cached] {
		cached++
	}
	if httpMode == "nocache" {
		cached = 0
	}
	lastPrompt = prompt
	out := 5
	if body.MaxTokens > 0 && body.MaxTokens < out {
		out = body.MaxTokens
	}
	reqSeq++
	kind := "miss"
	if cached > 0 {
		kind = "memory"
		bump("cache_memory_hits_total", 1)
	} else {
		bump("cache_misses_total", 1)
	}
	bump("prefill_tokens_total", len(prompt)-cached)
	bump("decode_tokens_total", out)
	bump("cached_tokens_total", cached)
	bump("completed_successful_requests_total", 1)
	if len(prompt) > hstate.Counters["max_sequence_tokens"].(int) {
		hstate.Counters["max_sequence_tokens"] = len(prompt)
	}
	row := map[string]any{"id": "r" + strconv.Itoa(reqSeq), "time": now(), "slot": 0, "prompt_tokens": len(prompt), "completion_tokens": out, "cached_tokens": cached, "ttft_ms": 12.5, "prefill_tps": 2000.0, "decode_tps": 50.0, "cache_kind": kind, "finish_reason": "stop", "outcome": "completed", "status": 200, "error_code": nil}
	hstate.Requests = append(hstate.Requests, row)
	if len(hstate.Requests) > 50 {
		hstate.Requests = hstate.Requests[1:]
	}
	hstate.Stuck = httpMode == "stuck"
	saveHTTPState()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Server-Timing", "ttft;dur=12.5, inter_token;dur=20, max_inter_token;dur=30")
	resp := map[string]any{"id": "chatcmpl-fake", "object": "chat.completion", "choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "ready"}}},
		"usage":   map[string]any{"prompt_tokens": len(prompt), "completion_tokens": out, "total_tokens": len(prompt) + out, "prompt_tokens_details": map[string]any{"cached_tokens": cached}},
		"timings": map[string]any{"prompt_n": len(prompt) - cached, "prompt_per_second": 2000.0, "predicted_n": out, "predicted_per_second": 50.0, "cache_n": cached}}
	json.NewEncoder(w).Encode(resp)
}

// refuseOrFail answers a chat request with the memory-guard refusal or a
// failed reply, and records the matching request row (CLI-CONTRACT requests).
func refuseOrFail(w http.ResponseWriter) {
	hmu.Lock()
	defer hmu.Unlock()
	reqSeq++
	row := map[string]any{"id": "r" + strconv.Itoa(reqSeq), "time": now(), "slot": nil, "prompt_tokens": 5, "completion_tokens": nil, "cached_tokens": nil, "ttft_ms": nil, "prefill_tps": nil, "decode_tps": nil, "cache_kind": nil, "finish_reason": nil}
	status, code := 404, "model_not_found"
	row["outcome"], row["status"], row["error_code"] = "failed", status, code
	if httpMode == "refuse" {
		status, code = 503, "insufficient_memory"
		row["outcome"], row["status"], row["error_code"] = "refused", status, code
		row["refusal"] = map[string]any{"code": code, "needed_bytes": 8589934592, "free_bytes": 2147483648}
	}
	hstate.Requests = append(hstate.Requests, row)
	if len(hstate.Requests) > 50 {
		hstate.Requests = hstate.Requests[1:]
	}
	saveHTTPState()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": code, "message": "fake " + code}})
}

func readHTTPState() httpState {
	s := httpState{Counters: baseCounters(), Requests: []any{}}
	path := filepath.Join(dir, "http-state.json")
	for n := 0; n < 400; n++ {
		b, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				return s
			}
			time.Sleep(5 * time.Millisecond) // the writer is replacing the file
			continue
		}
		var t httpState
		if json.Unmarshal(b, &t) == nil && t.Counters != nil {
			s = t
		}
		return s
	}
	return s
}

// httpAlways adds the objects a status carries with or without an instance.
func httpAlways(p map[string]any) {
	caps := p["capabilities"].(map[string]any)
	caps["machine_readings"] = true
	var manifest any
	if m := os.Getenv("BUBO_MANIFEST"); m != "" {
		manifest = m
	}
	p["model_selection"] = map[string]any{"model": nil, "manifest": manifest, "pointer": nil, "source": "environment"}
	if f, ok := readFixture("status-machine"); ok {
		if m, ok := f["payload"].(map[string]any)["machine"].(map[string]any); ok {
			if pw := os.Getenv("BUBO_FAKE_POWER"); pw == "battery" || pw == "unknown" {
				m["windows"].(map[string]any)["power_source"].(map[string]any)["value"] = pw
			}
			p["machine"] = m
		}
	}
}

// httpInstance replaces the zero counters of an instance status with the
// state the serve process wrote, and adds the Pi snippet and memory objects.
func httpInstance(p map[string]any, i map[string]any) {
	uiStatus(p, i)
	st := readHTTPState()
	p["counters"] = st.Counters
	if names := os.Getenv("BUBO_FAKE_NULL_COUNTERS"); names != "" {
		c := map[string]any{}
		for k, v := range st.Counters {
			c[k] = v
		}
		for _, n := range strings.Split(names, ",") {
			c[strings.TrimSpace(n)] = nil
		}
		p["counters"] = c
	}
	p["requests"] = st.Requests
	slot := idleSlot(0)
	p["sessions"] = 0
	if st.Stuck {
		slot["processing"] = true
		slot["state"] = "generating"
		p["sessions"] = 1
	}
	p["slots"] = []any{slot}
	p["live_throughput"] = map[string]any{"label": "aggregate wall throughput", "unit": "tokens_per_second", "interval_ms": nil, "prefill_tokens_per_second": nil, "decode_tokens_per_second": nil}
	p["last_completed_request"] = map[string]any{"prefill_tokens_per_second": nil, "decode_tokens_per_second": nil, "age_ms": nil, "rate_unit": "tokens_per_second", "age_unit": "milliseconds"}
	httpAlways(p)
}

// writeSmokeBundle writes the seven allow-list members. The leak mode copies
// the planted key into one of them.
func writeSmokeBundle(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	z := zip.NewWriter(f)
	for _, name := range []string{"versions.json", "settings.json", "status.json", "diagnose.json", "history.json", "logs.json", "manifest.json"} {
		w, e := z.Create(name)
		if e != nil {
			return e
		}
		text := "{\"member\":\"" + name + "\"}\n"
		if os.Getenv("BUBO_FAKE_BUNDLE_LEAK") == "1" && name == "settings.json" {
			text = "{\"key\":\"" + os.Getenv("BUBO_API_KEY") + "\"}\n"
		}
		w.Write([]byte(text))
	}
	if e := z.Close(); e != nil {
		return e
	}
	return f.Close()
}

// A serve process that echoes the planted key variable, when asked to.
func init() {
	if len(os.Args) > 1 && os.Args[1] == "serve" && os.Getenv("BUBO_FAKE_ENV_LEAK") == "1" {
		if v := os.Getenv("GUFO_EVAL_API_KEY"); v != "" {
			os.Stderr.WriteString("environment dump: GUFO_EVAL_API_KEY=" + v + "\n")
		}
	}
}
