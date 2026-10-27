//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// MVP test data comes from a CLI recording. It is deliberately fixture-derived
// after adding fake process identity, endpoints and in-flight HTTP state. Never
// use these budgets or context values as production defaults.
func mvpMemory() map[string]any {
	b, err := os.ReadFile(os.Getenv("BUBO_FAKE_MVP_RECORDING"))
	if err != nil {
		panic(err)
	}
	var event struct {
		Payload struct {
			Memory map[string]any `json:"memory_estimate"`
		} `json:"payload"`
	}
	if err = json.Unmarshal(b, &event); err != nil || event.Payload.Memory == nil {
		panic("MVP CLI memory recording missing")
	}
	return event.Payload.Memory
}

// mvpModelID names the fixture's prepared model (test-only, never a catalogue id).
const mvpModelID = "fixture-recorded-model"

// mvpEstimate answers `models estimate` from the same recording the fake
// server resolves its context from, as the real CLI uses one rule for both.
func mvpEstimate(payload map[string]any, sessions int) {
	if sessions == 0 {
		sessions = 1
	}
	m := mvpMemory()
	m["sessions"] = sessions
	m["context_tokens"] = mvpRecommendation(sessions)["context_per_session"]
	m["fits"] = true
	m["recommended"] = true
	payload["estimate_state"] = "estimated"
	payload["reason"] = nil
	payload["memory_estimate"] = m
}

func mvpRecommendation(sessions int) map[string]any {
	m := mvpMemory()
	rc := m["recommended_context"].(map[string]any)
	for _, v := range rc["by_sessions"].([]any) {
		row := v.(map[string]any)
		if row["sessions"].(float64) == float64(sessions) {
			return row
		}
	}
	panic("CLI recording has no requested session recommendation")
}

func mvpEffective(out map[string]any) {
	n, _ := strconv.Atoi(flag("--sessions"))
	if n == 0 {
		n = 1
	}
	if flag("--context") == "" {
		source := "default"
		if n > 1 {
			source = "recommended_context"
		}
		out["context"] = row(nil, mvpRecommendation(n)["context_per_session"], source, "Fixture: copied from CLI recommendation.")
		if n == 2 && os.Getenv("BUBO_FAKE_MVP_FAULT") == "wrong-context" {
			// A valid but incorrect resolution, consistent in ready/discovery
			// and status, so the smoke's recommendation comparison must fail.
			out["context"] = row(nil, mvpRecommendation(1)["context_per_session"], source, "Fixture: incorrect two-session resolution.")
		}
	}
	b, _ := json.Marshal(os.Args[1:])
	f, err := os.OpenFile(filepath.Join(dir, "mvp-serve-args.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		panic(err)
	}
	f.Write(append(b, '\n'))
	f.Close()
}

func mvpStatus(p, i map[string]any) {
	p["paths"].(map[string]any)["runtime_kind"] = "shared"
	m := mvpMemory()
	n := 1
	var effective map[string]any
	if i != nil {
		effective = i["effective_settings"].(map[string]any)
		n = int(effective["sessions"].(map[string]any)["effective"].(float64))
	}
	if !packagesMode() {
		// The real CLI lists the prepared package named by BUBO_MANIFEST; the
		// GUI estimates its memory by this id before any server runs.
		p["prepared_models"] = []any{map[string]any{"id": mvpModelID, "display_name": "Fixture recorded model", "manifest_path": os.Getenv("BUBO_MANIFEST"), "model_path": nil, "size_bytes": 1, "prepared_state": "prepared", "capabilities": map[string]any{"context_max": 262144, "mtp": false, "twin": false, "mmproj": false}}}
	}
	if i == nil {
		// Real status with no server (2026-10-15 04:29 recording): no
		// memory estimate, only the reason.
		p["memory_estimate"] = nil
		p["memory_estimate_reason"] = "No active server allocation settings are available."
		return
	}
	m["sessions"] = n
	m["context_tokens"] = mvpRecommendation(n)["context_per_session"]
	m["fits"] = true
	m["recommended"] = true
	m["recommendation_reason"] = "Fixture allocation copied from recorded recommendation."
	p["memory_estimate"] = m
	p["memory_estimate_reason"] = nil
	context := effective["context"].(map[string]any)["effective"]
	m["context_tokens"] = context
	ep := i["endpoint"].(string)
	p["endpoints"] = map[string]any{"openai_base_url": ep, "anthropic_base_url": ep[:len(ep)-len("/v1")]}
	p["api_key_env"] = "BUBO_API_KEY"
	state := readHTTPState()
	slots := make([]any, n)
	busy := 0
	for k := 0; k < n; k++ {
		slot := idleSlot(k)
		slot["context_tokens"] = context
		if state.Active[k] {
			slot["processing"] = true
			slot["state"] = "generating"
			busy++
		}
		slots[k] = slot
	}
	p["slots"] = slots
	p["sessions"] = busy
	// Live rates intentionally remain null: the fake cannot qualify real rates.
	switch os.Getenv("BUBO_FAKE_MVP_FAULT") {
	case "missing-endpoints":
		p["endpoints"] = nil
	}
}

func mvpChat(w http.ResponseWriter, r *http.Request, body fakeChatBody) {
	hmu.Lock()
	if hstate.Active == nil {
		hstate.Active = map[int]bool{}
	}
	slot := 0
	for hstate.Active[slot] {
		slot++
	}
	hstate.Active[slot] = true
	reqSeq++
	id := fmt.Sprintf("r%d", reqSeq)
	saveHTTPState()
	hmu.Unlock()
	if os.Getenv("BUBO_FAKE_MVP_FAULT") == "watch-silence" {
		_ = os.WriteFile(filepath.Join(dir, "watch-stall"), nil, 0600)
	}
	ms, _ := strconv.Atoi(os.Getenv("BUBO_FAKE_REQUEST_MS"))
	if ms <= 0 {
		ms = 2600
	}
	timer := time.NewTimer(time.Duration(ms) * time.Millisecond)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-r.Context().Done():
	}
	hmu.Lock()
	delete(hstate.Active, slot)
	out := min(200, body.MaxTokens)
	bump("prefill_tokens_total", 32)
	bump("decode_tokens_total", out)
	bump("completed_successful_requests_total", 1)
	hstate.Requests = append(hstate.Requests, map[string]any{"id": id, "time": now(), "slot": slot, "prompt_tokens": 32, "completion_tokens": out, "cached_tokens": 0, "ttft_ms": 12.5, "prefill_tps": nil, "decode_tps": nil, "cache_kind": "miss", "finish_reason": "stop", "outcome": "completed", "status": 200, "error_code": nil})
	saveHTTPState()
	hmu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Server-Timing", "ttft;dur=12.5")
	json.NewEncoder(w).Encode(map[string]any{"id": id, "choices": []any{map[string]any{"finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": "Fixture story about an owl."}}}, "usage": map[string]int{"prompt_tokens": 32, "completion_tokens": out, "total_tokens": 32 + out}, "timings": map[string]float64{"prompt_per_second": 2000, "predicted_per_second": 50}})
}
