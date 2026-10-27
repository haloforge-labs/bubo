package main

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
)

func uiEffective() map[string]any {
	return map[string]any{
		"profile":            row("default", "default", "default", nil),
		"context":            row(nil, 200000, "default", nil),
		"sessions":           row(nil, 1, "default", nil),
		"--npu-prefill":      row("on", "off", "profile", "Test CLI has no NPU support."),
		"--reasoning-effort": row(nil, "medium", "default", nil),
	}
}

// uiStatus adds the runtime objects a window shows. Every value uses the
// contract shape; the numbers are test data and say so in their reasons.
func uiStatus(p map[string]any, i map[string]any) {
	amount := func(b uint64, ev string) map[string]any {
		return map[string]any{"bytes": b, "gib": float64(b) / 1073741824.0, "evidence": ev}
	}
	p["effective_settings"] = i["effective_settings"]
	p["live_throughput"] = map[string]any{"label": "aggregate wall throughput", "unit": "tokens_per_second", "interval_ms": 1000, "prefill_tokens_per_second": 12.5, "decode_tokens_per_second": 7.25}
	p["last_completed_request"] = map[string]any{"prefill_tokens_per_second": 42.5, "decode_tokens_per_second": 19.75, "age_ms": 6000, "rate_unit": "tokens_per_second", "age_unit": "milliseconds"}
	p["counters"] = map[string]any{"prefill_tokens_total": 256, "decode_tokens_total": 64, "cached_tokens_total": 256, "cache_disk_hits_total": 2, "cache_memory_hits_total": 4, "cache_misses_total": 1, "max_sequence_tokens": 320, "draft_rounds_total": 0, "draft_tokens_total": 0, "draft_accepted_tokens_total": 0, "device_lost_total": 0, "prefill_seconds_total": 1.5, "decode_seconds_total": 2.5}
	p["memory_estimate"] = map[string]any{"table_version": 1, "evidence": "estimated", "byte_unit": "bytes", "display_unit": "GiB", "fit_verified": false, "reason": "Fixture only; no hardware allocation.",
		"weights": amount(2147483648, "estimated"), "per_session_state": amount(1048576, "estimated"), "session_state_total": amount(1048576, "estimated"), "sessions": 1, "context_tokens": 200000,
		"scratch_phase_allowance": map[string]any{"min": amount(1048576, "estimated"), "max": amount(2097152, "estimated")}, "host_cache_snapshot_allowance": amount(1048576, "estimated"), "reserve": amount(1048576, "estimated"),
		"total_range": map[string]any{"min": amount(2147483648, "estimated"), "max": amount(3221225472, "estimated")}, "carve_budget": nil, "host_budget": nil,
		"reference_coefficient": map[string]any{"bytes": 1048576, "gib": 0.0009765625, "evidence": "measured", "context_tokens": 262144, "conditions": "Fixture.", "source_path": "fixture"}, "formula": "fixture"}
	p["memory_estimate_reason"] = nil
	// The machine object exists only when the contract fixtures supply it.
	if f, ok := readFixture("status-machine"); ok {
		if m, ok := f["payload"].(map[string]any)["machine"]; ok {
			p["machine"] = m
		}
	}
	// The CLI gives a Pi snippet for a loopback endpoint only: the dummy key when
	// the server needs none, the env variable name BUBO_API_KEY when it does.
	if ep, ok := i["endpoint"].(string); ok && (strings.HasPrefix(ep, "http://127.0.0.1:") || strings.HasPrefix(ep, "http://[::1]:")) {
		apiKey := "bubo-loopback"
		if i["authenticated"] == true {
			apiKey = "BUBO_API_KEY"
		}
		p["pi_snippet"] = map[string]any{"providers": map[string]any{"bubo": map[string]any{"baseUrl": ep, "api": "openai-completions", "apiKey": apiKey, "models": []any{map[string]any{"id": servedName(i), "name": servedName(i), "contextWindow": 200000}}}}}
	}
	_ = p
}

// servedName is the model name the running server serves; a status process
// does not see the serve flags, so it reads the instance record.
func servedName(i map[string]any) string {
	if name, ok := i["served_model"].(string); ok && name != "" {
		return name
	}
	return servedModelName()
}
func uiPlan(component, digest string) {
	license := "Fixture licence — 日本語. Local proof only; no download occurs."
	for n, kind := range []string{"plan", "source", "license", "volume", "path"} {
		p := map[string]any{"component": component, "ready": false, "sources": []any{}, "licenses": []any{}, "paths": map[string]any{}, "volumes": []any{}, "plan_digest": digest, "record_kind": kind, "part": n, "parts": 5}
		switch kind {
		case "source":
			p["sources"] = []any{map[string]any{"file_id": "fixture", "url": "https://example.invalid/" + component, "size": 5, "sha256": strings.Repeat("b", 64), "repository": "fixture", "revision": "0123456789abcdef0123456789abcdef01234567", "path": "Modèles 日本語/" + component + ".bin", "license_id": "fixture"}}
		case "license":
			p["licenses"] = []any{map[string]any{"path": "LICENSE", "text": license, "sha256": strings.Repeat("c", 64), "text_offset": 0, "text_bytes": len([]byte(license))}}
		case "volume":
			free := 1000
			if _, err := os.Stat(filepath.Join(dir, "volume-short")); err == nil {
				free = 1
			}
			p["volumes"] = []any{map[string]any{"name": "C:", "free_bytes": free, "required_bytes": 5}}
		case "path":
			p["paths"] = map[string]any{component: "C:/fixture/Modèles 日本語/" + component + "-manifest.json"}
		}
		event("plan", p)
	}
}
func uiBundlePreview(saved bool) map[string]any {
	text := "{\"product\":\"fake-bubo\"}\n"
	return map[string]any{"schema_version": 1, "saved": saved, "bundle_sha256": strings.Repeat("d", 64), "zip_bytes": 100, "files": []string{"versions.json"}, "text_preview": map[string]string{"versions.json": text}, "sha256": map[string]string{"versions.json": strings.Repeat("e", 64)}}
}
func writeUIBundle(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	z := zip.NewWriter(f)
	w, err := z.Create("versions.json")
	if err == nil {
		_, err = w.Write([]byte("{\"product\":\"fake-bubo\"}\n"))
	}
	e := z.Close()
	if err == nil {
		err = e
	}
	e = f.Close()
	if err == nil {
		err = e
	}
	return err
}
