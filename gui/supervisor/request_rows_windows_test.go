//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// E14 item 5 (CLI-CONTRACT "requests", ms-memguard 342001a1): every request
// row has status and error_code; outcome "refused" rows carry the refusal.
func TestRequestRowsCarryStatusErrorCodeAndRefusal(t *testing.T) {
	s := supervisorFor(t, "ui")
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600)
	reply(t, s.Start())
	waitState(t, s, Ready)
	row := func(id, outcome string, status any, code any, extra map[string]any) map[string]any {
		m := map[string]any{"id": id, "time": "2026-10-20T08:00:00Z", "slot": nil, "prompt_tokens": 5, "completion_tokens": nil, "cached_tokens": nil, "ttft_ms": nil, "prefill_tps": nil, "decode_tps": nil, "cache_kind": nil, "finish_reason": nil, "outcome": outcome, "status": status, "error_code": code}
		for k, v := range extra {
			m[k] = v
		}
		return m
	}
	scriptStatus(t, map[string]any{
		"counters": countersJSON(1, 1), "sessions": 1, "queue": 0, "slots": []any{},
		"requests": []any{
			row("r1", "refused", 503, "insufficient_memory", map[string]any{"refusal": map[string]any{"code": "insufficient_memory", "needed_bytes": 8589934592, "free_bytes": 2147483648}}),
			row("r2", "failed", 404, "model_not_found", nil),
			row("r3", "refused", 503, "memory_pressure", map[string]any{"refusal": map[string]any{"code": "memory_pressure", "needed_bytes": nil, "free_bytes": nil}}),
		},
	})
	v := waitFor(t, s, "rows", 8*time.Second, func(v Snapshot) bool { return v.Status != nil && len(v.Status.Requests) == 3 })
	r := v.Status.Requests
	if r[0].Outcome != "refused" || r[0].Status == nil || *r[0].Status != 503 || r[0].ErrorCode == nil || *r[0].ErrorCode != "insufficient_memory" {
		t.Fatalf("refused row: %+v", r[0])
	}
	if r[0].Refusal == nil || r[0].Refusal.NeededBytes == nil || *r[0].Refusal.NeededBytes != 8589934592 || *r[0].Refusal.FreeBytes != 2147483648 {
		t.Fatalf("refusal lost: %+v", r[0].Refusal)
	}
	if r[1].Status == nil || *r[1].Status != 404 || r[1].Refusal != nil {
		t.Fatalf("failed row: %+v", r[1])
	}
	if r[2].Refusal == nil || r[2].Refusal.NeededBytes != nil || r[2].Refusal.FreeBytes != nil {
		t.Fatalf("memory_pressure numbers must be null: %+v", r[2].Refusal)
	}
	view := (&ViewBuilder{}).Build(ViewInput{Snapshot: v})
	if len(view.Requests) != 3 || view.Requests[0].Refusal == nil {
		t.Fatalf("the View lost the refusal: %+v", view.Requests)
	}
}
