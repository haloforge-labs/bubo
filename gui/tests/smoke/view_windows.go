//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"bubo/gui/supervisor"
)

type viewCapture struct {
	mu       sync.Mutex
	builder  supervisor.ViewBuilder
	key      string
	latest   supervisor.View
	received time.Time
	count    int
	err      error
	file     *os.File
}

func (r *Runner) receipt() time.Time {
	r.views.mu.Lock()
	defer r.views.mu.Unlock()
	return r.views.received
}

// This is the same Snapshot -> ViewBuilder boundary as Wails GetView. The
// receipt clock is harness telemetry, not a second frontend model. It measures
// actual validated watch arrivals even when CLI sample_time repeats. Assertions
// below this boundary consume View only. Cache one View per publication and
// save every constructed MVP View, avoiding repeated large idle disk writes.
func (r *Runner) view(sup *supervisor.Supervisor) supervisor.View {
	r.views.mu.Lock()
	defer r.views.mu.Unlock()
	s := sup.Snapshot()
	if !r.cfg.MVP {
		return r.views.builder.Build(supervisor.ViewInput{Snapshot: s, Registry: sup.Registry(), Logs: sup.Logs()})
	}
	key := fmt.Sprintf("%d|%s|%d|%s|%s|%v|%d", r.supN, s.State, s.Generation, s.StatusRecordAt.Format(time.RFC3339Nano), s.StatusFeed.State, s.LastError, len(sup.Logs()))
	if key == r.views.key {
		return r.views.latest
	}
	r.views.key = key
	r.views.latest = r.views.builder.Build(supervisor.ViewInput{Snapshot: s, Registry: sup.Registry(), Logs: sup.Logs(), Source: map[bool]string{true: "fixture", false: "recorded"}[r.cfg.Fake]})
	r.views.received = s.StatusRecordAt
	if r.views.file == nil && r.views.err == nil {
		r.views.file, r.views.err = os.OpenFile(filepath.Join(r.cfg.Out, "views.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	}
	if r.views.err == nil {
		r.views.err = json.NewEncoder(r.views.file).Encode(struct {
			ObservedAt       string          `json:"observed_at"`
			StatusReceivedAt string          `json:"status_received_at"`
			View             supervisor.View `json:"view"`
		}{stamp(), s.StatusRecordAt.UTC().Format(time.RFC3339Nano), r.views.latest})
		r.views.count++
	}
	return r.views.latest
}

func viewForced(v supervisor.View) bool { return v.Failure != nil && v.Failure.Forced }

// Only the retained v1 diagnostic helpers need raw wire values. The MVP reads
// typed View settings directly, without building another model or fact table.
func viewRaw(v supervisor.Value) json.RawMessage {
	var plain func(supervisor.Value) any
	plain = func(v supervisor.Value) any {
		switch v.Kind {
		case "string":
			return v.Text
		case "number":
			return json.Number(v.Text)
		case "boolean":
			return v.Boolean
		case "array":
			out := make([]any, len(v.Items))
			for i, x := range v.Items {
				out[i] = plain(x)
			}
			return out
		case "object":
			out := map[string]any{}
			for k, x := range v.Properties {
				out[k] = plain(x)
			}
			return out
		}
		return nil
	}
	b, _ := json.Marshal(plain(v))
	return b
}

func viewRawSettings(in map[string]supervisor.Value) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(in))
	for k, v := range in {
		out[k] = viewRaw(v)
	}
	return out
}
