package supervisor

import (
	"fmt"
	"time"
)

const liveHistoryWindow = 60 * time.Second
const liveCompletionHold = 5 * time.Second

// LiveSample is one received CLI status observation, never a View poll. Rates
// follow StatusWatch in src/cli/status_watch.hpp; no aggregate rate is used.
type LiveSample struct {
	InstanceID string   `json:"instance_id"`
	SlotID     uint64   `json:"slot_id"`
	RequestID  uint64   `json:"request_id"`
	SampleTime string   `json:"sample_time"`
	Phase      string   `json:"phase"`
	PrefillTPS *float64 `json:"prefill_tps"`
	DecodeTPS  *float64 `json:"decode_tps"`
	ElapsedMS  float64  `json:"elapsed_ms"`
	// RetainUntil is assigned only when a later sample shows this request ended.
	RetainUntil string `json:"retain_until,omitempty"`
}

type LiveSlotMetrics struct {
	RequestID  uint64
	PrefillTPS *float64
	DecodeTPS  *float64
	ElapsedMS  *float64
}

type liveSlotObservation struct {
	request                           uint64
	state                             string
	prefilled, generated              *uint64
	at, started, baselineAt, latestAt time.Time
	baseline, latest                  *uint64
}

type liveSlotSampler struct {
	instance   string
	generation uint64
	lastAt     time.Time
	previous   map[uint64]liveSlotObservation
	samples    []LiveSample
}

func liveRequestKey(slot, request uint64) string { return fmt.Sprintf("%d:%d", slot, request) }

func countMovedBack(a, b *uint64) bool { return a != nil && b != nil && *a < *b }

// StatusWatch rejects counters outside the exactly representable integer range.
func watchCount(v *uint64) *uint64 {
	if v == nil || *v > 9007199254740991 {
		return nil
	}
	return v
}

func (m *liveSlotSampler) observe(out *Snapshot, st *Status, generation uint64, now time.Time) {
	if st == nil || st.Identity == nil {
		*m = liveSlotSampler{}
		out.LiveSlots, out.LiveSamples = nil, nil
		out.LiveSampleTime = ""
		return
	}
	if m.instance != st.Identity.ID || m.generation != generation || now.Before(m.lastAt) {
		*m = liveSlotSampler{instance: st.Identity.ID, generation: generation}
	}
	if now.Equal(m.lastAt) {
		return
	}
	m.lastAt = now
	out.LiveSampleTime = now.UTC().Format(time.RFC3339Nano)
	next := map[uint64]liveSlotObservation{}
	metrics := map[uint64]LiveSlotMetrics{}
	active := map[string]bool{}
	for _, slot := range st.Slots {
		if watchCount(slot.RequestID) == nil || slot.ID > 9007199254740991 || (slot.State != "prefilling" && slot.State != "generating") {
			continue
		}
		slot.PrefilledTokens, slot.GeneratedTokens = watchCount(slot.PrefilledTokens), watchCount(slot.GeneratedTokens)
		request := *slot.RequestID
		key := liveRequestKey(slot.ID, request)
		active[key] = true
		cur := liveSlotObservation{request: request, state: slot.State, prefilled: slot.PrefilledTokens, generated: slot.GeneratedTokens, at: now, started: now, baseline: slot.PrefilledTokens, latest: slot.PrefilledTokens, baselineAt: now, latestAt: now}
		var prefill, decode *float64
		if old, ok := m.previous[slot.ID]; ok && old.request == request {
			reset := countMovedBack(slot.PrefilledTokens, old.prefilled) || countMovedBack(slot.GeneratedTokens, old.generated)
			if reset {
				kept := m.samples[:0]
				for _, point := range m.samples {
					if point.SlotID != slot.ID || point.RequestID != request {
						kept = append(kept, point)
					}
				}
				m.samples = kept
			} else {
				cur.started = old.started
				if old.state == slot.State {
					if slot.State == "prefilling" && cur.prefilled != nil && old.prefilled != nil {
						cur.baseline, cur.baselineAt, cur.latest, cur.latestAt = old.baseline, old.baselineAt, old.latest, old.latestAt
						if *cur.prefilled > *old.prefilled {
							cur.latest, cur.latestAt = cur.prefilled, now
						}
						seconds := cur.latestAt.Sub(cur.baselineAt).Seconds()
						if seconds > 0 && cur.baseline != nil && cur.latest != nil && *cur.latest > *cur.baseline {
							rate := float64(*cur.latest-*cur.baseline) / seconds
							prefill = &rate
						}
					} else if slot.State == "generating" && cur.generated != nil && old.generated != nil {
						if seconds := now.Sub(old.at).Seconds(); seconds > 0 {
							rate := float64(*cur.generated-*old.generated) / seconds
							decode = &rate
						}
					}
				}
			}
		}
		elapsed := float64(now.Sub(cur.started)) / float64(time.Millisecond)
		metrics[slot.ID] = LiveSlotMetrics{RequestID: request, PrefillTPS: prefill, DecodeTPS: decode, ElapsedMS: &elapsed}
		next[slot.ID] = cur
		m.samples = append(m.samples, LiveSample{InstanceID: m.instance, SlotID: slot.ID, RequestID: request, SampleTime: now.UTC().Format(time.RFC3339Nano), Phase: slot.State, PrefillTPS: prefill, DecodeTPS: decode, ElapsedMS: elapsed})
	}
	// Ended requests keep their final curve briefly while completed history arrives.
	for i := range m.samples {
		point := &m.samples[i]
		if !active[liveRequestKey(point.SlotID, point.RequestID)] && point.RetainUntil == "" {
			point.RetainUntil = now.Add(liveCompletionHold).UTC().Format(time.RFC3339Nano)
		}
	}
	m.samples = visibleLiveSamples(m.samples, now)
	// A malformed high-frequency source cannot grow memory without a bound.
	if len(m.samples) > 12000 {
		m.samples = append([]LiveSample(nil), m.samples[len(m.samples)-12000:]...)
	}
	m.previous = next
	out.LiveSlots, out.LiveSamples = metrics, m.samples
}

func visibleLiveSamples(samples []LiveSample, now time.Time) []LiveSample {
	var out []LiveSample
	for _, point := range samples {
		at, err := time.Parse(time.RFC3339Nano, point.SampleTime)
		if err != nil || now.Sub(at) > liveHistoryWindow {
			continue
		}
		if point.RetainUntil != "" {
			until, err := time.Parse(time.RFC3339Nano, point.RetainUntil)
			if err != nil || !now.Before(until) {
				continue
			}
		}
		out = append(out, point)
	}
	return out
}
