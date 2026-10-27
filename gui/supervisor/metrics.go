package supervisor

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// DefaultStallTimeout is the owner-approved default for the no-progress warning.
const DefaultStallTimeout = 120 * time.Second

// RateLabel and RateUnit are the contract wording for live rates.
const (
	RateLabel = "aggregate wall throughput"
	RateUnit  = "tokens_per_second"
)

// Rates are live token rates. A nil rate is unavailable (first sample, a new
// instance, a reset counter or a CLI that sends null); it is never zero.
// Source is "cli" when status.live_throughput carried the number, or
// "client_delta" when the supervisor divided counter deltas by monotonic time.
type Rates struct {
	PrefillTPS *float64 `json:"prefill_tokens_per_second"`
	DecodeTPS  *float64 `json:"decode_tokens_per_second"`
	IntervalMS *float64 `json:"interval_ms"`
	Source     string   `json:"source"`
	Label      string   `json:"label"`
	Unit       string   `json:"unit"`
}

// StallWarning is raised when work is in flight and no token counter moved for
// Seconds. It warns and offers Restart; nothing kills the server by itself.
type StallWarning struct {
	Seconds       int       `json:"seconds"`
	ThresholdSecs int       `json:"threshold_seconds"`
	Since         time.Time `json:"since"`
	Message       string    `json:"message"`
}

func stallMessage(seconds int) string { return fmt.Sprintf("No token progress for %d s", seconds) }

// sampler turns successive status samples of one instance into rates, counter
// deltas and the stall watchdog state. Time is monotonic (time.Now carries it).
// /health is never an input: only the CLI status counters and slots count.
type sampler struct {
	instance   string
	prev       *Counters
	prevAt     time.Time
	signature  string
	progressAt time.Time
}

func u64(p *uint64) uint64 {
	if p == nil {
		return 0
	}
	return *p
}

// counterValues returns the monotonic integer counters, and false if any
// counter the rates need is unavailable.
func counterValues(c *Counters) (map[string]uint64, bool) {
	if c == nil {
		return nil, false
	}
	pairs := map[string]*uint64{"prefill_tokens_total": c.PrefillTokens, "decode_tokens_total": c.DecodeTokens, "cached_tokens_total": c.CachedTokens, "cache_disk_hits_total": c.CacheDiskHits, "cache_memory_hits_total": c.CacheMemoryHits, "cache_misses_total": c.CacheMisses}
	out := map[string]uint64{}
	ok := true
	for k, v := range pairs {
		if v == nil {
			ok = false
			continue
		}
		out[k] = *v
	}
	return out, ok
}

// inFlight reports whether the CLI says work is running: a slot that is not
// idle, or queued requests. known is false when the CLI sent neither slots nor
// a queue (unavailable); then no stall can be judged.
func inFlight(st *Status) (busy, known bool) {
	if st.Slots != nil {
		known = true
		for _, s := range st.Slots {
			if s.State != "idle" || s.Processing {
				busy = true
			}
		}
	}
	if st.Queue != nil {
		known = true
		if *st.Queue > 0 {
			busy = true
		}
	}
	return
}

// progressSignature changes whenever any token counter or any slot token count
// moves. Slot counts matter because the totals can move only when a request
// completes.
func progressSignature(st *Status) string {
	var b strings.Builder
	if c := st.Counters; c != nil {
		fmt.Fprintf(&b, "c%d,%d,%d,%d;", u64(c.PrefillTokens), u64(c.DecodeTokens), u64(c.CachedTokens), u64(c.CompletedSuccessful))
	}
	slots := append([]Slot(nil), st.Slots...)
	sort.Slice(slots, func(i, j int) bool { return slots[i].ID < slots[j].ID })
	for _, s := range slots {
		fmt.Fprintf(&b, "s%d:%d,%d,%d,%d,%d;", s.ID, u64(s.TokensUsed), u64(s.PrefilledTokens), u64(s.GeneratedTokens), u64(s.PromptTokens), u64(s.RequestID))
	}
	return b.String()
}

// observe consumes one status sample. timeout <= 0 disables the watchdog.
func (m *sampler) observe(id string, st *Status, now time.Time, timeout time.Duration) (rates Rates, delta map[string]uint64, reset bool, stall *StallWarning) {
	if m.instance != id {
		*m = sampler{instance: id}
	}
	rates = Rates{Source: "", Label: RateLabel, Unit: RateUnit}
	if t := st.LiveThroughput; t != nil && (t.PrefillTPS != nil || t.DecodeTPS != nil) {
		rates.PrefillTPS, rates.DecodeTPS, rates.IntervalMS, rates.Source = t.PrefillTPS, t.DecodeTPS, t.IntervalMS, "cli"
	}
	reset = true
	if st.Counters != nil {
		if m.prev != nil && !m.prevAt.IsZero() {
			var base CounterBaseline
			if old, ok := counterValues(m.prev); ok {
				base = CounterBaseline{Instance: id, Values: old}
				if cur, ok := counterValues(st.Counters); ok {
					d, r := base.Delta(id, cur)
					delta, reset = d, r
					dt := now.Sub(m.prevAt).Seconds()
					if !r && dt > 0 && rates.Source == "" {
						ms := dt * 1000
						p, q := float64(d["prefill_tokens_total"])/dt, float64(d["decode_tokens_total"])/dt
						rates.PrefillTPS, rates.DecodeTPS, rates.IntervalMS, rates.Source = &p, &q, &ms, "client_delta"
					}
				}
			}
		}
		c := *st.Counters
		m.prev, m.prevAt = &c, now
	} else {
		m.prev = nil
	}
	if reset {
		delta = nil
	}
	busy, known := inFlight(st)
	sig := progressSignature(st)
	if m.progressAt.IsZero() || sig != m.signature || !busy {
		m.progressAt = now
	}
	m.signature = sig
	if timeout > 0 && known && busy {
		if idle := now.Sub(m.progressAt); idle >= timeout {
			secs := int(idle / time.Second)
			stall = &StallWarning{Seconds: secs, ThresholdSecs: int(timeout / time.Second), Since: time.Now().Add(-idle).UTC(), Message: stallMessage(secs)}
		}
	}
	return
}

// PiSnippetText is the CLI provider fragment as indented JSON text for copying.
func PiSnippetText(p *PiSnippet) string {
	if p == nil {
		return ""
	}
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return ""
	}
	return string(b)
}
