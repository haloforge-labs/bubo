package supervisor

import "time"

// StatusFeed describes freshness independently of the server lifecycle and the
// token-progress watchdog. An idle server can have a healthy status stream.
type StatusFeed struct {
	State      string    `json:"state"`
	SampleTime string    `json:"sample_time"`
	AgeSeconds *float64  `json:"age_seconds"`
	Error      *CLIError `json:"error"`
}

const (
	StatusInactive     = "inactive"
	StatusWaiting      = "waiting"
	StatusUpdating     = "updating"
	StatusReconnecting = "reconnecting"
	StatusNotUpdating  = "not_updating"
	StatusPaused       = "paused"
	StatusUnsupported  = "unsupported"
)

// statusFreshness uses the CLI's sample_time for age, and also detects silence
// before the first record. Pausing never pretends that old data is current.
func statusFreshness(v Snapshot, now time.Time) StatusFeed {
	f := v.StatusFeed
	if f.State == "" {
		// Offline recordings have no live observation clock. Preserve their
		// sample time without fabricating a current age on every regeneration.
		f.State = StatusInactive
		if v.Status != nil {
			f.SampleTime = v.Status.SampleTime
		}
		return f
	}
	if f.SampleTime == "" && v.Status != nil {
		f.SampleTime = v.Status.SampleTime
	}
	if f.SampleTime != "" {
		if sampled, err := time.Parse(time.RFC3339Nano, f.SampleTime); err == nil {
			age := max(0, now.Sub(sampled).Seconds())
			f.AgeSeconds = &age
		}
	}
	if f.State == StatusWaiting || f.State == StatusUpdating || f.State == StatusReconnecting {
		silent := now.Sub(v.StatusRecordAt) >= v.StatusStaleTimeout
		old := f.AgeSeconds != nil && *f.AgeSeconds >= v.StatusStaleTimeout.Seconds()
		if v.StatusStaleTimeout > 0 && (silent || old) {
			f.State = StatusNotUpdating
		}
	}
	return f
}
