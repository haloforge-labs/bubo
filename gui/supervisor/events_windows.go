//go:build windows

package supervisor

// StreamEvent carries already scrubbed data from CLI ingest. Receivers must
// return promptly; Wails forwards it to its asynchronous event transport.
type StreamKind uint8

const (
	PrepareProgress StreamKind = iota
	DownloadProgress
	LogLine
	UpdateProgressStream
	// Internal readiness notification; AppService consumes it before throttling.
	runtimeReadyStream
)

type StreamEvent struct {
	Kind     StreamKind
	Progress *Progress
	Job      *ModelJob
	Line     string
	Update   *UpdateProgress
}

func (s *Supervisor) SetEventSink(sink func(StreamEvent)) {
	s.eventMu.Lock()
	defer s.eventMu.Unlock()
	s.eventSink = sink
}
// SetSwitchGuard sets the function that filters the saved serve-switch
// overrides before every start.
func (s *Supervisor) SetSwitchGuard(guard func(Settings) map[string]string) {
	if guard == nil {
		s.switchGuard.Store(nil)
		return
	}
	s.switchGuard.Store(&guard)
}
func (s *Supervisor) emitStream(e StreamEvent) {
	s.eventMu.RLock()
	sink := s.eventSink
	s.eventMu.RUnlock()
	if sink != nil {
		sink(e)
	}
}
func (s *Supervisor) appendLog(line []byte, complete bool) {
	s.logs.Append(line, complete)
	s.emitStream(StreamEvent{Kind: LogLine, Line: string(line)})
}
