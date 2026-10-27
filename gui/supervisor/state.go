package supervisor

import (
	"fmt"
	"net/url"
	"strconv"
	"time"
)

type State string

// StartupPhase describes app initialization independently of server lifecycle.
type StartupPhase string

const StartingCLI StartupPhase = "starting_cli"

const (
	NotPrepared      State = "Not prepared"
	Preparing        State = "Preparing"
	Starting         State = "Starting"
	Ready            State = "Ready"
	Stopping         State = "Stopping"
	Stopped          State = "Stopped"
	Failed           State = "Failed"
	AttachedExternal State = "Attached-external"
)

type Failure struct {
	Code   string    `json:"code"`
	Source string    `json:"source"`
	Time   time.Time `json:"time"`
	Remedy string    `json:"remedy"`
	Detail string    `json:"detail,omitempty"`
	// Again marks a device_lost that followed the app's own restart.
	Again bool `json:"again,omitempty"`
	// Memory is set for a memory_insufficient failure.
	Memory    *MemoryShortfall `json:"memory,omitempty"`
	Forced    bool             `json:"forced"`
	Component string           `json:"component,omitempty"`
	Stage     string           `json:"stage,omitempty"`
	Switch    string           `json:"switch,omitempty"`
}

// ServerWarning is a lifecycle warning the running server sent. The code and
// the CLI's remedy show under Details only.
type ServerWarning struct {
	Code   string `json:"code"`
	Remedy string `json:"remedy"`
}

type Snapshot struct {
	// ServerWarnings are the distinct warnings of the server that runs.
	ServerWarnings []ServerWarning
	// AutoRestarted is true while the server that runs was restarted by the
	// supervisor after the engine lost the GPU (exit status 75).
	AutoRestarted bool
	Initialized   bool
	StartupPhase  StartupPhase
	// StartupWaiting is set when the first CLI call has waited past the notice
	// time; the View then says "Starting for the first time…".
	StartupWaiting      bool
	State               State
	StopPhase           string
	Generation          uint64
	Instance            *Instance
	LastError           *Failure
	Forced              bool
	Stale               bool
	Managed             bool
	ManagementSupported bool
	ManagementMessage   string
	Label               string
	SettingsWarning     string
	LogWarning          string
	Approval            *Approval
	OperationID         string
	Progress            *Progress
	Prepared            bool
	Preflight           *Preflight
	Preflights          []Preflight
	PlanParts           []Plan
	CounterDelta        map[string]uint64
	CountersReset       bool
	Status              *Status
	LiveSlots           map[uint64]LiveSlotMetrics
	LiveSamples         []LiveSample
	LiveSampleTime      string
	StatusFeed          StatusFeed
	StatusRecordAt      time.Time
	StatusStaleTimeout  time.Duration
	Discovered          []Instance
	Diagnostic          string
	// Rates are live token rates from the latest owned or attached status.
	// Stall is set while work is in flight and no token counter has moved for
	// the configured time. Both come only from CLI status samples.
	Rates          *Rates
	Stall          *StallWarning
	RegistryLoaded bool
	// User requests handed to the current serve child, retained to identify a
	// cleared request whose next CLI default is not yet observable.
	RunningSwitchOverrides map[string]string
	// GPU is the GUI's own machine GPU reading (not a CLI key). Null when no
	// source is set or nothing came back; each field inside is optional.
	GPU *GPUStats `json:"gpu"`
}
type Transition struct {
	From   State
	Action string
	To     State
}

// Transitions returns the complete set of state-changing inputs. Same-state
// updates (progress, stale samples, retained errors) do not create transitions.
var transitionTable = []Transition{
	{NotPrepared, "prepare", Preparing}, {Stopped, "prepare", Preparing}, {Failed, "prepare", Preparing},
	{Stopped, "start", Starting}, {Failed, "start", Starting},
	{Preparing, "prepared", Stopped}, {Preparing, "cancel", Stopping},
	{Starting, "verified-ready", Ready}, {Starting, "stop", Stopping}, {Ready, "stop", Stopping},
	{Stopping, "clean-exit", Stopped}, {Stopping, "failure", Failed},
	{Preparing, "failure", Failed}, {Starting, "failure", Failed}, {Ready, "failure", Failed},
	{NotPrepared, "prepared", Stopped}, {Stopped, "unprepared", NotPrepared}, {Failed, "unprepared", NotPrepared},
	{NotPrepared, "attach", AttachedExternal}, {Stopped, "attach", AttachedExternal}, {Failed, "attach", AttachedExternal},
	{AttachedExternal, "detach", Stopped}, {AttachedExternal, "external-exit", Stopped},
	{AttachedExternal, "failure", Failed},
	{NotPrepared, "failure", Failed}, {Stopped, "failure", Failed},
}

func Transitions() []Transition { return append([]Transition(nil), transitionTable...) }

func (s *Snapshot) transition(action string) error {
	for _, t := range transitionTable {
		if t.From == s.State && t.Action == action {
			s.State = t.To
			return nil
		}
	}
	return fmt.Errorf("action %s is not available in %s", action, s.State)
}
func (s *Snapshot) fail(code, source string, forced bool) {
	s.LastError = &Failure{Code: code, Source: source, Time: time.Now().UTC(), Remedy: safeRemedy(code), Forced: forced}
	s.Forced = s.Forced || forced
	if s.State != Failed {
		_ = s.transition("failure")
	}
}
func safeRemedy(code string) string {
	switch code {
	case "protocol_mismatch":
		return "Use a CLI with a supported JSON contract."
	case "unsupported":
		return "Stop this server in its terminal. Management is not supported."
	case "stop_timeout":
		return "The stop deadline expired. Review the local logs."
	case "startup_timeout":
		return "The load deadline expired. Review the local logs."
	case "query_timeout":
		return "The CLI status deadline expired. Retry the request."
	default:
		return "Review the local logs and retry on request."
	}
}
func ValidatePort(port int) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("port is not permitted")
	}
	return nil
}
func ValidateEndpoint(endpoint *string) error {
	if endpoint == nil {
		return nil
	}
	u, err := url.Parse(*endpoint)
	if err != nil {
		return err
	}
	// 0.0.0.0 is the bind address of local network access; the GUI never
	// connects to it, it only reports what the CLI says it bound.
	if u.Scheme != "http" || u.Hostname() != "127.0.0.1" && u.Hostname() != "0.0.0.0" {
		return protocol("endpoint must use loopback or the local network bind address")
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil {
		return protocol("endpoint has no explicit port")
	}
	return ValidatePort(p)
}
func sameInstance(a, b Instance) bool {
	return a.ID == b.ID && a.PID == b.PID && a.CreationTime == b.CreationTime && a.ExePath == b.ExePath
}
