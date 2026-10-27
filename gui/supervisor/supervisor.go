//go:build windows

package supervisor

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/sys/windows"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Config struct {
	Exe string
	// ProductVersion comes from the installed package manifest, shared with the updater.
	ProductVersion string
	// RuntimeDir is a user override. Empty leaves shared runtime resolution to the CLI.
	RuntimeDir   string
	Port         int
	LogDir       string
	SettingsPath string
	LoadTimeout  time.Duration
	StopTimeout  time.Duration
	DrainTimeout time.Duration
	QueryTimeout time.Duration
	// ColdStartTimeout bounds each initial discovery attempt, including the
	// identity handshake. Only bootstrap gets the larger budget and retries.
	ColdStartTimeout      time.Duration
	ColdStartRetryBackoff time.Duration
	// FirstCallNotice is how long the first CLI call may wait before the View
	// says "Starting for the first time…" (default 3 s). Windows can scan a
	// freshly unpacked unsigned program on its first run for 15 s or more.
	FirstCallNotice time.Duration
	PollInterval    time.Duration
	// App freshness and hidden-window policy, independent of CLI cadence.
	StatusStaleTimeout  time.Duration
	HiddenStatusTimeout time.Duration
	// StallTimeout is the no-token-progress warning time (default 120 s;
	// negative disables). The watchdog warns; it never stops the server.
	StallTimeout    time.Duration
	InitialPrepared bool
	// AutoRestartDelay is the wait before the supervisor restarts a server that
	// exited with status 75 (the engine lost the GPU). AutoRestartWindow is how
	// long after such a restart a second status 75 stops instead of restarting.
	AutoRestartDelay  time.Duration
	AutoRestartWindow time.Duration
	// Keys holds the serve API key for local network access. Nil uses
	// Windows Credential Manager.
	Keys *KeyManager
	// GPU is the optional GPU stats source (NewGPUSource on Windows). Nil
	// means snapshot.gpu stays null. The supervisor polls it at the status
	// interval, never faster than once a second, and closes it on exit.
	GPU GPUSource
}
type ResultReply struct{ Err error }

var testLoopbackOnly bool

type lanUnavailableError struct{}

func (*lanUnavailableError) Error() string {
	return "Local network access isn't available in this copy of Bubo."
}
func (*lanUnavailableError) ErrorCode() string { return "lan_unavailable" }

type DiagnosticReply struct {
	Event Event
	Err   error
}
type CommandError struct {
	Event    Event
	ExitCode uint32
}

func (e *CommandError) Error() string {
	c := e.Event.Value.(*CLIError)
	return "CLI " + c.Code + ": " + c.Remedy
}

type request struct {
	action   string
	args     []string
	input    []byte
	reply    chan ResultReply
	instance *Instance
}
type update struct {
	gen   uint64
	kind  string
	value any
}
type operation struct {
	gen   uint64
	stop  chan struct{}
	input chan []byte
	done  chan struct{}
	// pin is the saved setup an automatic restart must use again. Nil reads
	// the settings file, as every start the user makes does.
	pin *Settings
}
type Supervisor struct {
	processMu  sync.Mutex
	processes  map[*OwnedProcess]struct{}
	eventMu    sync.RWMutex
	eventSink  func(StreamEvent)
	ctx        context.Context
	cancel     context.CancelFunc
	helperMu   sync.Mutex
	helpers    sync.WaitGroup
	closing    bool
	submitMu   sync.Mutex
	closed     bool
	config     Config
	logs       *Logs
	commands   chan request
	updates    chan update
	published  atomic.Pointer[Snapshot]
	finished   chan struct{}
	pollNanos  atomic.Int64
	visible    atomic.Bool
	visibility chan struct{}
	// selection is the model the user selected, as status arguments
	// (--manifest PATH or --model PATH); none reports what plain status does.
	selection atomic.Pointer[[]string]
	stallNano atomic.Int64
	registry  atomic.Pointer[Switches]
	regOnce   sync.Once
	// cliAnswered is set after the first successful CLI call.
	cliAnswered atomic.Bool
	keys        *KeyManager
	gpuLatest   atomic.Pointer[GPUStats]
}

var crashHandler atomic.Value // func(string), installed by the native host

func SetCrashHandler(handler func(string)) { crashHandler.Store(handler) }

var crashLogDir atomic.Value // string: the app's log folder

func setCrashLog(dir string) { crashLogDir.Store(dir) }

// writeCrashLog keeps the cause and the stack: the window shows one fixed
// sentence, so the log is the only place a person can find out what broke.
func writeCrashLog(cause any) {
	dir, _ := crashLogDir.Load().(string)
	if dir == "" {
		return
	}
	if os.MkdirAll(dir, 0700) != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dir, "gui-crash.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s panic: %v\n%s\n\n", time.Now().UTC().Format(time.RFC3339), cause, debug.Stack())
}

// ReportPanic is deferred at a Bubo goroutine root. A fatal error is shown by
// the native host even when the WebView cannot render, then exits non-zero.
func ReportPanic() {
	if cause := recover(); cause != nil {
		writeCrashLog(cause)
		handler, _ := crashHandler.Load().(func(string))
		if handler == nil {
			panic(cause)
		}
		handler(NativeCrashSentence)
		os.Exit(1)
	}
}

func RunProtected(fn func()) { go func() { defer ReportPanic(); fn() }() }

func New(c Config) (*Supervisor, error) {
	if c.LogDir != "" {
		setCrashLog(c.LogDir)
	}
	if !filepath.IsAbs(c.Exe) {
		return nil, fmt.Errorf("CLI path must be absolute")
	}
	if c.RuntimeDir != "" && !filepath.IsAbs(c.RuntimeDir) {
		return nil, fmt.Errorf("runtime folder must be absolute")
	}
	if c.LoadTimeout <= 0 {
		c.LoadTimeout = 600 * time.Second
	}
	if c.StopTimeout <= 0 {
		c.StopTimeout = 45 * time.Second
	}
	if c.DrainTimeout <= 0 {
		c.DrainTimeout = 2 * time.Second
	}
	if c.QueryTimeout <= 0 {
		c.QueryTimeout = 10 * time.Second
	}
	if c.ColdStartTimeout <= 0 {
		c.ColdStartTimeout = 60 * time.Second
	}
	if c.ColdStartRetryBackoff <= 0 {
		c.ColdStartRetryBackoff = 250 * time.Millisecond
	}
	if c.FirstCallNotice <= 0 {
		c.FirstCallNotice = 3 * time.Second
	}
	if c.PollInterval <= 0 {
		c.PollInterval = 2 * time.Second
	}
	// Status 75 means the GPU was lost. A few seconds let the driver settle; a
	// second 75 within five minutes of a restart is not retried again.
	if c.AutoRestartDelay <= 0 {
		c.AutoRestartDelay = 5 * time.Second
	}
	if c.AutoRestartWindow <= 0 {
		c.AutoRestartWindow = 5 * time.Minute
	}
	if c.StatusStaleTimeout <= 0 {
		c.StatusStaleTimeout = 5 * time.Second
	}
	if c.HiddenStatusTimeout <= 0 {
		c.HiddenStatusTimeout = 30 * time.Second
	}
	logs, err := NewLogs(c.LogDir)
	if err != nil {
		return nil, err
	}
	s := &Supervisor{config: c, logs: logs, commands: make(chan request, 32), updates: make(chan update, 128), finished: make(chan struct{}), visibility: make(chan struct{}, 1)}
	s.visible.Store(true)
	s.keys = c.Keys
	if s.keys == nil {
		s.keys = NewKeyManager(nil)
	}
	s.pollNanos.Store(int64(c.PollInterval))
	if c.StallTimeout == 0 {
		c.StallTimeout = DefaultStallTimeout
		s.config.StallTimeout = c.StallTimeout
	}
	s.stallNano.Store(int64(c.StallTimeout))
	initial := Snapshot{State: NotPrepared, StartupPhase: StartingCLI, Prepared: c.InitialPrepared, Stale: true}
	if c.InitialPrepared {
		initial.State = Stopped
	}
	if c.SettingsPath != "" {
		settings, warning := LoadSettings(c.SettingsPath)
		if settings.LastError != nil && (c.ProductVersion == "" || settings.lastErrorVersion != c.ProductVersion) {
			settings.LastError = nil
			settings.lastErrorVersion = ""
			if err := SaveSettings(c.SettingsPath, settings); err != nil {
				initial.Label = "Settings save failed."
			}
		}
		initial.LastError = settings.LastError
		if warning != "" {
			initial.Label = warning
			initial.SettingsWarning = warning
		}
	}
	s.ctx, s.cancel = context.WithCancel(context.Background())
	s.publish(initial)
	RunProtected(func() { s.run(initial) })
	if c.GPU != nil {
		RunProtected(func() { s.pollGPU(c.GPU) })
	}
	return s, nil
}
func (s *Supervisor) Logs() []string {
	lines := s.logs.Ring()
	snap := s.Snapshot()
	var path *string
	if snap.Instance != nil {
		path = snap.Instance.LogPath
	}
	if path == nil && snap.Status != nil {
		path = snap.Status.LogPath
	}
	if path == nil || *path == "" {
		return lines
	}
	// The server writes its own log file; stderr alone does not contain its
	// request lines. Read only the tail so View polling stays bounded.
	file, err := os.Open(*path)
	if err != nil {
		return lines
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.IsDir() {
		return lines
	}
	const tailBytes int64 = 256 << 10
	offset := info.Size() - tailBytes
	if offset > 0 {
		if _, err := file.Seek(offset, io.SeekStart); err != nil {
			return lines
		}
	}
	data, err := io.ReadAll(io.LimitReader(file, tailBytes))
	if err != nil {
		return lines
	}
	if offset > 0 {
		if newline := bytes.IndexByte(data, '\n'); newline >= 0 {
			data = data[newline+1:]
		}
	}
	for _, line := range strings.Split(strings.TrimRight(string(s.keys.Scrub(data)), "\r\n"), "\n") {
		if line != "" {
			lines = append(lines, strings.TrimSuffix(line, "\r"))
		}
	}
	if len(lines) > 2000 {
		lines = lines[len(lines)-2000:]
	}
	return lines
}

// SetStallTimeout changes the no-progress warning time at once. Zero restores
// the default; a negative value turns the watchdog off.
func (s *Supervisor) SetStallTimeout(d time.Duration) {
	if d == 0 {
		d = DefaultStallTimeout
	}
	s.stallNano.Store(int64(d))
}
func (s *Supervisor) StallTimeout() time.Duration { return time.Duration(s.stallNano.Load()) }

// SetModelSelection tells status which model the user selected, so readiness is
// reported for that selection (engine job E20, item 12). A manifest wins over a
// model file; neither means plain status. A change restarts the status stream.
func (s *Supervisor) SetModelSelection(manifest, model string) {
	var args []string
	switch {
	case manifest != "":
		args = []string{"--manifest", manifest}
	case model != "":
		args = []string{"--model", model}
	}
	if previous := s.selection.Load(); previous == nil && len(args) == 0 || previous != nil && slices.Equal(*previous, args) {
		return
	}
	s.selection.Store(&args)
	select {
	case s.visibility <- struct{}{}:
	default:
	}
}

// selectionArgs are the status arguments of the selected model.
func (s *Supervisor) selectionArgs() []string {
	if p := s.selection.Load(); p != nil {
		return append([]string(nil), (*p)...)
	}
	return nil
}

// Registry returns the engine switch registry from `serve --list-switches
// --json`, or nil until the CLI has delivered it.
func (s *Supervisor) Registry() *Switches { return s.registry.Load() }

// SetVisible changes View sampling, retaining the CLI's own stream cadence.
// Hidden supervision suspends the stream after HiddenStatusTimeout.
func (s *Supervisor) SetVisible(visible bool) {
	interval := 10 * time.Second
	if visible {
		interval = s.config.PollInterval
	}
	s.pollNanos.Store(int64(interval))
	s.visible.Store(visible)
	select {
	case s.visibility <- struct{}{}:
	default:
	}
}
func (s *Supervisor) PollInterval() time.Duration { return time.Duration(s.pollNanos.Load()) }
func (s *Supervisor) Snapshot() Snapshot {
	v := s.published.Load()
	copy := copySnapshot(*v)
	copy.StatusFeed = statusFreshness(copy, time.Now())
	if copy.StatusFeed.State == StatusNotUpdating || copy.StatusFeed.State == StatusPaused {
		copy.Stale = true
	}
	if gpu := s.gpuLatest.Load(); gpu != nil {
		g := copyGPUStats(*gpu)
		copy.GPU = &g
	}
	return copy
}

// pollGPU samples the GPU source at the status interval (at least 1 s). The
// latest reading is merged into every Snapshot; a failed poll clears it.
func (s *Supervisor) pollGPU(src GPUSource) {
	defer src.Close()
	for {
		s.gpuLatest.Store(safeGPUSample(src))
		d := s.PollInterval()
		if d < time.Second {
			d = time.Second
		}
		t := time.NewTimer(d)
		select {
		case <-s.ctx.Done():
			t.Stop()
			return
		case <-t.C:
		}
	}
}

func safeGPUSample(src GPUSource) (out *GPUStats) {
	defer func() {
		if recover() != nil {
			out = nil
		}
	}()
	if g := src.Sample(); !g.Empty() {
		return g
	}
	return nil
}
func (s *Supervisor) publish(v Snapshot) {
	if s.logs.Err() != nil {
		v.LogWarning = "Local logs could not be written."
	}
	copy := copySnapshot(v)
	s.published.Store(&copy)
}

// Actions enqueue work and return immediately. Completion is delivered on a
// buffered channel; callers can select on it without blocking a window thread.
func (s *Supervisor) submit(r request) <-chan ResultReply {
	r.reply = make(chan ResultReply, 1)
	s.submitMu.Lock()
	defer s.submitMu.Unlock()
	if s.closed {
		r.reply <- ResultReply{errors.New("supervisor closed")}
		return r.reply
	}
	select {
	case <-s.finished:
		r.reply <- ResultReply{errors.New("supervisor closed")}
	case s.commands <- r:
	default:
		r.reply <- ResultReply{errors.New("supervisor is busy")}
	}
	return r.reply
}
func (s *Supervisor) Start(args ...string) <-chan ResultReply {
	return s.submit(request{action: "start", args: args})
}
func (s *Supervisor) Prepare(args ...string) <-chan ResultReply {
	return s.submit(request{action: "prepare", args: args})
}
func (s *Supervisor) Stop() <-chan ResultReply { return s.submit(request{action: "stop"}) }
func (s *Supervisor) Restart(args ...string) <-chan ResultReply {
	return s.submit(request{action: "restart", args: args})
}
func (s *Supervisor) Exit() <-chan ResultReply {
	select {
	case <-s.finished:
		r := make(chan ResultReply, 1)
		r <- ResultReply{}
		return r
	default:
		// Bootstrap queues ordinary actions until discovery finishes. Stop a
		// slow discovery first so Exit can reach that queue and drain cleanly.
		if !s.Snapshot().Initialized {
			s.cancel()
		}
		return s.submit(request{action: "exit"})
	}
}
func (s *Supervisor) Discover() <-chan ResultReply { return s.submit(request{action: "discover"}) }
func (s *Supervisor) SelectInstance(id string) <-chan ResultReply {
	return s.submit(request{action: "select", args: []string{id}})
}
func (s *Supervisor) Manage() <-chan ResultReply       { return s.submit(request{action: "manage"}) }
func (s *Supervisor) DismissError() <-chan ResultReply { return s.submit(request{action: "dismiss"}) }
func (s *Supervisor) Consent(approve bool) <-chan ResultReply {
	action := "decline"
	if approve {
		action = "approve"
	}
	return s.submit(request{action: action})
}

// Diagnose runs a bounded, owned CLI helper without changing the server state.
func (s *Supervisor) Diagnose(args ...string) <-chan DiagnosticReply {
	reply := make(chan DiagnosticReply, 1)
	go func() {
		defer ReportPanic()
		if _, err := s.query(s.ctx, nil); err != nil {
			reply <- DiagnosticReply{Err: err}
			return
		}
		e, err := s.capture(s.ctx, append([]string{"diagnose", "--json"}, args...))
		if err == nil && e.Type != "diagnose" && e.Type != "diagnose_artifact" && e.Type != "support_bundle" {
			err = protocol("invalid diagnose result")
		}
		reply <- DiagnosticReply{e, err}
	}()
	return reply
}
func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}
func (s *Supervisor) emit(u update) {
	select {
	case s.updates <- u:
	case <-s.ctx.Done():
	case <-s.finished:
	}
}

// preparedFor says whether a start can proceed. The status reports the default
// package only; a prepared catalogue model or manifest that the user selected
// does not need it, only the engine files.
func (s *Supervisor) preparedFor(st *Status) bool {
	if st.Prepared {
		return true
	}
	if !st.Readiness.Runtime || s.config.SettingsPath == "" {
		return false
	}
	cur, _ := LoadSettings(s.config.SettingsPath)
	return cur.ModelPath != "" || cur.ManifestPath != ""
}

func validateArgs(args []string) error {
	for _, a := range args {
		if a == "--api-key" || strings.HasPrefix(a, "--api-key=") {
			return fmt.Errorf("supervisor owns the API key argument")
		}
		if a == "--yes" || a == "--host" || a == "--port" || a == "--json" || a == "--jsonl" || a == "--instance-id" || regexp.MustCompile(`^--(host|port|instance-id|json|jsonl|yes)=`).MatchString(a) {
			return fmt.Errorf("supervisor owns the lifecycle argument %s", a)
		}
	}
	return nil
}
func (s *Supervisor) run(v Snapshot) {
	var op *operation
	var restart []string
	var restarting bool
	// Automatic restart after status 75 (GPU lost): the arguments and saved
	// setup of the last start, the pending timer, and when the last automatic
	// restart began.
	var launchArgs []string
	var launchSettings *Settings
	var pin *Settings
	var autoAt time.Time
	var autoTimer *time.Timer
	var autoC <-chan time.Time
	cancelAuto := func() {
		if autoTimer != nil {
			autoTimer.Stop()
		}
		autoTimer, autoC = nil, nil
	}
	defer cancelAuto()
	var exitReply chan ResultReply
	var discoverReply chan ResultReply
	var externalReply chan ResultReply
	var samp sampler
	var liveSlots liveSlotSampler
	var externalWatchID string
	var externalWatchCancel context.CancelFunc
	streamControl := make(chan statusTarget, 1)
	s.helpers.Add(1)
	RunProtected(func() { s.watchStatuses(streamControl) })
	var streamTarget statusTarget
	var streamPolicyState string
	lastStatusPayloadAt := time.Now()
	var hidden bool
	var hiddenExpired bool
	hiddenTimer := time.NewTimer(time.Hour)
	if !hiddenTimer.Stop() {
		<-hiddenTimer.C
	}
	defer hiddenTimer.Stop()
	var hiddenDeadline <-chan time.Time
	v.StatusStaleTimeout = s.config.StatusStaleTimeout
	// Reconnect precedes all user actions, including opt-in autostart. Actions
	// stay queued while the bounded bootstrap handshake runs on a worker.
	var commandSource <-chan request
	discoverReply = make(chan ResultReply, 1)
	initialGen := v.Generation
	go func() {
		defer ReportPanic()
		notice := time.AfterFunc(s.config.FirstCallNotice, func() {
			s.emit(update{initialGen, "startup-wait", nil})
		})
		st, err := s.initialDiscoverWithSlow(func() {
			s.emit(update{initialGen, "startup-slow", nil})
		})
		notice.Stop()
		// Exit cancels discovery to unblock bootstrap. Its completion must
		// still reach the loop so queued Exit can be handled.
		if err != nil {
			s.updates <- update{initialGen, "discovery", err}
		} else {
			s.updates <- update{initialGen, "discovery", st}
		}
	}()
	// applySample feeds one status of one instance to the rates and the stall
	// watchdog. A status without identity means no instance: clear both.
	applySample := func(st *Status, at time.Time) {
		liveSlots.observe(&v, st, v.Generation, at)
		if st == nil || st.Identity == nil {
			v.Rates, v.Stall, v.CounterDelta, v.CountersReset = nil, nil, nil, true
			samp = sampler{}
			return
		}
		rates, delta, reset, stall := samp.observe(st.Identity.ID, st, at, s.StallTimeout())
		v.Rates, v.CounterDelta, v.CountersReset, v.Stall = &rates, delta, reset, stall
		// The window and the tray read one line for the warning: the label.
		if stall != nil && (v.Label == "" || isStallLabel(v.Label)) {
			v.Label = stall.Message
		} else if stall == nil && isStallLabel(v.Label) {
			v.Label = ""
		}
	}
	clearSample := func() {
		if isStallLabel(v.Label) {
			v.Label = ""
		}
		v.Rates, v.Stall, v.CounterDelta, v.CountersReset = nil, nil, nil, true
		samp = sampler{}
		liveSlots = liveSlotSampler{}
		v.LiveSlots, v.LiveSamples = nil, nil
		v.LiveSampleTime = ""
	}
	persist := func() {
		if s.config.SettingsPath != "" {
			settings, _ := LoadSettings(s.config.SettingsPath)
			settings.LastError = v.LastError
			settings.lastErrorVersion = s.config.ProductVersion
			if err := SaveSettings(s.config.SettingsPath, settings); err != nil {
				v.Label = "Settings save failed."
			}
		}
	}
	clearError := func() {
		if v.LastError != nil {
			v.LastError = nil
			v.Label = ""
			persist()
		}
	}
	// Previous readiness detects edges only; the View still uses CLI status.
	// The initial sample must not clear a retained same-version session error.
	var readinessKnown, runtimeReady, modelReady bool
	runtimeBecameReady := func() {
		if !runtimeReady {
			runtimeReady = true
			s.emitStream(StreamEvent{Kind: runtimeReadyStream})
		}
	}
	observeReadiness := func(st *Status) {
		if readinessKnown && ((!runtimeReady && st.Readiness.Runtime) || (!modelReady && st.Readiness.Model)) {
			clearError()
		}
		if st.Readiness.Runtime {
			runtimeBecameReady()
		} else {
			runtimeReady = false
		}
		modelReady, readinessKnown = st.Readiness.Model, true
	}
	launch := func(action string, args []string) {
		v.Generation++
		v.Forced = false
		v.Stale = true
		v.Instance = nil
		v.Approval = nil
		v.PlanParts = nil
		v.Preflight = nil
		v.Preflights = nil
		clearSample()
		v.OperationID = ""
		v.StopPhase = ""
		if action == "start" {
			launchArgs, launchSettings = append([]string(nil), args...), nil
		}
		v.ServerWarnings = nil
		thisPin := pin
		pin = nil
		if thisPin == nil {
			v.AutoRestarted = false // a start the user made
		}
		op = &operation{v.Generation, make(chan struct{}, 1), make(chan []byte, 8), make(chan struct{}), thisPin}
		RunProtected(func() { s.operate(op, action, args) })
	}
	stop := func() {
		if op != nil {
			select {
			case op.stop <- struct{}{}:
			default:
			}
		}
	}
	finish := func() {
		s.publish(v)
		s.submitMu.Lock()
		s.closed = true
		for len(s.commands) > 0 {
			r := <-s.commands
			r.reply <- ResultReply{errors.New("supervisor closed")}
		}
		s.submitMu.Unlock()
		s.helperMu.Lock()
		s.closing = true
		s.cancel()
		s.helperMu.Unlock()
		s.helpers.Wait()
		if externalReply != nil {
			externalReply <- ResultReply{errors.New("supervisor closed")}
		}
		if discoverReply != nil {
			discoverReply <- ResultReply{errors.New("supervisor closed")}
		}
		s.logs.Close()
		close(s.finished)
		if exitReply != nil {
			exitReply <- ResultReply{}
		}
	}
	for {
		// Reconcile all lifecycle/selection/visibility changes through one owner.
		want := statusTarget{selection: strings.Join(s.selectionArgs(), "\x00"), gen: v.Generation, enabled: v.Initialized && !hiddenExpired && discoverReply == nil && externalReply == nil && (op == nil || v.State == Ready)}
		if v.Instance != nil {
			i := *v.Instance
			want.instance = &i
		}
		state := StatusInactive
		if hiddenExpired {
			state = StatusPaused
		}
		if want.enabled && (v.Status == nil || !v.Status.Capabilities.StatusWatch) {
			want.enabled = false
			state = StatusUnsupported
		}
		if !streamTarget.matches(want) || state != streamPolicyState {
			streamPolicyState = state
			want.serial = streamTarget.serial + 1
			streamTarget = want
			select {
			case <-streamControl:
			default:
			}
			streamControl <- want
			v.StatusRecordAt = time.Now()
			v.StatusFeed = StatusFeed{State: state}
			if want.enabled {
				v.StatusFeed.State = StatusWaiting
			}
			if state == StatusUnsupported {
				v.StatusFeed.Error = codedAction("protocol_mismatch", "CLI status watching is unavailable.").Error
			}
			s.publish(v)
		}
		select {
		case <-s.visibility:
			if nextHidden := !s.visible.Load(); nextHidden != hidden {
				hidden = nextHidden
				hiddenTimer.Stop()
				hiddenDeadline = nil
				hiddenExpired = false
				if hidden {
					hiddenTimer.Reset(s.config.HiddenStatusTimeout)
					hiddenDeadline = hiddenTimer.C
				}
			}
		case <-hiddenDeadline:
			hiddenDeadline = nil
			hiddenExpired = true
		case <-autoC:
			autoTimer, autoC = nil, nil
			// Nothing else has started or stopped the server since the exit.
			if op == nil && exitReply == nil && v.State == Failed && v.LastError != nil && v.LastError.Code == "device_lost" && launchSettings != nil {
				if err := v.transition("start"); err == nil {
					autoAt = time.Now()
					pin = launchSettings
					launch("start", launchArgs)
					v.AutoRestarted = true
				}
			}
		case r := <-commandSource:
			if r.action != "discover" {
				cancelAuto() // any action by the user ends a pending restart
			}
			var err error
			deferred := false
			switch r.action {
			case "start", "prepare":
				if op != nil || discoverReply != nil {
					err = appError("busy", "operation is active")
					break
				}
				if err = validateArgs(r.args); err != nil {
					break
				}
				if err = v.transition(r.action); err == nil {
					launch(r.action, r.args)
				}
			case "stop", "restart", "exit":
				if r.action == "restart" {
					if err = validateArgs(r.args); err != nil {
						break
					}
				}
				if v.State == AttachedExternal {
					if r.action == "exit" {
						exitReply = r.reply
						finish()
						return
					}
					if !v.Managed {
						err = appError("control_access", "Select Manage this server before Stop")
						break
					}
					if externalReply != nil {
						err = appError("busy", "stop is active")
						break
					}
					if r.action == "restart" {
						restart = append([]string(nil), r.args...)
						restarting = true
					}
					externalReply = r.reply
					deferred = true
					v.StopPhase = "request"
					i := *v.Instance
					gen := v.Generation
					s.helpers.Add(1)
					go func() {
						defer ReportPanic()
						defer s.helpers.Done()
						err := s.stopExternal(i)
						s.emit(update{gen, "external-stop", err})
					}()
				} else if op != nil {
					if v.State == Stopping {
						err = appError("busy", "stop is active")
						break
					}
					action := "stop"
					if v.State == Preparing {
						action = "cancel"
					}
					if v.State != Failed {
						err = v.transition(action)
					}
					if err != nil {
						break
					}
					v.StopPhase = "request"
					stop()
					if r.action == "restart" {
						restart = append([]string(nil), r.args...)
						restarting = true
					}
					if r.action == "exit" {
						exitReply = r.reply
						deferred = true
					}
				} else if r.action == "exit" {
					exitReply = r.reply
					finish()
					return
				} else if r.action != "stop" {
					err = appError("control_access", "no owned operation is active")
				}
			case "dismiss":
				v.LastError = nil
				v.Label = ""
				persist()
			case "approve", "decline":
				if op == nil || v.State != Preparing || v.Approval == nil {
					err = appError("internal_error", "no consent request is active")
					break
				}
				if r.action == "approve" {
					short := false
					for _, part := range v.PlanParts {
						for _, volume := range part.Volumes {
							short = short || volume.FreeBytes < volume.RequiredBytes
						}
					}
					if short {
						err = appError("insufficient_space", "The preparation plan requires more drive space than is free.")
						v.Approval = nil
						v.fail("insufficient_space", "supervisor", false)
						persist()
						stop()
						break
					}
				}
				b := ApprovalLine(r.action, v.OperationID, v.Approval.Digest)
				select {
				case op.input <- b:
					v.Approval = nil
					if r.action == "decline" {
						// Declining ends the run like a cancel: nothing failed.
						_ = v.transition("cancel")
					}
				default:
					err = appError("busy", "input is busy")
				}
			case "discover":
				if op != nil || discoverReply != nil {
					err = appError("busy", "operation is active")
					break
				}
				discoverReply = r.reply
				deferred = true
				gen := v.Generation
				go func() {
					defer ReportPanic()
					st, err := s.discover()
					if err != nil {
						s.emit(update{gen, "discovery", err})
					} else {
						s.emit(update{gen, "discovery", st})
					}
				}()
			case "select":
				if op != nil || discoverReply != nil {
					err = appError("busy", "operation is active")
					break
				}
				var selected *Instance
				for _, i := range v.Discovered {
					if i.ID == r.args[0] {
						copy := i
						selected = &copy
					}
				}
				if selected == nil {
					err = appError("control_access", "instance is not in discovery results")
					break
				}
				if err = VerifyIdentity(*selected, s.config.Exe); err != nil {
					break
				}
				discoverReply = r.reply
				deferred = true
				gen := v.Generation
				copy := *selected
				go func() {
					defer ReportPanic()
					st, err := s.query(s.ctx, &copy)
					if err == nil {
						found := false
						for _, i := range st.Instances {
							if sameInstance(i, copy) {
								found = true
								st.Instances = []Instance{i}
								break
							}
						}
						if !found {
							err = protocol("selected instance handshake did not agree")
						}
					}
					if err != nil {
						s.emit(update{gen, "discovery", err})
					} else {
						s.emit(update{gen, "discovery", st})
					}
				}()
			case "manage":
				if v.State != AttachedExternal {
					err = appError("control_access", "no external instance")
				} else if !v.ManagementSupported {
					err = appError("external_read_only", safeRemedy("unsupported"))
				} else if err = VerifyIdentity(*v.Instance, s.config.Exe); err == nil {
					v.Managed = true
				}
			default:
				err = appError("internal_error", "Unknown supervisor action: "+r.action)
			}
			s.publish(v)
			if !deferred {
				r.reply <- ResultReply{err}
			}
		case u := <-s.updates:
			if u.kind == "registry" {
				v.RegistryLoaded = true
				s.publish(v)
				continue
			}
			if u.gen != v.Generation {
				continue
			}
			switch u.kind {
			case "status-stream":
				sample := u.value.(statusUpdate)
				if sample.serial != streamTarget.serial || !streamTarget.enabled {
					continue
				}
				v.StatusFeed.State = sample.state
				v.StatusFeed.Error = nil
				if sample.err != nil {
					v.StatusFeed.Error = &CLIError{Code: errorCode(sample.err), Component: "status", Stage: "watch", Retryable: true, Remedy: sample.err.Error()}
					var cli *CommandError
					if errors.As(sample.err, &cli) {
						e := *cli.Event.Value.(*CLIError)
						v.StatusFeed.Error = &e
					}
					v.Stale = true
				} else {
					v.StatusRecordAt = sample.at
					v.StatusFeed.SampleTime = sample.sampleTime
					if sample.status != nil {
						applySample(sample.status, sample.at)
					}
					if sample.status != nil && (s.visible.Load() || time.Since(lastStatusPayloadAt) >= s.PollInterval()) {
						lastStatusPayloadAt = sample.at
						v.Status = sample.status
						// A later successful status supersedes a transient wire
						// failure. Do not keep an obsolete protocol panel stacked
						// beside the current setup's memory verdict.
						if v.LastError != nil && v.LastError.Code == "protocol_mismatch" && sample.status.LastError == nil && sample.at.After(v.LastError.Time) {
							clearError()
							if v.State == Failed && v.Instance == nil {
								v.State = Stopped
							}
						}
						observeReadiness(v.Status)
						v.Prepared = s.preparedFor(sample.status)
						if v.Prepared && v.State == NotPrepared && v.Instance == nil {
							_ = v.transition("prepared") // the user selected a prepared model
						}
						v.Stale = false
						if v.Instance != nil {
							for _, i := range sample.status.Instances {
								if sameInstance(i, *v.Instance) {
									v.Instance = &i
									break
								}
							}
						}
					}
				}
			case "identity":
				i := u.value.(Instance)
				v.Instance = &i
			case "switch_overrides":
				v.RunningSwitchOverrides = u.value.(map[string]string)
			case "event":
				e := u.value.(Event)
				if e.Value == nil {
					v.Diagnostic = "CLI sent an additive event. State did not change."
				}
				v.OperationID = e.OperationID
				switch value := e.Value.(type) {
				case *Plan:
					if v.Approval != nil && value.Digest != v.Approval.Digest {
						v.Approval = nil
					}
					v.PlanParts = append(v.PlanParts, *value)
				case *Preflight:
					if len(v.Preflights) >= 2000 {
						v.fail("protocol_mismatch", "supervisor", false)
						stop()
						break
					}
					v.Preflights = append(v.Preflights, *value)
					v.Preflight = value
					// A preflight opens a new plan of its component; plan parts of
					// an earlier round would pile up beside it.
					kept := v.PlanParts[:0:0]
					for _, part := range v.PlanParts {
						if part.Component != value.Component {
							kept = append(kept, part)
						}
					}
					v.PlanParts = kept
				case *Approval:
					v.Approval = value
				case *Progress:
					value.Overall = jobProgress(partJobFiles(v.PlanParts), value.Component, value.FileID, value.Phase, value.Done)
					v.Progress = value
					p := copyProgress(*value)
					s.emitStream(StreamEvent{Kind: PrepareProgress, Progress: &p})
				}
			case "ready":
				i := u.value.(Instance)
				v.Instance = &i
				if v.State == Starting {
					_ = v.transition("verified-ready")
					clearError()
				}
				v.Stale = false
			case "model-ready":
				clearError()
			case "sample":
				v.Stale = false
			case "status":
				v.Status = u.value.(*Status)
				observeReadiness(v.Status)
				applySample(v.Status, time.Now())
			case "external-exit":
				if v.State == AttachedExternal && v.Instance != nil && v.Instance.ID == u.value.(string) {
					clearSample()
					v.Instance = nil
					v.Managed = false
					v.Stale = true
					_ = v.transition("external-exit")
				}
			case "stale":
				v.Stale = true
			case "stop-phase":
				v.StopPhase = u.value.(string)
			case "failure":
				f := u.value.(Failure)
				if f.Code == "unsupported" {
					f.Code = "protocol_mismatch"
				}
				v.fail(f.Code, f.Source, f.Forced)
				if f.Source == "cli" {
					v.LastError.Remedy = f.Remedy
					v.LastError.Detail = f.Detail
					v.LastError.Memory = f.Memory
					v.LastError.Component = f.Component
					v.LastError.Stage = f.Stage
					v.LastError.Switch = f.Switch
				}
				persist()
			case "server_warning":
				w := u.value.(ServerWarning)
				known := false
				for _, old := range v.ServerWarnings {
					known = known || old.Code == w.Code
				}
				if !known && len(v.ServerWarnings) < 5 {
					v.ServerWarnings = append(v.ServerWarnings, w)
				}
			case "launch_settings":
				c := u.value.(Settings)
				launchSettings = &c
			case "complete":
				result := u.value.(completion)
				op = nil
				v.AutoRestarted = false // the run that the notice described has ended
				clearSample()
				v.Approval = nil
				v.StopPhase = ""
				v.Instance = nil
				if result.forced {
					if v.State == Failed && v.LastError != nil {
						copy := *v.LastError
						copy.Forced = true
						v.LastError = &copy
						v.Forced = true
					} else {
						v.fail("stop_timeout", "supervisor", true)
					}
					persist()
				} else if result.err != nil {
					if v.State != Failed || v.LastError == nil {
						v.fail(errorCode(result.err), "supervisor", false)
						var admission *actionUnavailableError
						if errors.As(result.err, &admission) {
							v.LastError.Remedy = admission.Reason
						}
						var cli *CommandError
						if errors.As(result.err, &cli) {
							c := cli.Event.Value.(*CLIError)
							v.LastError.Source = "cli"
							v.LastError.Remedy = c.Remedy
							v.LastError.Detail = c.Detail
							v.LastError.Memory = memoryShortfall(c)
							v.LastError.Component = c.Component
							v.LastError.Stage = c.Stage
							v.LastError.Switch = c.Switch
						}
					}
					v.Label = result.err.Error()
					persist()
				} else if v.State == Preparing {
					clearError()
					for _, part := range v.PlanParts {
						if part.Component == "runtime" {
							runtimeBecameReady()
							break
						}
					}
					v.Prepared = true
					_ = v.transition("prepared")
				} else if v.State == Stopping {
					_ = v.transition("clean-exit")
				} else if v.State != Failed {
					v.fail("internal_error", "supervisor", false)
					persist()
				}
				if exitReply != nil {
					finish()
					return
				}
				if restarting {
					restarting = false
					if v.State == Stopped || v.State == Failed {
						_ = v.transition("start")
						launch("start", restart)
					}
				}
				// Status 75: the engine lost the GPU. Restart once after a short
				// delay, never after the user's own Stop, never twice in a window.
				if v.State == Failed && exitReply == nil && result.exit == 75 && !result.userStop && !result.forced && launchSettings != nil && v.LastError != nil && v.LastError.Code == "device_lost" {
					if autoAt.IsZero() || time.Since(autoAt) >= s.config.AutoRestartWindow {
						cancelAuto()
						autoTimer = time.NewTimer(s.config.AutoRestartDelay)
						autoC = autoTimer.C
					} else {
						v.LastError.Again = true // the GPU was lost again after the app's restart
					}
				}
			case "discovery":
				var err error
				if e, ok := u.value.(error); ok {
					err = e
					v.Label = e.Error()
					if v.State == AttachedExternal {
						v.Stale = true
					} else {
						v.fail(errorCode(e), "supervisor", false)
						persist()
					}
				} else {
					st := u.value.(*Status)
					v.Status = st
					observeReadiness(st)
					applySample(st, time.Now())
					if st.Capabilities.SwitchRegistry {
						registryGen := v.Generation
						s.regOnce.Do(func() {
							go func() {
								defer ReportPanic()
								for attempt := 0; s.ctx.Err() == nil; attempt++ {
									if s.LoadRegistry() == nil {
										s.emit(update{registryGen, "registry", nil})
										return
									}
									if !s.retryDelay(attempt) {
										return
									}
								}
							}()
						})
					}
					v.Discovered = st.Instances
					v.Prepared = s.preparedFor(st)
					if len(st.Instances) > 1 {
						err = appError("control_access", "Select one server instance")
					} else if len(st.Instances) == 1 {
						i := st.Instances[0]
						if err = VerifyIdentity(i, s.config.Exe); err == nil {
							v.Instance = &i
							_ = v.transition("attach")
							v.Label = "Started outside this window"
							v.Managed = false
							v.ManagementSupported = st.Capabilities.InstanceControl
							v.ManagementMessage = ""
							if !v.ManagementSupported {
								v.ManagementMessage = safeRemedy("unsupported")
							}
							v.Stale = false
							if externalWatchID != i.ID {
								if externalWatchCancel != nil {
									externalWatchCancel()
								}
								externalWatchID = i.ID
								copy := i
								gen := v.Generation
								ctx, cancel := context.WithCancel(s.ctx)
								externalWatchCancel = cancel
								s.helpers.Add(1)
								go func() { defer ReportPanic(); defer s.helpers.Done(); s.watchExternal(ctx, copy, gen) }()
							}
						}
					} else if s.preparedFor(st) {
						if v.State == NotPrepared {
							_ = v.transition("prepared")
						}
					} else if v.State == Stopped || v.State == Failed {
						_ = v.transition("unprepared")
					}
				}
				if err != nil {
					v.Label = err.Error()
				}
				discoverReply <- ResultReply{err}
				discoverReply = nil
				v.Initialized = true
				v.StartupPhase = ""
				v.StartupWaiting = false
				commandSource = s.commands
			case "startup-wait":
				if !v.Initialized && !v.StartupWaiting {
					v.StartupWaiting = true
					s.publish(v)
					s.appendLog([]byte("startup_wait: the first CLI call has waited more than the notice time"), true)
				}
			case "startup-slow":
				if !v.Initialized {
					v.StartupPhase = "startup_slow"
					s.publish(v)
					s.appendLog([]byte("startup_slow: first status request timed out; retrying discovery"), true)
				}
			case "external-stop":
				err, _ := u.value.(error)
				if err == nil {
					clearSample()
					v.Instance = nil
					v.Managed = false
					_ = v.transition("external-exit")
				} else {
					v.Label = err.Error()
					v.Stale = true
				}
				v.StopPhase = ""
				externalReply <- ResultReply{err}
				externalReply = nil
				if restarting {
					restarting = false
					if err == nil {
						_ = v.transition("start")
						launch("start", restart)
					}
				}
			}
			s.publish(v)
		}
	}
}

// statusDLLNotFound is the exit status of a program that Windows could not
// start because a DLL it needs is missing (STATUS_DLL_NOT_FOUND). For bubo.exe
// that is the Microsoft Visual C++ runtime; the process prints nothing.
const statusDLLNotFound = 0xC0000135

type vcRuntimeMissingError struct{}

func (*vcRuntimeMissingError) Error() string     { return "the Visual C++ runtime is missing" }
func (*vcRuntimeMissingError) ErrorCode() string { return "vcredist_missing" }

// cliExitError names a CLI that ended with no record: usually "CLI exit N",
// but exit 0xC0000135 is the missing Visual C++ runtime.
func cliExitError(code uint32) error {
	if code == statusDLLNotFound {
		return &vcRuntimeMissingError{}
	}
	return fmt.Errorf("CLI exit %d", code)
}

func errorCode(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return "query_timeout"
	}
	var c *CommandError
	if errors.As(err, &c) {
		code := c.Event.Value.(*CLIError).Code
		if code == "unsupported" {
			return "protocol_mismatch"
		}
		return code
	}
	var p *ProtocolError
	if errors.As(err, &p) {
		return "protocol_mismatch"
	}
	var coded interface{ ErrorCode() string }
	if errors.As(err, &coded) {
		return coded.ErrorCode()
	}
	return "internal_error"
}

type completion struct {
	err    error
	forced bool
	// exit is the process exit status; userStop is true when the user's Stop
	// was already running when the process ended.
	exit     uint32
	userStop bool
}

func (s *Supervisor) watchExternal(ctx context.Context, i Instance, gen uint64) {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, i.PID)
	if err != nil {
		return
	}
	defer windows.CloseHandle(h)
	created, err := creationTime(h)
	if err != nil || created != i.CreationTime {
		return
	}
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			state, err := windows.WaitForSingleObject(h, 0)
			if err != nil {
				return
			}
			if state == windows.WAIT_OBJECT_0 {
				s.emit(update{gen, "external-exit", i.ID})
				return
			}
		}
	}
}

// query helpers also have their own Job Object, so a stuck helper cannot escape.
func (s *Supervisor) capture(ctx context.Context, args []string) (Event, error) {
	events, err := s.captureEvents(ctx, args, 1)
	if len(events) == 0 {
		return Event{}, err
	}
	return events[0], err
}

// captureEvents runs one bounded helper and returns its records in order. At
// most max records are accepted; one more is a Protocol error. An error event
// is returned with a CommandError that holds the raw exit code.
func (s *Supervisor) captureEvents(ctx context.Context, args []string, max int) ([]Event, error) {
	limit := s.config.QueryTimeout
	if len(args) > 0 && args[0] == "stop" {
		// Stop waits for the process to exit; its own wait is the deadline.
		limit = s.config.StopTimeout + s.config.QueryTimeout
	}
	return s.captureEventsTimeout(ctx, args, max, limit)
}

func (s *Supervisor) captureEventsTimeout(ctx context.Context, args []string, max int, limit time.Duration) ([]Event, error) {
	// Until the CLI has answered once, a call may wait for the first-run scan
	// of the program files: it gets the cold-start limit.
	if !s.cliAnswered.Load() && limit < s.config.ColdStartTimeout {
		limit = s.config.ColdStartTimeout
	}
	events, err := s.captureEventsLimit(ctx, args, max, limit)
	if err == nil {
		s.cliAnswered.Store(true)
	}
	return events, err
}

func (s *Supervisor) captureEventsLimit(ctx context.Context, args []string, max int, limit time.Duration) ([]Event, error) {
	s.helperMu.Lock()
	if s.closing {
		s.helperMu.Unlock()
		return nil, errors.New("supervisor closed")
	}
	s.helpers.Add(1)
	s.helperMu.Unlock()
	defer s.helpers.Done()
	ctx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	lines := make(chan Event, max+1)
	bad := make(chan error, 1)
	p, err := launchOwnedEnv(s.config.Exe, args, childEnvironment(os.Environ(), "", s.config.RuntimeDir), s.config.DrainTimeout, func(b []byte, complete bool) {
		e, err := ParseEvent(s.keys.Scrub(b))
		if !complete {
			err = protocol("truncated stdout record")
		}
		if err != nil {
			select {
			case bad <- err:
			default:
			}
			return
		}
		select {
		case lines <- e:
		default:
			select {
			case bad <- protocol("one-shot emitted extra records"):
			default:
			}
		}
	}, func(b []byte, complete bool) { s.appendLog(s.keys.Scrub(b), complete) }, nil)
	if err != nil {
		return nil, appError("cli_unavailable", err.Error())
	}
	defer s.trackProcess(p)()
	select {
	case result := <-p.Done:
		select {
		case err := <-bad:
			return nil, err
		default:
		}
		var events []Event
		for len(lines) > 0 {
			events = append(events, <-lines)
		}
		if len(events) > max {
			return nil, protocol("multiple one-shot records")
		}
		if len(events) > 0 && events[len(events)-1].Type == "error" {
			last := events[len(events)-1]
			return events, &CommandError{last, result.ExitCode}
		}
		if result.Err != nil {
			return nil, result.Err
		}
		if result.DrainTimedOut {
			return nil, protocol("helper pipes did not reach EOF")
		}
		if len(events) == 0 {
			if result.ExitCode != 0 {
				return nil, cliExitError(result.ExitCode)
			}
			return nil, protocol("missing one-shot record")
		}
		if result.ExitCode != 0 && !(events[0].Type == "diagnose" && result.ExitCode == 1 && events[0].Value.(*Diagnose).Status == "FAIL") {
			return nil, cliExitError(result.ExitCode)
		}
		return events, nil
	case <-ctx.Done():
		p.Force()
		<-p.Done
		return nil, ctx.Err()
	}
}
func (s *Supervisor) query(ctx context.Context, i *Instance) (*Status, error) {
	return s.queryTimeout(ctx, i, s.config.QueryTimeout)
}

func (s *Supervisor) queryTimeout(ctx context.Context, i *Instance, timeout time.Duration) (*Status, error) {
	args := []string{"status", "--json"}
	if i != nil {
		args = append(args, "--instance-id", i.ID)
	} else {
		args = append(args, s.selectionArgs()...)
	}
	events, err := s.captureEventsTimeout(ctx, args, 1, timeout)
	if err != nil {
		return nil, err
	}
	return statusRecord(events[0])
}

func (s *Supervisor) discover() (*Status, error) {
	return s.discoverTimeout(s.ctx, s.config.QueryTimeout)
}

func (s *Supervisor) initialDiscover() (*Status, error) {
	return s.initialDiscoverWithSlow(nil)
}

// A cold CLI may outlast several deadlines. Keep the checking View available
// until it answers or the supervisor closes; fail fast on non-timeout errors.
func (s *Supervisor) initialDiscoverWithSlow(onSlow func()) (*Status, error) {
	for attempt := 0; ; attempt++ {
		var st *Status
		ctx, cancel := context.WithTimeout(s.ctx, s.config.ColdStartTimeout)
		st, err := s.discoverTimeout(ctx, s.config.ColdStartTimeout)
		cancel()
		if err == nil || !errors.Is(err, context.DeadlineExceeded) || s.ctx.Err() != nil {
			return st, err
		}
		if attempt == 0 && onSlow != nil {
			onSlow()
		}
		if !s.retryDelay(attempt) {
			return nil, s.ctx.Err()
		}
	}
}

func (s *Supervisor) retryDelay(attempt int) bool {
	if attempt > 5 {
		attempt = 5
	}
	delay := s.config.ColdStartRetryBackoff * time.Duration(1<<attempt)
	if delay > 8*time.Second {
		delay = 8 * time.Second
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-s.ctx.Done():
		return false
	}
}

func (s *Supervisor) discoverTimeout(ctx context.Context, timeout time.Duration) (*Status, error) {
	if info, err := os.Stat(s.config.Exe); err != nil || info.IsDir() {
		return nil, &cliMissingError{path: s.config.Exe}
	}
	st, err := s.queryTimeout(ctx, nil, timeout)
	if err != nil {
		return nil, err
	}
	if len(st.Instances) == 1 && st.Capabilities.InstanceControl {
		i := st.Instances[0]
		if err = VerifyIdentity(i, s.config.Exe); err != nil {
			return nil, err
		}
		verified, err := s.queryTimeout(ctx, &i, timeout)
		if err != nil {
			return nil, err
		}
		if len(verified.Instances) != 1 || !sameInstance(i, verified.Instances[0]) {
			return nil, protocol("discovery identity handshake did not agree")
		}
		return verified, nil
	}
	return st, nil
}

type cliMissingError struct{ path string }

func (e *cliMissingError) Error() string   { return "Bubo CLI is missing or cannot run: " + e.path }
func (*cliMissingError) ErrorCode() string { return "cli_missing" }

func (s *Supervisor) operate(op *operation, action string, args []string) {
	defer close(op.done)
	send := func(kind string, val any) { s.emit(update{op.gen, kind, val}) }
	st, err := s.query(s.ctx, nil)
	if err != nil {
		send("complete", completion{err: err})
		return
	}
	if action == "start" && !st.Capabilities.InstanceControl {
		send("failure", Failure{Code: "protocol_mismatch", Source: "supervisor"})
		send("complete", completion{err: appError("protocol_mismatch", "instance control is missing")})
		return
	}
	if action == "prepare" && !st.Capabilities.PrepareJSONL {
		send("complete", completion{err: protocol("prepare capability is missing")})
		return
	}
	if len(st.Instances) > 0 {
		send("complete", completion{err: appError("external_read_only", "an external instance is active; discover before Start")})
		return
	}
	if action == "start" && !s.preparedFor(st) {
		send("complete", completion{err: appError("not_prepared", "package is not prepared")})
		return
	}
	select {
	case <-op.stop:
		send("complete", completion{})
		return
	default:
	}
	events := make(chan Event, 128)
	bad := make(chan error, 1)
	argv := []string{"prepare", "--json"}
	if action == "prepare" && !st.Readiness.Runtime {
		argv = []string{"prepare", "runtime", "--jsonl"}
	}
	bindHost, apiKey := "127.0.0.1", ""
	var overrideArgs []string
	var overrideEnv map[string]string
	if action == "start" {
		cur, _ := LoadSettings(s.config.SettingsPath)
		if op.pin != nil {
			cur = *op.pin // an automatic restart: the setup the user started with
		}
		cur = startSettingsForArguments(cur, args)
		if err := s.admitSavedStart(cur, st); err != nil {
			send("complete", completion{err: err})
			return
		}
		select {
		case <-op.stop:
			send("complete", completion{})
			return
		default:
		}
		send("launch_settings", cur)
		// Loopback (default): no key. Local network access: bind all
		// interfaces and pass the stored key through the child environment
		// (BUBO_API_KEY), never on the command line, so process listings and
		// command-line logs cannot show it. The GUI also scrubs the key from
		// CLI output.
		if s.config.SettingsPath != "" && cur.LanAccess {
			if testLoopbackOnly {
				send("failure", Failure{Code: "lan_unavailable", Source: "supervisor"})
				send("complete", completion{err: &lanUnavailableError{}})
				return
			}
			key, kerr := s.keys.Ensure()
			if kerr != nil {
				s.turnLANOff()
				send("complete", completion{err: kerr})
				return
			}
			bindHost, apiKey = "0.0.0.0", key
		}
		argv = append([]string{"serve"}, buildServeArguments(Settings{}, bindHost, s.config.Port, true)...)
		if len(cur.SwitchOverrides) > 0 {
			if s.Registry() == nil {
				if err := s.LoadRegistry(); err != nil {
					send("complete", completion{err: err})
					return
				}
			}
			var err error
			overrideArgs, overrideEnv, err = switchCommand(cur.SwitchOverrides, s.Registry())
			if err != nil {
				send("complete", completion{err: appError("setting_rejected", err.Error())})
				return
			}
		}
		send("switch_overrides", cur.SwitchOverrides)
	}
	argv = append(argv, args...)
	argv = append(argv, overrideArgs...)
	scrub := func(b []byte) []byte { return s.keys.Scrub(scrubKey(b, apiKey)) }
	env := switchEnvironment(childEnvironment(os.Environ(), apiKey, s.config.RuntimeDir), overrideEnv)
	p, err := launchOwnedEnv(s.config.Exe, argv, env, s.config.DrainTimeout, func(b []byte, complete bool) {
		e, err := ParseEvent(scrub(b))
		if !complete {
			err = protocol("truncated stdout record")
		}
		if err != nil {
			select {
			case bad <- err:
			default:
			}
			return
		}
		events <- e
	}, func(b []byte, complete bool) { s.appendLog(scrub(b), complete) }, nil)
	if err != nil {
		send("complete", completion{err: appError("cli_unavailable", err.Error())})
		return
	}
	defer s.trackProcess(p)()
	// The server creates its instance id and reports it in `starting`. Until
	// then only the process we launched (PID and creation time) is known.
	identity := Instance{PID: p.PID, CreationTime: p.CreationTime, ExePath: s.config.Exe}
	send("identity", identity)
	load := time.NewTimer(s.config.LoadTimeout)
	if action == "prepare" {
		load.Stop()
	}
	defer load.Stop()
	poll := time.NewTicker(s.config.PollInterval)
	var lastQuery time.Time
	defer poll.Stop()
	var stopTimer *time.Timer
	var stopDeadline <-chan time.Time
	var queryResult chan any
	var readyEvent *Event
	var terminalErr error
	var forced, stopping, cleanStopped, prepared, queryActive, verifiedReady, cancelConfirmed bool
	var seq uint64
	var wireOp string
	planBytes := 0
	stdoutEnded := p.StdoutEOF
	var pipeDeadline <-chan time.Time
	var pipeTimer *time.Timer
	var lastPrepareProgress time.Time
	var prepareDone uint64
	var preparePhase, prepareFile string
	defer func() {
		if pipeTimer != nil {
			pipeTimer.Stop()
		}
	}()
	// stoppedUnnamed: the user stopped a server that had not yet sent its first
	// lifecycle record. The process is ours (PID and creation time are known) and
	// serves nothing, so the job is ended at once and that is a clean stop.
	var stoppedUnnamed bool
	beginStop := func() {
		if stopping {
			return
		}
		stopping = true
		if action == "start" && identity.ID == "" && terminalErr == nil {
			stoppedUnnamed = true
			go func() {
				defer ReportPanic()
				_ = p.Force()
			}()
		}
		send("stop-phase", "request")
		stopTimer = time.NewTimer(s.config.StopTimeout)
		stopDeadline = stopTimer.C
		copy := identity
		operationID := wireOp
		go func() {
			defer ReportPanic()
			if action == "prepare" {
				b := CancelLine(operationID)
				_ = p.Input(b)
				p.CloseInput()
			} else if copy.ID != "" {
				if err := s.requestStop(copy); err != nil {
					s.logs.Append([]byte("graceful stop request failed: "+err.Error()), true)
				}
			}
			send("stop-phase", "await exit")
		}()
	}
	handleEvent := func(e Event) {
		if e.Type == "plan" {
			planBytes += len(e.Payload)
			if planBytes > 8<<20 {
				terminalErr = protocol("preparation plan exceeds 8 MiB")
				beginStop()
				return
			}
		}
		if wireOp == "" {
			wireOp = e.OperationID
		}
		if e.OperationID != wireOp || e.Seq <= seq {
			terminalErr = protocol("operation identity or sequence did not increase")
			beginStop()
			return
		}
		seq = e.Seq
		if action == "start" && (e.Type == "starting" || e.Type == "loading" || e.Type == "ready" || e.Type == "warning" || e.Type == "stopping" || e.Type == "stopped") {
			life, _ := e.Value.(*Lifecycle)
			wrong := life == nil || e.InstanceID == "" || e.PID != identity.PID || e.CreationTime != identity.CreationTime || life.Identity.PID != identity.PID || life.Identity.CreationTime != identity.CreationTime
			if !wrong && identity.ID != "" && e.InstanceID != identity.ID {
				wrong = true
			}
			if !wrong && !sameExePath(life.Identity.ExePath, identity.ExePath) {
				wrong = true
			}
			if wrong {
				terminalErr = protocol("lifecycle identity does not match owned process")
				send("failure", Failure{Code: "protocol_mismatch", Source: "supervisor"})
				beginStop()
				return
			}
			if e.Type == "warning" && life.Code != "" {
				send("server_warning", ServerWarning{Code: life.Code, Remedy: life.Remedy})
			}
			if identity.ID == "" {
				identity.ID = e.InstanceID
				// The first verified event: from here the CLI's own canonical path
				// is the identity's path.
				identity.ExePath = life.Identity.ExePath
				send("identity", identity)
			}
		}
		if e.Type == "ready" {
			life := e.Value.(*Lifecycle)
			if life.Endpoint == nil || life.Model == "" || life.Effective == nil || !matchesBind(life.Endpoint, bindHost, s.config.Port) {
				terminalErr = protocol("ready payload is incomplete")
				beginStop()
				return
			}
			identity.Endpoint = life.Endpoint
			copy := e
			readyEvent = &copy
		}
		if e.Type == "stopped" {
			cleanStopped = true
		}
		if r, ok := e.Value.(*Result); ok {
			prepared = r.Prepared && r.ExitCode == 0
		}
		if action == "prepare" {
			switch progress := e.Value.(type) {
			case *Progress:
				if lastPrepareProgress.IsZero() || progress.Done != prepareDone || progress.Phase != preparePhase || progress.FileID != prepareFile {
					lastPrepareProgress = time.Now()
					prepareDone, preparePhase, prepareFile = progress.Done, progress.Phase, progress.FileID
				}
			}
		}
		if e.Type == "error" {
			c := e.Value.(*CLIError)
			if c.Code == "cancelled" && stopping && action == "prepare" {
				// The owner asked for Cancel; the CLI confirmed it (cancelled, exit 3).
				cancelConfirmed = true
			} else {
				terminalErr = fmt.Errorf("CLI %s", c.Code)
				send("failure", Failure{Code: c.Code, Source: "cli", Remedy: c.Remedy, Detail: c.Detail, Memory: memoryShortfall(c), Component: c.Component, Stage: c.Stage, Switch: c.Switch})
				beginStop()
			}
		}
		send("event", e)
	}
	for {
		select {
		case <-stdoutEnded:
			stdoutEnded = nil
			// The stream can close well before Windows reports the exit (a busy
			// machine, or a large process tearing down); the exit status decides.
			// Only a process still running after the stop bound lost its pipe.
			pipeTimer = time.NewTimer(s.config.StopTimeout)
			pipeDeadline = pipeTimer.C
		case <-pipeDeadline:
			pipeDeadline = nil
			if !stopping && !prepared {
				terminalErr = protocol("lifecycle stream closed before process exit")
				send("failure", Failure{Code: "protocol_mismatch", Source: "supervisor"})
				beginStop()
			}
		case e := <-events:
			handleEvent(e)
		case err := <-bad:
			terminalErr = err
			send("failure", Failure{Code: "protocol_mismatch", Source: "supervisor"})
			beginStop()
		case b := <-op.input:
			// The idle watchdog starts when the approval (or decline) line is
			// sent, never while approval_required waits for the user.
			if action == "prepare" {
				lastPrepareProgress = time.Now()
				var answer struct {
					Action string `json:"action"`
				}
				if json.Unmarshal(b, &answer) == nil && answer.Action == "decline" {
					stopping = true // the CLI's reply "cancelled" is the user's answer
				}
			}
			go func() {
				defer ReportPanic()
				if err := p.Input(b); err != nil {
					select {
					case bad <- err:
					default:
					}
				}
			}()
		case <-op.stop:
			beginStop()
		case <-load.C:
			if !verifiedReady {
				terminalErr = appError("startup_timeout", "load deadline expired")
				send("failure", Failure{Code: "startup_timeout", Source: "supervisor"})
				beginStop()
			}
		case <-stopDeadline:
			forced = true
			send("stop-phase", "force owned job")
			_ = p.Force()
			stopDeadline = nil
		case <-poll.C:
			if action == "prepare" && !stopping && !prepared && !lastPrepareProgress.IsZero() {
				limit := stallLimit(s.StallTimeout())
				if limit > 0 && time.Since(lastPrepareProgress) >= limit {
					terminalErr = appError("download_stalled", "Engine download stopped reporting progress.")
					send("failure", Failure{Code: "download_stalled", Source: "supervisor"})
					// Ask the CLI to cancel; only StopTimeout forces it.
					beginStop()
					continue
				}
			}
			if interval := s.PollInterval(); interval > s.config.PollInterval && time.Since(lastQuery) < interval {
				continue
			}
			if action == "start" && !stopping && !verifiedReady && readyEvent != nil && !queryActive && identity.ID != "" {
				lastQuery = time.Now()
				queryActive = true
				queryResult = make(chan any, 1)
				ch := queryResult
				copy := identity
				go func() {
					defer ReportPanic()
					st, err := s.query(s.ctx, &copy)
					if err != nil {
						ch <- err
					} else {
						ch <- st
					}
				}()
			}
		case value := <-queryResult:
			queryResult = nil
			queryActive = false
			if st, ok := value.(*Status); ok {
				send("status", st)
				agrees := false
				for _, i := range st.Instances {
					life := (*Lifecycle)(nil)
					if readyEvent != nil {
						life = readyEvent.Value.(*Lifecycle)
					}
					effective, _ := json.Marshal(i.Effective)
					var readyEffective []byte
					if life != nil {
						readyEffective, _ = json.Marshal(life.Effective)
					}
					if sameInstance(i, identity) && i.Ready && i.Phase == "ready" && life != nil && i.ServedModel() == life.Model && bytes.Equal(effective, readyEffective) && i.Endpoint != nil && identity.Endpoint != nil && *i.Endpoint == *identity.Endpoint && VerifyIdentity(i, s.config.Exe) == nil {
						agrees = true
						identity = i
					}
				}
				if agrees && readyEvent != nil && !stopping {
					verifiedReady = true
					send("ready", identity)
					load.Stop()
					poll.Stop()
				} else {
					send("stale", nil)
				}
			} else {
				send("stale", nil)
			}
		case result := <-p.Done:
			if stopTimer != nil {
				stopTimer.Stop()
			}
			send("stop-phase", "drain EOF")
			// Readers have finished before Done. Consume all events before the outcome.
			for len(events) > 0 {
				handleEvent(<-events)
			}
			select {
			case err := <-bad:
				terminalErr = err
			default:
			}
			if result.Err != nil {
				terminalErr = result.Err
			}
			if result.DrainTimedOut {
				terminalErr = protocol("post-exit drain deadline expired")
				forced = true
			}
			if result.ExitCode != 0 && !forced && terminalErr == nil && !stoppedUnnamed && !(cancelConfirmed && result.ExitCode == 3) {
				code := "server_exited"
				if action == "start" && result.ExitCode == 75 {
					code = "device_lost" // the engine's exit status for a lost GPU (EX_TEMPFAIL)
				}
				if result.ExitCode == statusDLLNotFound {
					code = "vcredist_missing"
				}
				terminalErr = appError(code, fmt.Sprintf("unexpected exit %d", result.ExitCode))
			}
			if terminalErr == nil && !forced {
				if action == "prepare" && !prepared && !stopping {
					terminalErr = protocol("prepare did not publish success")
				}
				if action == "start" && (!stopping || !cleanStopped) && !stoppedUnnamed {
					terminalErr = appError("server_exited", "process exited without a verified clean stop")
				}
			}
			send("complete", completion{terminalErr, forced, result.ExitCode, stopping})
			return
		}
	}
}

// sameExePath compares two paths of one file: letter case, junctions and
// symbolic links make no difference. Either side may be the one that went
// through a link, so both are resolved.
func sameExePath(a, b string) bool {
	return strings.EqualFold(canonicalPath(a), canonicalPath(b))
}

func (s *Supervisor) requestStop(i Instance) error {
	if err := VerifyIdentity(i, s.config.Exe); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(s.ctx, s.config.StopTimeout)
	defer cancel()
	st, err := s.query(ctx, &i)
	if err != nil {
		return err
	}
	found := false
	for _, current := range st.Instances {
		if sameInstance(current, i) {
			found = true
		}
	}
	if !found || !st.Capabilities.InstanceControl {
		return protocol("stop identity handshake failed")
	}
	if err = VerifyIdentity(i, s.config.Exe); err != nil {
		return err
	}
	wait := s.config.StopTimeout.Milliseconds()
	if wait < 0 || wait > 4294967294 {
		wait = 4294967294
	}
	events, err := s.captureEvents(ctx, []string{"stop", "--json", "--instance-id", i.ID, "--wait-ms", strconv.FormatInt(wait, 10)}, 2)
	if err != nil {
		return err
	}
	return checkStop(events, i)
}
func (s *Supervisor) stopExternal(i Instance) error {
	if err := VerifyIdentity(i, s.config.Exe); err != nil {
		return err
	}
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, i.PID)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	created, err := creationTime(h)
	if err != nil {
		return err
	}
	if created != i.CreationTime {
		return protocol("external identity changed before stop")
	}
	if err := s.requestStop(i); err != nil {
		return err
	}
	deadline := time.NewTimer(s.config.StopTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return s.ctx.Err()
		case <-deadline.C:
			return errors.New("External server did not stop. Forced stop is not permitted.")
		case <-tick.C:
			state, err := windows.WaitForSingleObject(h, 0)
			if err != nil {
				return err
			}
			if state == windows.WAIT_OBJECT_0 {
				return nil
			}
		}
	}
}

// checkStop accepts the stop result only when the CLI confirmed the exit: an
// acknowledged stopping record may come first, and the last record must be
// stopped with process_exited true for this identity.
func checkStop(events []Event, i Instance) error {
	if len(events) == 0 {
		return protocol("missing stop result")
	}
	for _, e := range events {
		life, ok := e.Value.(*Lifecycle)
		if !ok || (e.Type != "stopping" && e.Type != "stopped") || life.Identity != i.asIdentity() {
			return protocol("invalid stop acknowledgement")
		}
		if e.Type == "stopping" && (life.Acknowledged == nil || !*life.Acknowledged) {
			return protocol("stop was not acknowledged")
		}
	}
	last := events[len(events)-1]
	life := last.Value.(*Lifecycle)
	if last.Type != "stopped" || life.ProcessExited == nil || !*life.ProcessExited {
		return protocol("stop did not confirm process exit")
	}
	return nil
}

// LoadRegistry reads `serve --list-switches --json` once and keeps the rows
// for the window. It needs no model and no engine load.
func (s *Supervisor) LoadRegistry() error {
	e, err := s.capture(s.ctx, []string{"serve", "--list-switches", "--json"})
	if err != nil {
		return err
	}
	sw, ok := e.Value.(*Switches)
	if !ok {
		return protocol("registry listing is not a switches event")
	}
	s.registry.Store(sw)
	return nil
}

func isStallLabel(label string) bool { return strings.HasPrefix(label, "No token progress for ") }

// matchesBind checks the requested network binding without copying a CLI default
// port or endpoint path. With no port override the CLI chooses the port.
func matchesBind(endpoint *string, host string, port int) bool {
	if endpoint == nil || ValidateEndpoint(endpoint) != nil {
		return false
	}
	u, err := url.Parse(*endpoint)
	return err == nil && u.Hostname() == host && (port == 0 || u.Port() == strconv.Itoa(port))
}
