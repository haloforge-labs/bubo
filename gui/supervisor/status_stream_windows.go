//go:build windows

package supervisor

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

var productVersion = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+`)

type statusTarget struct {
	serial   uint64
	gen      uint64
	enabled  bool
	instance *Instance
	// selection is the selected model as status arguments joined by NUL; a
	// change restarts the stream so readiness follows the selection.
	selection string
}

func (a statusTarget) matches(b statusTarget) bool {
	if a.gen != b.gen || a.enabled != b.enabled || a.selection != b.selection || (a.instance == nil) != (b.instance == nil) {
		return false
	}
	return a.instance == nil || sameInstance(*a.instance, *b.instance)
}

type statusUpdate struct {
	serial     uint64
	status     *Status
	state      string
	err        error
	at         time.Time
	sampleTime string
}

// watchStatuses is the sole owner of the status child. Target changes close
// stdin and reap the old child before launching another, including on exit.
// launchOwnedEnv supplies hidden creation, suspended Job Object assignment and
// the existing bounded drain readers; no shell or separate polling process.
func (s *Supervisor) watchStatuses(control <-chan statusTarget) {
	defer s.helpers.Done()
	var target statusTarget
	var p *OwnedProcess
	var untrack func()
	var done <-chan ProcessResult
	var ended *ProcessResult
	var records chan Event
	var bad chan error
	var wireOp string
	var seq uint64
	var latestSample string
	backoff := 250 * time.Millisecond
	const maxBackoff = 10 * time.Second
	timer := time.NewTimer(time.Hour)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()
	var retry <-chan time.Time
	logExit := func(result ProcessResult) {
		s.appendLog([]byte(fmt.Sprintf("status stream exit: %d (drain timeout: %t)", result.ExitCode, result.DrainTimedOut)), true)
	}
	stop := func() {
		if p == nil {
			return
		}
		p.CloseInput()
		deadline := time.NewTimer(s.config.QueryTimeout)
		var result ProcessResult
		select {
		case result = <-p.Done:
			deadline.Stop()
		case <-deadline.C:
			_ = p.Force() // only this owned job, after the stdin grace deadline
			result = <-p.Done
		}
		if ended != nil {
			result = *ended
		}
		logExit(result)
		untrack()
		p, done = nil, nil
		ended = nil
	}
	defer stop()
	emit := func(st *Status, state string, err error) {
		s.emit(update{target.gen, "status-stream", statusUpdate{target.serial, st, state, err, time.Now(), latestSample}})
	}
	schedule := func() {
		timer.Reset(backoff)
		retry = timer.C
		backoff = min(2*backoff, maxBackoff)
	}
	for {
		// Done means both drain readers finished. Consume every queued record
		// first so a final CLI error cannot be replaced by a generic exit code.
		if ended != nil && len(records) == 0 && len(bad) == 0 {
			result := *ended
			logExit(result)
			untrack()
			p, done, records, bad, ended = nil, nil, nil, nil, nil
			err := result.Err
			if err == nil {
				err = fmt.Errorf("status stream exited (%d)", result.ExitCode)
			}
			emit(nil, StatusReconnecting, err)
			schedule()
		}
		select {
		case <-s.ctx.Done():
			return
		case next := <-control:
			stop()
			records, bad = nil, nil
			timer.Stop()
			retry = nil
			target = next
			backoff = 250 * time.Millisecond
			if target.enabled {
				timer.Reset(0)
				retry = timer.C
			}
		case <-retry:
			retry = nil
			args := []string{"status", "--jsonl", "--watch"}
			if target.instance != nil {
				args = append(args, "--instance-id", target.instance.ID)
			} else if target.selection != "" {
				args = append(args, strings.Split(target.selection, "\x00")...)
			}
			records = make(chan Event, 4)
			bad = make(chan error, 1)
			// Capture per-child channels: a late drain must never enter the next
			// child's sequence/identity validation.
			out, invalid := records, bad
			var err error
			p, err = launchOwnedEnv(s.config.Exe, args, childEnvironment(os.Environ(), "", s.config.RuntimeDir), s.config.DrainTimeout, func(b []byte, complete bool) {
				e, err := ParseEvent(s.keys.Scrub(b))
				if !complete {
					err = protocol("truncated status record")
				}
				if err != nil {
					select {
					case invalid <- err:
					default:
					}
					return
				}
				// The View needs the latest status, not an unbounded backlog.
				select {
				case out <- e:
				default:
					select {
					case invalid <- protocol("status stream consumer overflow"):
					default:
					}
				}
			}, func(b []byte, complete bool) { s.appendLog(s.keys.Scrub(b), complete) }, nil)
			if err != nil {
				emit(nil, StatusReconnecting, err)
				schedule()
				continue
			}
			untrack = s.trackProcess(p)
			done = p.Done
			wireOp, seq, latestSample = "", 0, ""
		case e := <-records:
			var err error
			if e.Product.Name != "bubo" || !productVersion.MatchString(e.Product.Version) || (wireOp != "" && wireOp != e.OperationID) || e.Seq != seq+1 {
				err = protocol("status stream product, operation or sequence")
			}
			if e.Type == "capabilities" && err == nil {
				caps := e.Value.(*Capabilities)
				if seq != 0 || !caps.StatusWatch || !caps.StatusJSON {
					err = protocol("status watch capabilities")
				} else {
					wireOp, seq = e.OperationID, e.Seq
					continue
				}
			}
			if e.Value == nil && err == nil && seq > 0 {
				wireOp, seq = e.OperationID, e.Seq
				s.appendLog([]byte(fmt.Sprintf("status stream additive event: %s operation=%s seq=%d", e.Type, e.OperationID, e.Seq)), true)
				continue
			}
			var st *Status
			if err == nil {
				st, err = statusRecord(e)
			}
			if err == nil && (seq == 0 || !st.Capabilities.StatusWatch) {
				err = protocol("missing status watch capabilities")
			}
			if err == nil && target.instance != nil {
				err = protocol("status stream instance does not match")
				for _, i := range st.Instances {
					if sameInstance(i, *target.instance) && st.Identity != nil && *st.Identity == target.instance.asIdentity() && VerifyIdentity(i, s.config.Exe) == nil {
						err = nil
						break
					}
				}
			}
			if err != nil {
				emit(nil, StatusReconnecting, err)
				stop()
				records, bad = nil, nil
				schedule()
				continue
			}
			wireOp, seq = e.OperationID, e.Seq
			latestSample = st.SampleTime
			// Reset only after sustained operation. A child that emits one record
			// then crashes must still back off exponentially.
			if seq >= 4 {
				backoff = 250 * time.Millisecond
			}
			// Session rates need every engine step. The supervisor retains the
			// samples independently of its hidden-window View payload throttle.
			emit(st, StatusUpdating, nil)
		case err := <-bad:
			emit(nil, StatusReconnecting, err)
			stop()
			records, bad = nil, nil
			schedule()
		case result := <-done:
			ended = &result
			done = nil
		}
	}
}

func statusRecord(e Event) (*Status, error) {
	if _, ok := e.Value.(*CLIError); ok {
		return nil, &CommandError{Event: e}
	}
	st, ok := e.Value.(*Status)
	if !ok || e.Product.Name != "bubo" || !productVersion.MatchString(e.Product.Version) || !st.Capabilities.StatusJSON {
		return nil, protocol("status handshake product, version or capabilities")
	}
	return st, nil
}
