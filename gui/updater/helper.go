package updater

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"bubo/gui/updater/jsonwire"
)

var errInstallBusy = errors.New("installation mutex is busy")

// HelperOptions are command-line inputs to the external helper, not GUI state.
// StagedDir is the transaction root containing package/, the copied helper, and
// the journal/result. Its only other write targets are InstallDir and its
// .previous sibling. WaitPIDs uses PID:creation-filetime pairs from Install;
// bare PIDs are accepted for standalone administration and opened before waiting.
type HelperOptions struct {
	InstallDir    string
	StagedDir     string
	WaitPIDs      string
	HealthTimeout time.Duration
	WaitTimeout   time.Duration
	Version       string
	Seal          string
	Recover       bool
}

type journal struct {
	Phase      string          `json:"phase"`
	Version    string          `json:"version"`
	Seal       string          `json:"seal"`
	OldSeal    string          `json:"old_seal"`
	Token      string          `json:"token"`
	NewProcess processIdentity `json:"new_process"`
}

type transaction struct {
	opts       HelperOptions
	packageDir string
	previous   string
	j          journal
}

func parsePIDs(s string) ([]processIdentity, error) {
	if s == "" {
		return nil, fmt.Errorf("at least one exact GUI PID is required")
	}
	var ids []processIdentity
	seen := make(map[uint32]bool)
	for _, part := range strings.Split(s, ",") {
		pidText, createdText, _ := strings.Cut(part, ":")
		pid, err := strconv.ParseUint(pidText, 10, 32)
		if err != nil || pid == 0 {
			return nil, fmt.Errorf("invalid wait PID %q", part)
		}
		var created uint64
		if createdText != "" {
			created, err = strconv.ParseUint(createdText, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid PID identity %q", part)
			}
		}
		if seen[uint32(pid)] {
			continue
		}
		seen[uint32(pid)] = true
		ids = append(ids, processIdentity{PID: uint32(pid), Created: created})
	}
	return ids, nil
}

func validateHelper(opts HelperOptions) (*transaction, *Failure) {
	if !filepath.IsAbs(opts.InstallDir) || !filepath.IsAbs(opts.StagedDir) || filepath.Dir(opts.InstallDir) == filepath.Clean(opts.InstallDir) {
		return nil, fail(CodeConfig, "helper paths must be absolute package folders")
	}
	opts.InstallDir, opts.StagedDir = filepath.Clean(opts.InstallDir), filepath.Clean(opts.StagedDir)
	root := filepath.Join(filepath.Dir(opts.InstallDir), "."+filepath.Base(opts.InstallDir)+".bubo-update")
	if !within(root, opts.StagedDir) || filepath.Dir(opts.StagedDir) != root {
		return nil, fail(CodeConfig, "staging must be a direct transaction folder beside this install, on the same volume")
	}
	if _, err := parseVersion(opts.Version); err != nil {
		return nil, fail(CodeConfig, err.Error())
	}
	seal, err := hex.DecodeString(opts.Seal)
	if err != nil || len(seal) != 32 {
		return nil, fail(CodeConfig, "staged tree seal must be SHA-256")
	}
	if opts.HealthTimeout < time.Millisecond || opts.WaitTimeout < time.Millisecond {
		return nil, fail(CodeConfig, "helper timeouts must be positive")
	}
	t := &transaction{opts: opts, packageDir: filepath.Join(opts.StagedDir, "package"), previous: opts.InstallDir + ".previous"}
	for _, name := range []string{opts.InstallDir, t.previous, opts.StagedDir} {
		if err := safeAncestors(name); err != nil {
			return nil, fail(CodeIO, err.Error())
		}
		if !shortPath(name) {
			return nil, fail(CodeConfig, "helper path reaches the 260-character Windows limit")
		}
	}
	exe, err := os.Executable()
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	for _, lockedRoot := range []string{opts.InstallDir, t.previous, t.packageDir} {
		if strings.EqualFold(filepath.Clean(exe), lockedRoot) || within(lockedRoot, exe) || strings.EqualFold(filepath.Clean(cwd), lockedRoot) || within(lockedRoot, cwd) {
			return nil, fail(CodeConfig, "helper image and working directory must be outside the folders being swapped")
		}
	}
	return t, nil
}

// RunHelper waits only for the supplied identities and for its own new child.
// It never scans or terminates processes. Rename intent is flushed before each
// folder mutation so Recover can restore .previous after an interrupted helper.
func RunHelper(ctx context.Context, opts HelperOptions) (InstallResult, *Failure) {
	t, f := validateHelper(opts)
	if f != nil {
		return InstallResult{}, f
	}
	unlockInstall, err := lockInstall(t.opts.InstallDir)
	if err != nil {
		return InstallResult{}, fail(CodeBusy, "another helper owns this installation")
	}
	defer unlockInstall()
	unlock, err := lockTransaction(opts.StagedDir)
	if err != nil {
		return InstallResult{}, fail(CodeBusy, "another helper holds the transaction lock")
	}
	defer unlock()
	ids, err := parsePIDs(opts.WaitPIDs)
	if err != nil {
		return t.finish("failed", fail(CodeProcesses, err.Error()), false)
	}
	var watched []*watchedProcess
	defer func() {
		for _, p := range watched {
			p.close()
		}
	}()
	// Open every handle before waiting for any PID, so PID reuse during a long
	// wait never makes the helper wait for an unrelated process.
	for _, id := range ids {
		if id.PID == uint32(os.Getpid()) {
			return t.finish("failed", fail(CodeProcesses, "helper cannot wait for itself"), false)
		}
		p, err := watchProcess(id)
		if err != nil {
			return t.finish("failed", fail(CodeProcesses, fmt.Sprintf("cannot open PID %d: %v", id.PID, err)), false)
		}
		watched = append(watched, p)
	}
	// The original 60-second wait is a progress boundary, not permission to
	// abandon an otherwise healthy update. Keep waiting for these exact handles
	// for one bounded grace period before reporting a blocked handoff.
	waitLimit := opts.WaitTimeout
	if waitLimit >= DefaultWaitTimeout {
		waitLimit += 30 * time.Second
	}
	waitCtx, cancel := context.WithTimeout(ctx, waitLimit)
	defer cancel()
	for _, p := range watched {
		if err := p.wait(waitCtx); err != nil {
			if opts.Recover {
				return t.pending(fmt.Sprintf("PID %d did not exit before recovery's wait deadline", p.identity.PID))
			}
			return t.finish("failed", fail(CodeProcesses, fmt.Sprintf("PID %d did not exit before the wait deadline; no folders moved", p.identity.PID)), false)
		}
	}
	if opts.Recover {
		return t.recover(ctx)
	}
	if exists(filepath.Join(opts.StagedDir, journalName)) {
		return InstallResult{}, fail(CodeRollbackPending, "transaction journal exists; invoke Recover")
	}
	if exists(t.previous) {
		return t.finish("failed", fail(CodeRollbackPending, ".previous already exists; no folders moved"), false)
	}
	if err := safeTree(opts.InstallDir); err != nil {
		return t.finish("failed", fail(CodeIO, err.Error()), false)
	}
	seal, err := treeSeal(t.packageDir)
	if err != nil {
		return t.finish("failed", fail(CodeIO, err.Error()), false)
	}
	if seal != opts.Seal {
		return t.finish("failed", fail(CodeHash, "staged tree changed after Install; no folders moved"), false)
	}
	oldSeal, err := treeSeal(opts.InstallDir)
	if err != nil {
		return t.finish("failed", fail(CodeIO, "cannot seal original package: "+err.Error()), false)
	}
	t.j = journal{Phase: "moving_old", Version: opts.Version, Seal: opts.Seal, OldSeal: oldSeal, Token: rand.Text()}
	if err := t.save(); err != nil {
		return InstallResult{}, fail(CodeIO, "cannot persist transaction intent: "+err.Error())
	}
	if err := renameWithRetry(ctx, opts.InstallDir, t.previous); err != nil {
		result, f := t.finish("failed", swapFailure(opts.InstallDir, err), true)
		return t.restartOld(result, f)
	}
	t.j.Phase = "moving_new"
	if err := t.save(); err != nil {
		return t.rollback(ctx, fail(CodeIO, "cannot persist new-folder intent: "+err.Error()))
	}
	if err := renameWithRetry(ctx, t.packageDir, opts.InstallDir); err != nil {
		return t.rollback(ctx, swapFailure(t.packageDir, err))
	}
	t.j.Phase = "starting"
	if err := t.save(); err != nil {
		return t.rollback(ctx, fail(CodeIO, "cannot persist GUI-start intent: "+err.Error()))
	}
	deadline := time.Now().Add(opts.HealthTimeout)
	p, err := startDetached(filepath.Join(opts.InstallDir, filepath.FromSlash(guiRelative)), opts.InstallDir, HealthFlag, t.j.Token, HealthDeadlineFlag, deadline.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return t.rollback(ctx, fail(CodeHealth, "new GUI could not start: "+err.Error()))
	}
	defer p.close()
	t.j.Phase = "health"
	t.j.NewProcess = p.identity
	if err := t.save(); err != nil {
		return t.afterFailedHealth(ctx, p, fail(CodeIO, "cannot persist new GUI identity: "+err.Error()))
	}
	if f := t.waitHealth(ctx, p, deadline); f != nil {
		return t.afterFailedHealth(ctx, p, f)
	}
	t.j.Phase = "committed"
	if err := t.save(); err != nil {
		return t.afterFailedHealth(ctx, p, fail(CodeIO, "cannot persist health commit: "+err.Error()))
	}
	return t.commit()
}

func (t *transaction) save() error {
	return writeJSON(filepath.Join(t.opts.StagedDir, journalName), t.j)
}

func (t *transaction) finish(status string, f *Failure, clearJournal bool) (InstallResult, *Failure) {
	r := InstallResult{Status: status, Version: t.opts.Version, Failure: f}
	if err := writeJSON(filepath.Join(t.opts.StagedDir, ResultFileName), r); err != nil {
		return r, fail(CodeIO, "cannot persist update result; journal retained: "+err.Error())
	}
	if clearJournal {
		if err := os.Remove(filepath.Join(t.opts.StagedDir, journalName)); err != nil && !os.IsNotExist(err) {
			r.Failure = fail(CodeIO, "cannot clear completed transaction journal; invoke Recover: "+err.Error())
			if err := writeJSON(filepath.Join(t.opts.StagedDir, ResultFileName), r); err != nil {
				return r, fail(CodeIO, "cannot persist journal-cleanup failure: "+err.Error())
			}
			return r, r.Failure
		}
	}
	return r, nil
}

func swapFailure(name string, err error) *Failure {
	f := fail(CodeSwap, "folder rename failed after sharing-violation retries: "+err.Error())
	f.Path = name
	f.Retry = "close programs holding package files, then retry Install"
	return f
}

func renameWithRetry(ctx context.Context, from, to string) error {
	return retryFileMove(ctx, from, to, moveDirectory)
}

func retryFileMove(ctx context.Context, from, to string, move func(string, string) error) error {
	if err := safeAncestors(from); err != nil {
		return err
	}
	if err := safeAncestors(to); err != nil {
		return err
	}
	for attempt := 0; ; attempt++ {
		err := move(from, to)
		if err == nil || !sharingViolation(err) || attempt == 5 {
			return err
		}
		timer := time.NewTimer((50 * time.Millisecond) << attempt)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func (t *transaction) waitHealth(ctx context.Context, p *watchedProcess, deadline time.Time) *Failure {
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		exited, err := p.exited()
		if err != nil {
			return fail(CodeHealth, "cannot observe the new GUI process")
		}
		if exited {
			return fail(CodeHealth, "new GUI exited before acknowledging health")
		}
		if !time.Now().Before(deadline) {
			return fail(CodeHealth, "new GUI health deadline expired")
		}
		data, err := readRegular(filepath.Join(t.opts.InstallDir, HealthFileName))
		if err == nil && bytes.Equal(data, HealthContent(t.j.Token)) {
			return nil
		}
		if err != nil && !os.IsNotExist(err) {
			return fail(CodeHealth, "cannot read new GUI health acknowledgment: "+err.Error())
		}
		select {
		case <-ctx.Done():
			return fail(CodeHealth, "health wait canceled")
		case <-ticker.C:
		}
	}
}

func (t *transaction) afterFailedHealth(ctx context.Context, p *watchedProcess, reason *Failure) (InstallResult, *Failure) {
	waitCtx, cancel := context.WithTimeout(ctx, t.opts.WaitTimeout)
	defer cancel()
	if err := p.wait(waitCtx); err != nil {
		f := fail(CodeRollbackPending, "new GUI is still running; .previous is intact and rollback must wait for its exit")
		f.Path = t.previous
		f.Retry = "close the new GUI and its server, then invoke Recover; no processes were killed"
		return t.finish("rollback_pending", f, false)
	}
	return t.rollback(ctx, reason)
}

func (t *transaction) rollback(ctx context.Context, reason *Failure) (InstallResult, *Failure) {
	// Even after cancellation, allow bounded recovery rather than abandoning a
	// missing current folder. Rename retries remain finite and do not kill images.
	recoveryCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer cancel()
	if exists(t.opts.InstallDir) {
		if exists(t.packageDir) {
			return t.pending("both current and staging package exist; preserve both for recovery")
		}
		if err := renameWithRetry(recoveryCtx, t.opts.InstallDir, t.packageDir); err != nil {
			return t.pending("cannot move failed candidate back to staging: " + err.Error())
		}
	}
	if !exists(t.previous) {
		return t.pending("previous package is missing; manual recovery is required")
	}
	if err := renameWithRetry(recoveryCtx, t.previous, t.opts.InstallDir); err != nil {
		return t.pending("cannot restore .previous: " + err.Error())
	}
	r, f := t.finish("rolled_back", reason, true)
	return t.restartOld(r, f)
}

func (t *transaction) pending(message string) (InstallResult, *Failure) {
	f := fail(CodeRollbackPending, message)
	f.Path = t.previous
	f.Retry = "close programs using package files and invoke Recover; .previous has been retained"
	return t.finish("rollback_pending", f, false)
}

func (t *transaction) restartOld(r InstallResult, f *Failure) (InstallResult, *Failure) {
	if f != nil {
		return r, f
	}
	p, err := startDetached(filepath.Join(t.opts.InstallDir, filepath.FromSlash(guiRelative)), t.opts.InstallDir)
	if err != nil {
		r.Failure = fail(CodeIO, "old folder restored, but its GUI could not restart: "+err.Error())
		return t.finish(r.Status, r.Failure, false)
	}
	p.close()
	return r, nil
}

func (t *transaction) commit() (InstallResult, *Failure) {
	if err := safeTree(t.previous); err != nil && !os.IsNotExist(err) {
		return t.finish("installed", fail(CodeIO, "new GUI is healthy; previous-folder cleanup refused: "+err.Error()), false)
	}
	if err := os.RemoveAll(t.previous); err != nil {
		return t.finish("installed", fail(CodeIO, "new GUI is healthy; previous-folder cleanup needs Recover: "+err.Error()), false)
	}
	if err := os.Remove(filepath.Join(t.opts.InstallDir, HealthFileName)); err != nil && !os.IsNotExist(err) {
		return t.finish("installed", fail(CodeIO, "new GUI is healthy; health-marker cleanup needs Recover"), false)
	}
	if err := os.Remove(filepath.Join(t.opts.StagedDir, "archive.zip")); err != nil && !os.IsNotExist(err) {
		return t.finish("installed", fail(CodeIO, "new GUI is healthy; archive cleanup needs Recover: "+err.Error()), false)
	}
	return t.finish("installed", nil, true)
}

func (t *transaction) recover(ctx context.Context) (InstallResult, *Failure) {
	data, err := readRegular(filepath.Join(t.opts.StagedDir, journalName))
	if os.IsNotExist(err) && exists(t.opts.InstallDir) && exists(t.packageDir) && !exists(t.previous) {
		// The helper can be interrupted while waiting for the original GUI,
		// before any rename intent was written. Never strand that handoff.
		if err := safeTree(t.opts.InstallDir); err != nil {
			return t.pending(err.Error())
		}
		r, f := t.finish("failed", fail(CodeProcesses, "handoff interrupted before folder mutation; original package retained"), true)
		return t.restartOld(r, f)
	}
	if err != nil {
		return InstallResult{}, fail(CodeIO, "cannot read recovery journal: "+err.Error())
	}
	if err := jsonwire.Unmarshal(data, &t.j); err != nil || t.j.Version != t.opts.Version || t.j.Seal != t.opts.Seal {
		return InstallResult{}, fail(CodeIO, "recovery journal does not match the saved handoff")
	}
	if t.j.Phase == "committed" {
		return t.commit()
	}
	switch t.j.Phase {
	case "moving_old", "moving_new", "starting", "health":
	default:
		return InstallResult{}, fail(CodeIO, "unknown recovery phase")
	}
	if t.j.NewProcess.PID != 0 {
		p, err := watchProcess(t.j.NewProcess)
		if err != nil {
			return t.pending("cannot observe interrupted candidate process: " + err.Error())
		}
		defer p.close()
		waitCtx, cancel := context.WithTimeout(ctx, t.opts.WaitTimeout)
		defer cancel()
		if err := p.wait(waitCtx); err != nil {
			return t.pending("interrupted candidate is still running; wait for its exit")
		}
	}
	if !exists(t.previous) {
		if exists(t.opts.InstallDir) && exists(t.packageDir) {
			seal, err := treeSeal(t.opts.InstallDir)
			if err != nil || seal != t.j.OldSeal {
				return t.pending("original package cannot be identified after interrupted recovery")
			}
			status, code, message := "rolled_back", CodeHealth, "interrupted rollback had already restored the original folder"
			if t.j.Phase == "moving_old" {
				status, code, message = "failed", CodeSwap, "update interrupted before the first rename; original folder retained"
			}
			r, f := t.finish(status, fail(code, message), true)
			return t.restartOld(r, f)
		}
		return t.pending("journal exists but previous package is missing")
	}
	return t.rollback(ctx, fail(CodeHealth, "interrupted update rolled back from its journal"))
}
