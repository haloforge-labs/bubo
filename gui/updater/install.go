package updater

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"bubo/gui/updater/jsonwire"
)

type activeTransaction struct {
	Root                   string `json:"root"`
	Version                string `json:"version"`
	Seal                   string `json:"seal"`
	Helper                 string `json:"helper"`
	HelperHash             string `json:"helper_sha256"`
	RuntimeChanged         bool   `json:"runtime_changed"`
	OldRuntimePath         string `json:"old_runtime_path,omitempty"`
	RuntimePrepared        bool   `json:"runtime_prepared,omitempty"`
	RuntimeFollowupStarted bool   `json:"runtime_followup_started,omitempty"`
	ResultDismissed        bool   `json:"result_dismissed,omitempty"`
	// HelperPID and HelperCreated identify the helper process that launchHelper
	// started. Only launchHelper knows them. They are zero until it has written
	// them, and a handoff without them is never declared abandoned.
	HelperPID     uint32 `json:"helper_pid,omitempty"`
	HelperCreated uint64 `json:"helper_created,omitempty"`
}

func (c *Client) activePath() string { return filepath.Join(c.root, "active.json") }

func (c *Client) pendingHandoff() *Failure {
	a, f := c.readActive()
	if f != nil || a.Root == "" {
		return f
	}
	if exists(filepath.Join(a.Root, journalName)) {
		return fail(CodeRollbackPending, "an earlier transaction needs recovery before another update")
	}
	if !exists(filepath.Join(a.Root, ResultFileName)) {
		gone, f := c.helperGone(a)
		if f != nil {
			return f
		}
		if !gone {
			return fail(CodeBusy, "an earlier helper handoff has not finished; wait for LastResult")
		}
		// The helper ended before it changed anything. Keep the report for the next
		// LastResult and let this update go on.
		return c.settleAbandoned(a)
	}
	return nil
}

// helperGone reports whether the helper that launchHelper started has ended
// without a journal and without a result. It is true only when the exact process
// (PID and creation time) is gone and no helper holds the install lock. A helper
// that is alive, or has not yet taken its lock, is alive as a process, so it is
// never reported gone. A handoff whose helper identity is unknown stays not gone.
func (c *Client) helperGone(a activeTransaction) (bool, *Failure) {
	if a.HelperPID == 0 || a.HelperCreated == 0 {
		return false, nil
	}
	p, err := watchProcess(processIdentity{PID: a.HelperPID, Created: a.HelperCreated})
	if err != nil {
		return false, nil // not proven gone: keep the handoff busy
	}
	defer p.close()
	if exited, err := p.exited(); err != nil || !exited {
		return false, nil
	}
	probe, err := lockInstall(c.cfg.InstallDir)
	if err != nil {
		return false, nil // another helper owns the installation
	}
	probe()
	// A helper writes its journal and result before it ends, so read them after
	// the exit check, never before.
	if exists(filepath.Join(a.Root, journalName)) || exists(filepath.Join(a.Root, ResultFileName)) {
		return false, nil
	}
	return true, nil
}

// settleAbandoned writes the result of a handoff whose helper ended before it
// changed anything. It writes nothing else.
func (c *Client) settleAbandoned(a activeTransaction) *Failure {
	r := InstallResult{Status: "failed", Version: a.Version, Failure: fail(CodeUpdateStopped, "The update stopped before it changed anything. Your installed version is unchanged.")}
	if err := writeJSON(filepath.Join(a.Root, ResultFileName), r); err != nil {
		return fail(CodeIO, "cannot record the stopped update: "+err.Error())
	}
	return nil
}

func (c *Client) Install(staged *Staged) (*Handoff, *Failure) {
	if !c.mu.TryLock() {
		return nil, fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	if runtime.GOOS != "windows" {
		return nil, fail(CodeUnsupported, "installation requires Windows")
	}
	unlock, err := lockHandoff(c.cfg.InstallDir)
	if err != nil {
		return nil, fail(CodeBusy, "another GUI is preparing an update handoff")
	}
	defer unlock()
	probe, err := lockInstall(c.cfg.InstallDir)
	if err != nil {
		return nil, fail(CodeBusy, "an update helper is still running for this installation")
	}
	probe()
	if f := c.preflight(staged); f != nil {
		return nil, f
	}
	ids, f := c.processIDs()
	if f != nil {
		return nil, f
	}
	// A running image and its working directory must both be outside the folder
	// Windows is about to rename. This copy lives on the install volume, inside
	// the transaction staging folder, rather than the potentially different TEMP.
	helperDir := filepath.Join(staged.root, "helper-"+rand.Text()[:12])
	if err := ensureDir(helperDir); err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	helper := filepath.Join(helperDir, "bubo-update.exe")
	hash, err := copyHelper(filepath.Join(c.cfg.InstallDir, filepath.FromSlash(helperRelative)), helper)
	if err != nil {
		os.RemoveAll(helperDir)
		return nil, fail(CodeIO, "cannot copy the shipped update helper: "+err.Error())
	}
	active := activeTransaction{Root: staged.root, Version: staged.Version, Seal: staged.seal, Helper: helper, HelperHash: hash, RuntimeChanged: staged.RuntimeChanged, OldRuntimePath: staged.OldRuntimePath}
	if err := os.Remove(filepath.Join(staged.root, ResultFileName)); err != nil && !os.IsNotExist(err) {
		return nil, fail(CodeIO, "cannot clear the earlier transaction result: "+err.Error())
	}
	if err := writeJSON(c.activePath(), active); err != nil {
		return nil, fail(CodeIO, "cannot save helper handoff: "+err.Error())
	}
	handoff, f := c.launchHelper(active, ids, false)
	if f == nil {
		c.staged = nil
	} else {
		// No process started: preserve an actionable result instead of stranding
		// active.json in the "handoff in progress" state.
		_ = writeJSON(filepath.Join(staged.root, ResultFileName), InstallResult{Status: "failed", Version: staged.Version, Failure: f})
	}
	return handoff, f
}

// Preflight checks every stable install prerequisite before the caller stops
// the server. Install repeats the checks under the handoff lock to catch edits
// between the two phases.
func (c *Client) Preflight(staged *Staged) *Failure {
	if !c.mu.TryLock() {
		return fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	return c.preflight(staged)
}

func (c *Client) preflight(staged *Staged) *Failure {
	if staged == nil || c.staged == nil || *staged != *c.staged {
		return fail(CodeArchive, "Install requires this client's verified staged package")
	}
	if f := c.pendingHandoff(); f != nil {
		return f
	}
	if exists(c.cfg.InstallDir + ".previous") {
		return fail(CodeRollbackPending, "a previous package already exists; recover or resolve that transaction first")
	}
	if exists(filepath.Join(staged.root, journalName)) {
		return fail(CodeBusy, "install transaction already exists; read LastResult and recover it")
	}
	parent := filepath.Dir(c.cfg.InstallDir)
	if err := safeAncestors(parent); err != nil {
		return fail(CodePathRedirected, err.Error())
	}
	info, err := os.Stat(parent)
	if err != nil {
		return fail(CodeIO, "cannot inspect install folder: "+err.Error())
	}
	if info.Mode().Perm()&0200 == 0 {
		return fail(CodeFolderReadOnly, "install folder's parent is read-only")
	}
	seal, err := treeSeal(staged.Directory)
	if err != nil {
		return fail(CodeIO, err.Error())
	}
	if seal != staged.seal {
		return fail(CodeHash, "staged files changed after verification; retry Download")
	}
	if _, err := os.Stat(filepath.Join(c.cfg.InstallDir, filepath.FromSlash(helperRelative))); err != nil {
		return fail(CodeIO, "installed update helper is unavailable: "+err.Error())
	}
	return nil
}

func (c *Client) processIDs() ([]processIdentity, *Failure) {
	if c.cfg.ProcessIDs == nil {
		return nil, fail(CodeProcesses, "supervisor must provide exact GUI and server-child PIDs before Install")
	}
	pids, err := c.cfg.ProcessIDs()
	if err != nil {
		return nil, fail(CodeProcesses, "cannot obtain process PIDs: "+err.Error())
	}
	if len(pids) == 0 {
		return nil, fail(CodeProcesses, "supervisor supplied no GUI PID")
	}
	ids, err := snapshotProcesses(pids)
	if err != nil {
		return nil, fail(CodeProcesses, err.Error())
	}
	return ids, nil
}

func copyHelper(from, to string) (string, error) {
	if !shortPath(from) || !shortPath(to) {
		return "", fmt.Errorf("helper path reaches the 260-character Windows limit")
	}
	if err := safeAncestors(from); err != nil {
		return "", err
	}
	r, err := os.Open(from)
	if err != nil {
		return "", err
	}
	defer r.Close()
	i, err := r.Stat()
	if err != nil {
		return "", err
	}
	if !i.Mode().IsRegular() {
		return "", fmt.Errorf("helper is not a regular file")
	}
	w, err := os.OpenFile(to, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(w, h), r)
	syncErr, closeErr := w.Sync(), w.Close()
	if copyErr != nil {
		return "", copyErr
	}
	if syncErr != nil {
		return "", syncErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	if n != i.Size() {
		return "", fmt.Errorf("helper changed while copying")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (c *Client) launchHelper(active activeTransaction, ids []processIdentity, recover bool) (*Handoff, *Failure) {
	var wait []string
	for _, id := range ids {
		wait = append(wait, strconv.FormatUint(uint64(id.PID), 10)+":"+strconv.FormatUint(id.Created, 10))
	}
	args := []string{"--install", c.cfg.InstallDir, "--staged", active.Root, "--wait-pids", strings.Join(wait, ","), "--health-timeout", c.cfg.HealthTimeout.String(), "--wait-timeout", c.cfg.WaitTimeout.String(), "--version", active.Version, "--seal", active.Seal}
	if recover {
		args = append(args, "--recover")
	}
	p, err := startDetached(active.Helper, filepath.Dir(active.Helper), args...)
	if err != nil {
		return nil, fail(CodeIO, "cannot start external update helper: "+err.Error())
	}
	defer p.close()
	// Record the helper's identity so a later start can tell a helper that ended
	// without a result from one that still runs. If this write fails the helper
	// still runs; the handoff just stays "busy" until its result arrives.
	active.HelperPID, active.HelperCreated = p.identity.PID, p.identity.Created
	_ = writeJSON(c.activePath(), active)
	return &Handoff{HelperPID: int(p.identity.PID), HelperExe: active.Helper}, nil
}

func (c *Client) readActive() (activeTransaction, *Failure) {
	var a activeTransaction
	data, err := readRegular(c.activePath())
	if os.IsNotExist(err) {
		return a, nil
	}
	if err != nil {
		return a, fail(CodeIO, err.Error())
	}
	if err := jsonwire.Unmarshal(data, &a); err != nil {
		return a, fail(CodeIO, "invalid saved update handoff")
	}
	if !within(c.root, a.Root) || filepath.Dir(a.Root) != c.root || !within(a.Root, a.Helper) || a.Seal == "" {
		return activeTransaction{}, fail(CodeIO, "saved update handoff is outside staging")
	}
	if err := safeAncestors(a.Root); err != nil {
		return activeTransaction{}, fail(CodeIO, err.Error())
	}
	return a, nil
}

func (c *Client) LastResult() (*InstallResult, *Failure) {
	if !c.mu.TryLock() {
		return nil, fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	a, f := c.readActive()
	if f != nil || a.Root == "" {
		return nil, f
	}
	data, err := readRegular(filepath.Join(a.Root, ResultFileName))
	if os.IsNotExist(err) {
		if exists(filepath.Join(a.Root, journalName)) {
			// The new GUI may ask during its own health handshake. A live helper
			// is still performing the transaction, not an interrupted update.
			probe, err := lockInstall(c.cfg.InstallDir)
			if err != nil {
				if errors.Is(err, errInstallBusy) {
					return nil, nil
				}
				return nil, fail(CodeIO, "cannot inspect update helper ownership: "+err.Error())
			}
			probe()
			return &InstallResult{Status: "rollback_pending", Version: a.Version, Failure: fail(CodeRollbackPending, "an interrupted update needs Recover")}, nil
		}
		gone, f := c.helperGone(a)
		if f != nil {
			return nil, f
		}
		if !gone {
			return nil, nil // Handoff exists but the helper has not produced a result yet.
		}
		// The helper ended before it changed anything. Report that once.
		if f := c.settleAbandoned(a); f != nil {
			return nil, f
		}
		data, err = readRegular(filepath.Join(a.Root, ResultFileName))
	}
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	var result InstallResult
	if err := jsonwire.Unmarshal(data, &result); err != nil {
		return nil, fail(CodeIO, "invalid updater result file")
	}
	switch result.Status {
	case "installed", "rolled_back", "failed", "rollback_pending":
	default:
		return nil, fail(CodeIO, "unknown updater result status")
	}
	result.RuntimeChanged = a.RuntimeChanged
	result.OldRuntimePath = a.OldRuntimePath
	if a.RuntimePrepared {
		result.RuntimeChanged = false
	}
	result.RuntimeFollowupStarted = a.RuntimeFollowupStarted
	result.ResultDismissed = a.ResultDismissed
	return &result, nil
}

func (c *Client) updateActive(change func(*activeTransaction)) *Failure {
	if !c.mu.TryLock() {
		return fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	a, f := c.readActive()
	if f != nil {
		return f
	}
	if a.Root == "" {
		return fail(CodeIO, "saved update handoff is missing")
	}
	change(&a)
	if err := writeJSON(c.activePath(), a); err != nil {
		return fail(CodeIO, "cannot save runtime update follow-up: "+err.Error())
	}
	return nil
}

func (c *Client) MarkRuntimeFollowupStarted() *Failure {
	return c.updateActive(func(a *activeTransaction) { a.RuntimeFollowupStarted = true })
}

func (c *Client) MarkRuntimePrepared() *Failure {
	return c.updateActive(func(a *activeTransaction) { a.RuntimePrepared = true })
}

func (c *Client) PendingResult() bool {
	a, f := c.readActive()
	return f == nil && a.Root != "" && !exists(filepath.Join(a.Root, ResultFileName))
}

// Dismissing a completed result also removes its persisted replay source.
func (c *Client) ClearLastResult() *Failure {
	if !c.mu.TryLock() {
		return fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	a, f := c.readActive()
	if f != nil || a.Root == "" {
		return f
	}
	if exists(filepath.Join(a.Root, journalName)) {
		return fail(CodeRollbackPending, "recover the interrupted update first")
	}
	if a.RuntimeChanged && !a.RuntimePrepared {
		a.ResultDismissed = true
		if err := writeJSON(c.activePath(), a); err != nil {
			return fail(CodeIO, "cannot dismiss update result: "+err.Error())
		}
		return nil
	}
	if err := os.Remove(filepath.Join(a.Root, ResultFileName)); err != nil && !os.IsNotExist(err) {
		return fail(CodeIO, "cannot clear update result: "+err.Error())
	}
	if err := os.Remove(c.activePath()); err != nil && !os.IsNotExist(err) {
		return fail(CodeIO, "cannot clear update handoff: "+err.Error())
	}
	return nil
}

// Recover retries an interrupted rollback. The caller supplies current GUI and
// server-child PIDs just as for Install, calls Recover, and exits. No live image
// is moved or killed. The failed candidate is retained in staging.
func (c *Client) Recover() (*Handoff, *Failure) {
	if !c.mu.TryLock() {
		return nil, fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	unlock, err := lockHandoff(c.cfg.InstallDir)
	if err != nil {
		return nil, fail(CodeBusy, "another GUI is preparing a recovery handoff")
	}
	defer unlock()
	probe, err := lockInstall(c.cfg.InstallDir)
	if err != nil {
		return nil, fail(CodeBusy, "an update helper is still running; wait before Recover")
	}
	probe()
	a, f := c.readActive()
	if f != nil {
		return nil, f
	}
	if a.Root == "" || (!exists(filepath.Join(a.Root, journalName)) && exists(filepath.Join(a.Root, ResultFileName))) {
		return nil, fail(CodeUnsupported, "there is no interrupted update to recover")
	}
	ids, f := c.processIDs()
	if f != nil {
		return nil, f
	}
	i, err := os.Stat(a.Helper)
	if err != nil {
		return nil, fail(CodeIO, "saved helper is missing")
	}
	if f := verifyArchive(context.Background(), a.Helper, Asset{Size: i.Size(), SHA256: a.HelperHash}); f != nil {
		return nil, f
	}
	return c.launchHelper(a, ids, true)
}
