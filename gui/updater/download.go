package updater

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

func (c *Client) assetRoot(a Asset) string {
	return filepath.Join(c.root, a.SHA256[:24])
}

// PlanSpace credits only the reusable bytes of this exact immutable asset.
// Other partials already reduce FreeBytes and receive no credit. Windows free
// space excludes existing bytes: additional = archive - partial + unpacked +
// rollback. Including partial a second time would incorrectly block resumes.
func (c *Client) PlanSpace(a Asset, partial int64) (SpacePlan, *Failure) {
	p := SpacePlan{ArchiveBytes: a.Size, ExistingPartialBytes: partial, UnpackedBytes: a.UnpackedBytes}
	if a.Size <= 0 || a.UnpackedBytes <= 0 || partial < 0 || partial > a.Size {
		return p, fail(CodeSize, "invalid archive or partial size for space planning")
	}
	rollback, err := treeBytes(c.cfg.InstallDir)
	if err != nil {
		return p, fail(CodeIO, "cannot measure rollback package: "+err.Error())
	}
	p.RollbackBytes = rollback
	need := uint64(a.Size-partial) + uint64(a.UnpackedBytes)
	if uint64(rollback) > math.MaxUint64-need {
		return p, fail(CodeSpace, "update space requirement overflows uint64")
	}
	p.AdditionalBytes = need + uint64(rollback)
	p.FreeBytes, err = c.freeSpace(filepath.Dir(c.cfg.InstallDir))
	if err != nil {
		return p, fail(CodeIO, "cannot read volume free space: "+err.Error())
	}
	if p.FreeBytes < p.AdditionalBytes {
		f := fail(CodeSpace, fmt.Sprintf("update needs %d additional bytes; volume has %d free (archive %d, existing partial %d, unpacked %d, rollback %d)", p.AdditionalBytes, p.FreeBytes, p.ArchiveBytes, partial, p.UnpackedBytes, rollback))
		f.Partial, f.Retry = "kept", "free space on the install volume, then retry Download"
		return p, f
	}
	return p, nil
}

func emit(progress func(Progress), phase string, received, total int64) {
	if progress != nil {
		progress(Progress{Phase: phase, ReceivedBytes: received, TotalBytes: total})
	}
}

func fileSize(name string) (int64, error) {
	if err := safeAncestors(name); err != nil {
		return 0, err
	}
	i, err := os.Stat(name)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if !i.Mode().IsRegular() {
		return 0, fmt.Errorf("not a regular archive: %s", name)
	}
	return i.Size(), nil
}

func downloadFailure(f *Failure, a Asset, name string, received int64, partial string) *Failure {
	f.URL, f.Path, f.ExpectedBytes, f.ReceivedBytes, f.Partial = a.URL, name, a.Size, received, partial
	f.ExpectedHash = a.SHA256
	if f.Code == CodeHash {
		f.Retry = "retry Download to fetch a fresh archive; if it repeats, use the official release page"
	} else if f.Code != CodeSpace {
		f.Retry = "retry Download; kept partial data is resumed and fully hashed again"
	}
	if f.Code == CodeResume {
		f.Retry = "retry Download; if Range is still refused, remove the named partial and retry, or install from the official release page"
	}
	return f
}

func (c *Client) Download(ctx context.Context, progress func(Progress)) (*Staged, *Failure) {
	if !c.mu.TryLock() {
		return nil, fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	if c.release == nil {
		return nil, fail(CodeNotChecked, "Check must return Available before Download")
	}
	if f := c.pendingHandoff(); f != nil {
		return nil, f
	}
	c.staged = nil
	a := c.release.Asset
	root := c.assetRoot(a)
	if err := ensureDir(root); err != nil {
		return nil, fail(CodeIO, "cannot create same-volume staging: "+err.Error())
	}
	if exists(filepath.Join(root, journalName)) {
		return nil, fail(CodeBusy, "a previous install transaction needs recovery before downloading again")
	}
	partial, complete := filepath.Join(root, "archive.part"), filepath.Join(root, "archive.zip")
	completeSize, err := fileSize(complete)
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	name := partial
	if completeSize > 0 {
		name = complete
	}
	n, err := fileSize(name)
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	if n > a.Size {
		state := "removed"
		if err := os.Remove(name); err != nil {
			state = "kept"
		}
		return nil, downloadFailure(fail(CodeSize, "existing archive exceeds the expected size"), a, name, n, state)
	}
	if _, f := c.PlanSpace(a, n); f != nil {
		return nil, downloadFailure(f, a, name, n, "kept")
	}
	if n < a.Size {
		if name == complete {
			state := "removed"
			if err := os.Remove(complete); err != nil {
				state = "kept"
			}
			return nil, downloadFailure(fail(CodeSize, "completed archive has the wrong size"), a, complete, n, state)
		}
		if f := c.transfer(ctx, a, partial, n, progress); f != nil {
			return nil, f
		}
	}
	emit(progress, "verifying", a.Size, a.Size)
	if f := verifyArchive(ctx, name, a); f != nil {
		if f.Code == CodeHash || f.Code == CodeSize {
			f.Partial = "removed"
			if err := os.Remove(name); err != nil {
				f.Partial = "kept"
			}
		}
		return nil, f
	}
	if name == partial {
		if err := replaceFile(partial, complete); err != nil {
			return nil, downloadFailure(fail(CodeIO, "cannot mark verified archive complete: "+err.Error()), a, partial, a.Size, "kept")
		}
	}
	emit(progress, "staging", 0, a.UnpackedBytes)
	s, f := c.stageArchive(ctx, complete, root, a, progress)
	if f != nil {
		f = downloadFailure(f, a, complete, a.Size, "kept")
		f.Retry = "retry Download to restage the verified archive; if it repeats, install from the official release page"
		return nil, f
	}
	c.staged = s
	s.RuntimeChanged = c.release.Manifest.Runtime.Changed
	if s.RuntimeChanged {
		s.OldRuntimePath = c.cfg.OldRuntimePath
	}
	emit(progress, "ready", a.UnpackedBytes, a.UnpackedBytes)
	copy := *s
	return &copy, nil
}

type transferWriter struct {
	file     *os.File
	received int64
	total    int64
	progress func(Progress)
	err      error
	activity func()
}

func (w *transferWriter) Write(p []byte) (int, error) {
	n, err := w.file.Write(p)
	w.received += int64(n)
	w.err = err
	if n > 0 && w.activity != nil {
		w.activity()
	}
	emit(w.progress, "downloading", w.received, w.total)
	return n, err
}

func (c *Client) transfer(ctx context.Context, a Asset, name string, offset int64, progress func(Progress)) *Failure {
	readCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var stalled atomic.Bool
	const idleLimit = 30 * time.Second
	timer := time.AfterFunc(idleLimit, func() { stalled.Store(true); cancel() })
	defer timer.Stop()
	rangeHeader := ""
	if offset > 0 {
		rangeHeader = fmt.Sprintf("bytes=%d-", offset)
	}
	resp, f := c.request(readCtx, a.URL, rangeHeader)
	if f != nil {
		if stalled.Load() {
			f = fail(CodeDownloadStalled, "download stopped receiving data")
		}
		return downloadFailure(f, a, name, offset, "kept")
	}
	timer.Reset(idleLimit)
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusPartialContent {
		start, end, total, ok := contentRange(resp.Header.Get("Content-Range"))
		if !ok || start != offset || end != a.Size-1 || total != a.Size {
			return downloadFailure(fail(CodeResume, "server returned a mismatched Content-Range"), a, name, offset, "kept")
		}
	} else if resp.StatusCode == http.StatusOK {
		// A server may ignore Range. Restart safely rather than append a full ZIP.
		offset = 0
	} else {
		return downloadFailure(fail(CodeNetwork, fmt.Sprintf("asset returned HTTP %d", resp.StatusCode)), a, name, offset, "kept")
	}
	if resp.ContentLength >= 0 && resp.ContentLength != a.Size-offset {
		return downloadFailure(fail(CodeSize, "HTTP Content-Length differs from expected remaining bytes"), a, name, offset, "kept")
	}
	flags := os.O_CREATE | os.O_WRONLY
	if offset == 0 {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_APPEND
	}
	file, err := os.OpenFile(name, flags, 0600)
	if err != nil {
		return downloadFailure(fail(CodeIO, err.Error()), a, name, offset, "kept")
	}
	w := &transferWriter{file: file, received: offset, total: a.Size, progress: progress, activity: func() { timer.Reset(idleLimit) }}
	emit(progress, "downloading", offset, a.Size)
	_, copyErr := io.Copy(w, io.LimitReader(resp.Body, a.Size-offset+1))
	syncErr, closeErr := file.Sync(), file.Close()
	if w.received > a.Size {
		state := "removed"
		if err := os.Remove(name); err != nil {
			state = "kept"
		}
		return downloadFailure(fail(CodeSize, "asset exceeds its declared size"), a, name, w.received, state)
	}
	if copyErr != nil || syncErr != nil || closeErr != nil || w.received != a.Size || ctx.Err() != nil {
		code, message := CodeNetwork, "download interrupted before all expected bytes arrived"
		if stalled.Load() {
			code, message = CodeDownloadStalled, "download stopped receiving data"
		}
		if w.err != nil || syncErr != nil || closeErr != nil {
			code, message = CodeIO, "cannot write or flush the partial download"
		}
		if ctx.Err() != nil {
			code, message = CodeCanceled, "download canceled; partial data kept"
		}
		return downloadFailure(fail(code, message), a, name, w.received, "kept")
	}
	return nil
}

func contentRange(s string) (start, end, total int64, ok bool) {
	s, prefix := strings.CutPrefix(s, "bytes ")
	if !prefix {
		return
	}
	span, totalText, found := strings.Cut(s, "/")
	if !found {
		return
	}
	startText, endText, found := strings.Cut(span, "-")
	if !found {
		return
	}
	var e1, e2, e3 error
	start, e1 = strconv.ParseInt(startText, 10, 64)
	end, e2 = strconv.ParseInt(endText, 10, 64)
	total, e3 = strconv.ParseInt(totalText, 10, 64)
	ok = e1 == nil && e2 == nil && e3 == nil && start >= 0 && end >= start && total > end
	return
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}

func verifyArchive(ctx context.Context, name string, a Asset) *Failure {
	if err := safeAncestors(name); err != nil {
		return downloadFailure(fail(CodeIO, err.Error()), a, name, 0, "kept")
	}
	f, err := os.Open(name)
	if err != nil {
		return downloadFailure(fail(CodeIO, err.Error()), a, name, 0, "kept")
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, contextReader{ctx, f})
	if err != nil {
		code := CodeIO
		if ctx.Err() != nil {
			code = CodeCanceled
		}
		return downloadFailure(fail(code, "cannot finish archive verification"), a, name, n, "kept")
	}
	if n != a.Size {
		return downloadFailure(fail(CodeSize, "downloaded size does not match the manifest"), a, name, n, "kept")
	}
	actual := hex.EncodeToString(h.Sum(nil))
	if actual != a.SHA256 {
		f := downloadFailure(fail(CodeHash, "downloaded SHA-256 does not match the manifest"), a, name, n, "kept")
		f.ExpectedHash, f.ReceivedHash = a.SHA256, actual
		return f
	}
	return nil
}
