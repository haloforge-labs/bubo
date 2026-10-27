package updater

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	cliRelative    = "bin/bubo.exe"
	guiRelative    = "bin/Bubo.Control.Web.exe"
	helperRelative = "bin/bubo-update.exe"
	journalName    = "transaction.json"
)

func exists(name string) bool { _, err := os.Lstat(name); return !os.IsNotExist(err) }

func zipRelative(name, top string) (string, error) {
	if name == "" || strings.ContainsAny(name, "\\:") || strings.HasPrefix(name, "/") {
		return "", fmt.Errorf("unsafe ZIP path %q", name)
	}
	name = strings.TrimSuffix(name, "/")
	for _, p := range strings.Split(name, "/") {
		if p == "" || p == "." || p == ".." || strings.HasSuffix(p, ".") || strings.HasSuffix(p, " ") || strings.ContainsAny(p, "\x00<>\"|?*") {
			return "", fmt.Errorf("unsafe ZIP component %q", p)
		}
		// filepath.IsLocal also rejects Windows reserved device names.
		if !filepath.IsLocal(p) {
			return "", fmt.Errorf("non-local ZIP component %q", p)
		}
	}
	if top != "" {
		if name == top {
			return "", nil
		}
		var ok bool
		name, ok = strings.CutPrefix(name, top+"/")
		if !ok {
			return "", fmt.Errorf("ZIP entry is outside expected top folder %q", top)
		}
	}
	if !filepath.IsLocal(filepath.FromSlash(name)) {
		return "", fmt.Errorf("ZIP entry is not local")
	}
	return filepath.FromSlash(name), nil
}

func (c *Client) stageArchive(ctx context.Context, archive, root string, a Asset, progress func(Progress)) (*Staged, *Failure) {
	// Stage is a separate trust boundary: a cached/resumed archive is hashed too.
	if f := verifyArchive(ctx, archive, a); f != nil {
		return nil, f
	}
	z, err := zip.OpenReader(archive)
	if err != nil {
		return nil, fail(CodeArchive, "cannot open package ZIP: "+err.Error())
	}
	defer z.Close()
	dest := filepath.Join(root, "package")
	var total int64
	seen := make(map[string]bool)
	rels := make([]string, len(z.File))
	for i, f := range z.File {
		rel, err := zipRelative(f.Name, a.TopFolder)
		if err != nil {
			return nil, fail(CodeArchive, err.Error())
		}
		if f.Mode()&os.ModeType != 0 && !f.FileInfo().IsDir() {
			return nil, fail(CodeArchive, "links and special ZIP entries are refused: "+f.Name)
		}
		if rel == "" {
			if !f.FileInfo().IsDir() {
				return nil, fail(CodeArchive, "top_folder is a file")
			}
			continue
		}
		key := strings.ToLower(rel)
		if _, ok := seen[key]; ok {
			return nil, fail(CodeArchive, "duplicate or case-colliding ZIP path: "+f.Name)
		}
		seen[key] = f.FileInfo().IsDir()
		rels[i] = rel
		for _, base := range []string{dest, c.cfg.InstallDir, c.cfg.InstallDir + ".previous"} {
			if !shortPath(filepath.Join(base, rel)) {
				return nil, fail(CodeArchive, "ZIP path reaches the 260-character Windows limit: "+f.Name)
			}
		}
		if !f.FileInfo().IsDir() {
			if f.UncompressedSize64 > uint64(math.MaxInt64-total) {
				return nil, fail(CodeArchive, "ZIP unpacked size overflows int64")
			}
			total += int64(f.UncompressedSize64)
		}
	}
	if total != a.UnpackedBytes {
		return nil, fail(CodeArchive, fmt.Sprintf("ZIP unpacked bytes: expected %d, received %d", a.UnpackedBytes, total))
	}
	for rel := range seen {
		for parent := filepath.Dir(rel); parent != "."; parent = filepath.Dir(parent) {
			if directory, ok := seen[parent]; ok && !directory {
				return nil, fail(CodeArchive, "ZIP file is also used as a directory")
			}
		}
	}
	for _, required := range PackageExecutables() {
		if dir, ok := seen[strings.ToLower(filepath.FromSlash(required))]; !ok || dir {
			return nil, fail(CodeArchive, "package is missing "+required)
		}
	}
	if err := safeAncestors(dest); err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	if exists(dest) {
		if err := safeTree(dest); err != nil {
			return nil, fail(CodeIO, err.Error())
		}
		if err := os.RemoveAll(dest); err != nil {
			return nil, fail(CodeIO, err.Error())
		}
	}
	if err := ensureDir(dest); err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	finished := false
	defer func() {
		if !finished {
			os.RemoveAll(dest)
		}
	}()
	var unpacked int64
	for i, f := range z.File {
		if err := ctx.Err(); err != nil {
			return nil, fail(CodeCanceled, "staging canceled; verified archive kept")
		}
		rel := rels[i]
		if rel == "" {
			continue
		}
		name := filepath.Join(dest, rel)
		if f.FileInfo().IsDir() {
			if err := ensureDir(name); err != nil {
				return nil, fail(CodeIO, err.Error())
			}
			continue
		}
		if err := ensureDir(filepath.Dir(name)); err != nil {
			return nil, fail(CodeIO, err.Error())
		}
		r, err := f.Open()
		if err != nil {
			return nil, fail(CodeArchive, err.Error())
		}
		w, err := os.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
		if err != nil {
			r.Close()
			return nil, fail(CodeIO, err.Error())
		}
		n, copyErr := io.Copy(w, contextReader{ctx, io.LimitReader(r, int64(f.UncompressedSize64)+1)})
		r.Close()
		syncErr, closeErr := w.Sync(), w.Close()
		if copyErr != nil || n != int64(f.UncompressedSize64) || syncErr != nil || closeErr != nil {
			code := CodeArchive
			if ctx.Err() != nil {
				code = CodeCanceled
			}
			return nil, fail(code, "cannot extract and verify ZIP entry: "+f.Name)
		}
		unpacked += n
		emit(progress, "staging", unpacked, total)
	}
	seal, err := treeSeal(dest)
	if err != nil {
		return nil, fail(CodeIO, err.Error())
	}
	finished = true
	return &Staged{Directory: dest, Version: c.release.Version, root: root, asset: a, seal: seal}, nil
}
