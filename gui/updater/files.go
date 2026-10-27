package updater

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf16"
)

func shortPath(name string) bool { return len(utf16.Encode([]rune(name))) < 260 }

// Reject reparse points at every existing ancestor, not just the final path.
// Nothing in a portable package transaction may redirect into user state.
func safeAncestors(name string) error {
	for p := filepath.Clean(name); ; p = filepath.Dir(p) {
		if err := noReparse(p); err != nil && !os.IsNotExist(err) {
			return err
		}
		if filepath.Dir(p) == p {
			return nil
		}
	}
}

func ensureDir(name string) error {
	if !shortPath(name) {
		return fmt.Errorf("path reaches the 260-character Windows limit: %s", name)
	}
	if err := safeAncestors(name); err != nil {
		return err
	}
	return os.MkdirAll(name, 0700)
}

func safeTree(root string) error {
	if err := safeAncestors(root); err != nil {
		return err
	}
	return filepath.WalkDir(root, func(name string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !shortPath(name) {
			return fmt.Errorf("path reaches the 260-character Windows limit: %s", name)
		}
		if err := noReparse(name); err != nil {
			return err
		}
		if !d.IsDir() && !d.Type().IsRegular() {
			return fmt.Errorf("non-regular package entry: %s", name)
		}
		return nil
	})
}

func readRegular(name string) ([]byte, error) {
	if err := safeAncestors(name); err != nil {
		return nil, err
	}
	info, err := os.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > 4<<20 {
		return nil, fmt.Errorf("not a bounded regular metadata file: %s", name)
	}
	f, err := openSharedRead(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, (4<<20)+1))
	if len(data) > 4<<20 {
		return nil, fmt.Errorf("metadata grew past its size limit: %s", name)
	}
	return data, err
}

func writeJSON(name string, value interface{}) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return atomicWrite(name, append(data, '\n'))
}

func atomicWrite(name string, data []byte) error {
	if !shortPath(name) || !shortPath(filepath.Join(filepath.Dir(name), ".updater-write-1234567890")) {
		return fmt.Errorf("metadata path reaches the 260-character Windows limit: %s", name)
	}
	if err := ensureDir(filepath.Dir(name)); err != nil {
		return err
	}
	if err := safeAncestors(name); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(name), ".updater-write-")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return retryFileMove(context.Background(), tmp, name, replaceFile)
}

func treeBytes(root string) (int64, error) {
	var total int64
	if err := safeTree(root); err != nil {
		return 0, err
	}
	err := filepath.WalkDir(root, func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.Size() > math.MaxInt64-total {
			return fmt.Errorf("package size overflows int64")
		}
		total += info.Size()
		return nil
	})
	return total, err
}

func treeSeal(root string) (string, error) {
	if err := safeTree(root); err != nil {
		return "", err
	}
	h := sha256.New()
	err := filepath.WalkDir(root, func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(root, name)
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "%s\x00", filepath.ToSlash(rel))
		f, err := os.Open(name)
		if err != nil {
			return err
		}
		fh := sha256.New()
		_, copyErr := io.Copy(fh, f)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		h.Write(fh.Sum(nil))
		return nil
	})
	return hex.EncodeToString(h.Sum(nil)), err
}

func within(root, name string) bool {
	rel, err := filepath.Rel(root, name)
	return err == nil && rel != "." && filepath.IsLocal(rel) && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
