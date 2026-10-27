package updater

import (
	"os"
	"path/filepath"
	"strings"
)

// RemoveOldRuntime is called only after the new server reports ready. The
// path must be one version folder in the user's shared Bubo runtime root.
func RemoveOldRuntime(oldPath, newPath string) *Failure {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" || !filepath.IsAbs(base) || !filepath.IsAbs(oldPath) || !filepath.IsAbs(newPath) {
		return fail(CodeConfig, "runtime paths are unavailable")
	}
	root := filepath.Clean(filepath.Join(base, "Bubo", "runtime"))
	oldPath, newPath = filepath.Clean(oldPath), filepath.Clean(newPath)
	if !strings.EqualFold(filepath.Dir(oldPath), root) || strings.EqualFold(oldPath, newPath) {
		return fail(CodeConfig, "old runtime is not a distinct Bubo runtime version folder")
	}
	if err := safeTree(oldPath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fail(CodeIO, "cannot inspect old runtime: "+err.Error())
	}
	if err := os.RemoveAll(oldPath); err != nil {
		return fail(CodeIO, "cannot remove old runtime: "+err.Error())
	}
	return nil
}
