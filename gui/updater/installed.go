package updater

import (
	"os"
	"path/filepath"

	"bubo/gui/updater/jsonwire"
)

// InstalledVersion reads the version shipped with the installed package. It
// does not start the CLI, which may be the reason an update is needed.
func InstalledVersion(installDir string) (string, *Failure) {
	data, err := readRegular(filepath.Join(installDir, "BUILD-MANIFEST.json"))
	if err != nil {
		return "", fail(CodeIO, "cannot read installed package manifest: "+err.Error())
	}
	var manifest struct {
		Product struct {
			Version string `json:"product_version"`
		} `json:"product"`
	}
	if err := jsonwire.Unmarshal(data, &manifest); err != nil {
		return "", fail(CodeVersionInvalid, "installed package manifest is invalid")
	}
	if _, err := parseVersion(manifest.Product.Version); err != nil {
		return "", fail(CodeVersionInvalid, "installed package version is invalid")
	}
	return manifest.Product.Version, nil
}

// ClearReleaseHistory is a deliberate user recovery action. Check's freshness
// guard remains in force until this is called.
func (c *Client) ClearReleaseHistory() *Failure {
	if !c.mu.TryLock() {
		return fail(CodeBusy, "another updater operation is running")
	}
	defer c.mu.Unlock()
	name := filepath.Join(c.root, "seen.json")
	if err := safeAncestors(name); err != nil {
		return fail(CodePathRedirected, err.Error())
	}
	if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
		return fail(CodeIO, "cannot clear release history: "+err.Error())
	}
	return nil
}
