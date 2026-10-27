//go:build windows && bubo_test

package supervisor

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

// TestIsolation is loaded before main, so a misplaced test executable fails
// before instance, settings, credentials, or native UI initialization.
type TestIsolation struct {
	EvidenceRoot   string `json:"evidence_root"`
	InstanceSuffix string `json:"instance_suffix"`
	LoginKeyPath   string `json:"login_key_path"`
	StartupFolder  string `json:"startup_folder"`
	KeyTarget      string `json:"key_target"`
	CaptureFile    string `json:"capture_file"`
	DialogFile     string `json:"dialog_file"`
	DialogFolder   string `json:"dialog_folder"`
	ManifestURL    string `json:"manifest_url"`
	RootCAFile     string `json:"root_ca_file"`
}

var testIsolation TestIsolation

func TestConfiguration() TestIsolation { return testIsolation }

func testPathUnder(root, candidate string) bool {
	if !filepath.IsAbs(candidate) {
		return false
	}
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func init() {
	path := os.Getenv("BUBO_TEST_CONFIG")
	if path == "" {
		panic("bubo_test requires BUBO_TEST_CONFIG")
	}
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &testIsolation) != nil {
		panic("bubo_test configuration is unreadable")
	}
	c := testIsolation
	root := filepath.Clean(c.EvidenceRoot)
	exe, err := os.Executable()
	if err != nil || !filepath.IsAbs(root) || !testPathUnder(root, path) || !testPathUnder(root, exe) || !testPathUnder(root, os.Getenv("LOCALAPPDATA")) || !testPathUnder(root, c.StartupFolder) || !testPathUnder(root, c.CaptureFile) || !testPathUnder(root, c.DialogFile) || !testPathUnder(root, c.DialogFolder) || !testPathUnder(root, c.RootCAFile) {
		panic("bubo_test path escapes evidence root")
	}
	if c.InstanceSuffix == "" || !strings.HasPrefix(c.InstanceSuffix, ".Test.") || c.KeyTarget == "" || c.KeyTarget == KeyTarget || !strings.HasPrefix(c.KeyTarget, "Bubo.Test.") || !strings.HasPrefix(c.LoginKeyPath, `Software\BuboTest\`) || strings.EqualFold(c.LoginKeyPath, loginRunKey) {
		panic("bubo_test names must be isolated")
	}
	u, err := url.Parse(c.ManifestURL)
	if err != nil || u.Scheme != "https" || net.ParseIP(u.Hostname()) == nil || !net.ParseIP(u.Hostname()).IsLoopback() || u.Port() == "" {
		panic(fmt.Sprintf("bubo_test manifest is not loopback HTTPS: %s", c.ManifestURL))
	}
	instanceSuffix = c.InstanceSuffix
	loginKeyPath = c.LoginKeyPath
	loginStartupPath = c.StartupFolder
	keyTarget = c.KeyTarget
	// The fake CLI keeps its socket on loopback even when the saved LAN
	// preference makes the supervisor pass --host 0.0.0.0.
	testLoopbackOnly = false
}
