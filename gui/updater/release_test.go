package updater

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReleaseGeneratorRejectsInconsistentMetadata(t *testing.T) {
	archive := os.Getenv("BUBO_RELEASE_ZIP")
	if archive == "" {
		t.Skip("set BUBO_RELEASE_ZIP to a real package")
	}
	raw, err := os.ReadFile("../tests/scenarios/recordings/release-update/bubo-release.json")
	if err != nil {
		t.Fatal(err)
	}
	var recorded Manifest
	if err := json.Unmarshal(raw, &recorded); err != nil {
		t.Fatal(err)
	}
	changed, bytes := false, int64(0)
	good := ReleaseOptions{MinUpdatableFrom: recorded.MinUpdatableFrom, Notes: recorded.Notes.Summary, Released: recorded.Released, InstallPathLength: 60, RuntimeChanged: &changed, RuntimeDownloadBytes: &bytes, ModelsChanged: &changed, ModelsNeedsPrepare: []string{}}
	m, _, err := GenerateReleaseManifest(archive, good)
	if err != nil || m.Version != recorded.Version || m.Commit != recorded.Commit {
		t.Fatalf("manifest=%+v err=%v", m, err)
	}
	for _, name := range []string{"version", "commit", "tag", "minimum", "path-budget"} {
		t.Run(name, func(t *testing.T) {
			opts := good
			switch name {
			case "version":
				opts.Version = "0.1.0"
			case "commit":
				opts.Commit = "3ff74dce9a67182e572dbddbd8279c058e60fe31"
			case "tag":
				opts.Tag = "v0.1.0"
			case "minimum":
				opts.MinUpdatableFrom = "0.1.1"
			case "path-budget":
				opts.InstallPathLength = 259
			}
			if _, _, err := GenerateReleaseManifest(archive, opts); err == nil {
				t.Fatal("inconsistent metadata accepted")
			}
		})
	}
}

// This opt-in integration test checks the actual generated ZIP with the private
// production validators, without downloading or starting any packaged process.
func TestGeneratedReleasePackage(t *testing.T) {
	manifestPath, archive := os.Getenv("BUBO_RELEASE_MANIFEST"), os.Getenv("BUBO_RELEASE_ZIP")
	if manifestPath == "" || archive == "" {
		t.Skip("set BUBO_RELEASE_MANIFEST and BUBO_RELEASE_ZIP for packaging proof")
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var metadata Manifest
	if err := json.Unmarshal(data, &metadata); err != nil {
		t.Fatal(err)
	}
	install := t.TempDir()
	c, failure := New(Config{InstalledVersion: metadata.MinUpdatableFrom, InstallDir: install})
	if failure != nil {
		t.Fatal(failure)
	}
	defer c.Close()
	m, a, failure := c.decodeManifest(data)
	if failure != nil {
		t.Fatal(failure)
	}
	c.release = &Release{Version: m.Version}
	staged, failure := c.stageArchive(context.Background(), archive, t.TempDir(), a, nil)
	if failure != nil {
		t.Fatal(failure)
	}
	if err := CheckPackageExecutables(staged.Directory); err != nil {
		t.Fatal(err)
	}
	sums, err := os.ReadFile(filepath.Join(filepath.Dir(manifestPath), "SHA256SUMS.txt"))
	if err != nil || strings.TrimSpace(string(sums)) != a.SHA256+"  "+a.Name {
		t.Fatalf("SHA256SUMS differs from generated manifest: %v", err)
	}
	buildData, err := os.ReadFile(filepath.Join(staged.Directory, "BUILD-MANIFEST.json"))
	if err != nil {
		t.Fatal(err)
	}
	var build struct {
		Files []struct {
			Path   string `json:"path"`
			Size   int64  `json:"size"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	if err := json.Unmarshal(buildData, &build); err != nil {
		t.Fatal(err)
	}
	recorded := make(map[string]bool)
	for _, file := range build.Files {
		data, err := os.ReadFile(filepath.Join(staged.Directory, filepath.FromSlash(file.Path)))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(data)
		if int64(len(data)) != file.Size || hex.EncodeToString(h[:]) != file.SHA256 {
			t.Fatalf("BUILD-MANIFEST bytes differ: %s", file.Path)
		}
		recorded[file.Path] = true
	}
	for _, required := range PackageExecutables() {
		if !recorded[required] {
			t.Fatalf("missing executable hash: %s", required)
		}
	}
	if err := os.Remove(filepath.Join(staged.Directory, filepath.FromSlash(helperRelative))); err != nil {
		t.Fatal(err)
	}
	if err := CheckPackageExecutables(staged.Directory); err == nil || !strings.Contains(err.Error(), helperRelative) {
		t.Fatalf("missing helper failure: %v", err)
	}
	consoleImage, err := os.ReadFile(filepath.Join(staged.Directory, filepath.FromSlash(cliRelative)))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staged.Directory, filepath.FromSlash(helperRelative)), consoleImage, 0700); err != nil {
		t.Fatal(err)
	}
	if err := CheckPackageExecutables(staged.Directory); err == nil || !strings.Contains(err.Error(), "Windows GUI subsystem") {
		t.Fatalf("console helper failure: %v", err)
	}
	t.Logf("PASS: private decodeManifest + stageArchive; ZIP hash, size, top folder, unpacked bytes and executable contract; missing-helper rejected")
}
