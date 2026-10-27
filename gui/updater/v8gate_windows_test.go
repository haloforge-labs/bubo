//go:build windows && v8gate

package updater

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Gate copies never use the owner's installation or LOCALAPPDATA tree.
func a4Install(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(root, "copy-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		rel, err := filepath.Rel(root, dir)
		if err != nil || rel == "." || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
			t.Errorf("refusing cleanup outside A4 evidence root: %q (%v)", dir, err)
			return
		}
		deadline := time.Now().Add(5 * time.Second)
		for {
			err = os.RemoveAll(dir)
			if err == nil {
				return
			}
			if time.Now().After(deadline) {
				t.Errorf("remove test copy: %v", err)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	})
	return filepath.Join(dir, "Bubo")
}

func TestV8F11IdleUpdateDownloadStops(t *testing.T) {
	const idleLimit = 35 * time.Second
	var manifest Manifest
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "bubo-release.json") {
			_ = json.NewEncoder(w).Encode(manifest)
			return
		}
		w.Header().Set("Content-Length", "100")
		_, _ = w.Write([]byte("x"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	manifest = testManifest(server.URL, []byte("100 inert fixture bytes"))
	manifest.Assets[0].Size = 100
	client := testClient(t, server, a4Install(t))
	client.freeSpace = func(string) (uint64, error) { return math.MaxUint64, nil }
	if r := client.Check(context.Background()); r.Status != Available {
		t.Fatalf("manifest check: %+v", r)
	}
	ctx, cancel := context.WithTimeout(context.Background(), idleLimit)
	defer cancel()
	begin := time.Now()
	_, failure := client.Download(ctx, nil)
	if failure == nil || failure.Code != Code("download_stalled") || time.Since(begin) >= idleLimit {
		t.Fatalf("F11: idle body reader must stop before %s with download_stalled; elapsed=%s failure=%+v", idleLimit, time.Since(begin), failure)
	}
}

func TestV8F19SuccessfulInstallRemovesArchive(t *testing.T) {
	f, staged := integrationPackage(t, a4Install(t), "good")
	_, result := handoffAndWait(t, f.client, staged)
	if result.Status != "installed" || result.Failure != nil {
		t.Fatalf("F19: install did not succeed: %+v", result)
	}
	archive := filepath.Join(staged.root, "archive.zip")
	if exists(archive) {
		t.Fatalf("F19: successful install retained %s", archive)
	}
}

func TestV8F12JunctionAncestorHasSpecificCause(t *testing.T) {
	install := a4Install(t)
	parent := filepath.Dir(install)
	real := filepath.Join(parent, "real")
	link := filepath.Join(parent, "junction")
	if err := os.MkdirAll(real, 0700); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("cmd.exe", "/d", "/c", "mklink", "/J", link, real)
	hideProcess(cmd)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("create evidence-root junction: %v: %s", err, output)
	}
	t.Cleanup(func() {
		if err := os.Remove(link); err != nil {
			t.Errorf("remove test junction: %v", err)
		}
	})
	_, failure := New(Config{InstalledVersion: "0.1.1", InstallDir: filepath.Join(link, "Bubo")})
	if failure == nil || failure.Code != Code("update_path_redirected") {
		t.Fatalf("F12: junction ancestor needs update_path_redirected; got %+v", failure)
	}
}

func TestV8F12bReadOnlyParentHasSpecificCause(t *testing.T) {
	f, staged := integrationPackage(t, a4Install(t), "good")
	parent := filepath.Dir(f.client.cfg.InstallDir)
	if err := os.Chmod(parent, 0555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(parent, 0700); err != nil {
			t.Errorf("restore own test folder: %v", err)
		}
	})
	waiter := processFixture(t, 300*time.Millisecond)
	f.client.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
	handoff, failure := f.client.Install(staged)
	if handoff != nil || failure == nil || failure.Code != Code("update_folder_read_only") {
		t.Fatalf("F12b: read-only parent must stop before handoff with update_folder_read_only; handoff=%+v failure=%+v", handoff, failure)
	}
}

func TestV8F12cInvalidManifestVersionHasSpecificCause(t *testing.T) {
	var manifest Manifest
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(manifest)
	}))
	defer server.Close()
	manifest = testManifest(server.URL, []byte("fixture"))
	manifest.Version = "not-a-version"
	client := testClient(t, server, a4Install(t))
	result := client.Check(context.Background())
	if result.Failure == nil || result.Failure.Code != Code("update_version_invalid") {
		t.Fatalf("F12c: invalid version needs update_version_invalid; result=%+v", result)
	}
}

func TestV8F17HelperWaitOverSixtySecondsStillRestarts(t *testing.T) {
	f, staged := integrationPackage(t, a4Install(t), "good")
	f.client.cfg.WaitTimeout = DefaultWaitTimeout
	waiter := processFixture(t, DefaultWaitTimeout+2*time.Second)
	f.client.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
	handoff, failure := f.client.Install(staged)
	if failure != nil {
		t.Fatal(failure)
	}
	p, err := watchProcess(processIdentity{PID: uint32(handoff.HelperPID)})
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
	defer cancel()
	if err := p.wait(ctx); err != nil {
		t.Fatal(err)
	}
	result, failure := f.client.LastResult()
	if failure != nil || result == nil || result.Status != "installed" {
		t.Fatalf("F17: helper lost update after >60 s wait: result=%+v failure=%+v", result, failure)
	}
}

func TestV8F3ColdStatusAtFortySecondsPassesHealth(t *testing.T) {
	install := a4Install(t)
	cold := filepath.Join(filepath.Dir(install), "cold-gui.exe")
	build := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go.exe"), "build", "-ldflags=-H=windowsgui", "-o", cold, filepath.Join("..", "tests", "v8", "faults", "cold-gui.go"))
	hideProcess(build)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build inert cold GUI: %v: %s", err, output)
	}
	candidate, err := os.ReadFile(cold)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(install, "bin"), 0700); err != nil {
		t.Fatal(err)
	}
	for _, entry := range []struct {
		name string
		body []byte
	}{
		{guiRelative, fixtureGUI}, {helperRelative, fixtureHelper}, {"old-package.txt", []byte("old copy")},
	} {
		if err := os.WriteFile(filepath.Join(install, filepath.FromSlash(entry.name)), entry.body, 0700); err != nil {
			t.Fatal(err)
		}
	}
	f := servePackage(t, install, fixtureEntries(candidate, fixtureHelper))
	f.client.cfg.HealthTimeout = DefaultHealthTimeout
	f.client.cfg.WaitTimeout = 2 * time.Second
	staged, failure := f.client.Download(context.Background(), nil)
	if failure != nil {
		t.Fatal(failure)
	}
	waiter := processFixture(t, 150*time.Millisecond)
	f.client.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
	handoff, failure := f.client.Install(staged)
	if failure != nil {
		t.Fatal(failure)
	}
	p, err := watchProcess(processIdentity{PID: uint32(handoff.HelperPID)})
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	ctx, cancel := context.WithTimeout(context.Background(), 95*time.Second)
	defer cancel()
	if err := p.wait(ctx); err != nil {
		t.Fatal(err)
	}
	result, failure := f.client.LastResult()
	if failure != nil || result == nil || result.Status != "installed" {
		// The base times out at 30 seconds. Wait for this test's exact cold
		// candidate to exit before removing its copied executable on Windows.
		var j journal
		if data, err := readRegular(filepath.Join(staged.root, journalName)); err == nil && json.Unmarshal(data, &j) == nil && j.NewProcess.PID != 0 {
			if candidateProcess, err := watchProcess(j.NewProcess); err == nil {
				waitCtx, stop := context.WithTimeout(context.Background(), 20*time.Second)
				_ = candidateProcess.wait(waitCtx)
				stop()
				candidateProcess.close()
			}
		}
		t.Fatalf("F3: a 40-second first status must fit the health deadline; result=%+v failure=%+v", result, failure)
	}
}
