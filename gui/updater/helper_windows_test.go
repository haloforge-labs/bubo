//go:build windows

package updater

import (
	"context"
	"debug/pe"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

var fixtureGUI, fixtureHelper []byte
var fixtureGUIPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "bubo-update-proof-")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	code := func() int {
		defer os.RemoveAll(dir)
		fixtureGUIPath = filepath.Join(dir, "fake-gui.exe")
		helper := filepath.Join(dir, "bubo-update.exe")
		for _, build := range []struct{ output, pkg string }{{helper, "./cmd/bubo-update"}, {fixtureGUIPath, "./testdata/fake-gui"}} {
			cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go.exe"), "build", "-ldflags=-H=windowsgui", "-o", build.output, build.pkg)
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
			hideProcess(cmd)
			if output, err := cmd.CombinedOutput(); err != nil {
				fmt.Fprintf(os.Stderr, "fixture build: %v\n%s", err, output)
				return 1
			}
		}
		fixtureGUI, err = os.ReadFile(fixtureGUIPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		fixtureHelper, err = os.ReadFile(helper)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		binary, err := pe.Open(helper)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		header, ok := binary.OptionalHeader.(*pe.OptionalHeader64)
		binary.Close()
		if !ok || header.Subsystem != pe.IMAGE_SUBSYSTEM_WINDOWS_GUI {
			fmt.Fprintln(os.Stderr, "helper must use the Windows GUI subsystem")
			return 1
		}
		fmt.Printf("helper proof: %d bytes; PE subsystem=Windows GUI; CGO_ENABLED=0\n", len(fixtureHelper))
		return m.Run()
	}()
	os.Exit(code)
}

func processFixture(t *testing.T, d time.Duration) *watchedProcess {
	t.Helper()
	p, err := startDetached(fixtureGUIPath, filepath.Dir(fixtureGUIPath), "--fixture-wait", d.String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := p.wait(ctx); err != nil {
			t.Errorf("own fixture did not exit: %v", err)
		}
		p.close()
	})
	return p
}

func integrationPackage(t *testing.T, install, mode string) (*downloadFixture, *Staged) {
	t.Helper()
	if install == "" {
		install = filepath.Join(t.TempDir(), "Bubo")
	}
	if err := os.MkdirAll(filepath.Join(install, "bin"), 0700); err != nil {
		t.Fatal(err)
	}
	for _, file := range []struct {
		name string
		data []byte
	}{{guiRelative, fixtureGUI}, {helperRelative, fixtureHelper}, {"old-package.txt", []byte("old package\x00bytes")}} {
		if err := os.WriteFile(filepath.Join(install, filepath.FromSlash(file.name)), file.data, 0700); err != nil {
			t.Fatal(err)
		}
	}
	entries := append(fixtureEntries(fixtureGUI, fixtureHelper), archiveEntry{"bubo-v0.1.2-windows-x64-gfx1151/fixture-mode", []byte(mode)})
	f := servePackage(t, install, entries)
	f.client.cfg.WaitTimeout = 2 * time.Second
	staged, err := f.client.Download(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return f, staged
}

func handoffAndWait(t *testing.T, c *Client, s *Staged) (*Handoff, *InstallResult) {
	t.Helper()
	gui, server := processFixture(t, 150*time.Millisecond), processFixture(t, 200*time.Millisecond)
	c.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{gui.identity.PID, server.identity.PID}, nil }
	handoff, f := c.Install(s)
	if f != nil {
		t.Fatal(f)
	}
	if within(c.cfg.InstallDir, handoff.HelperExe) || within(s.Directory, handoff.HelperExe) {
		t.Fatal("helper is running from a renamed folder")
	}
	if _, err := os.Stat(filepath.Join(c.cfg.InstallDir, "old-package.txt")); err != nil {
		t.Fatalf("helper moved folder before exact PIDs exited: %v", err)
	}
	waitHelper(t, handoff.HelperPID)
	r, f := c.LastResult()
	if f != nil || r == nil {
		t.Fatalf("result=%+v failure=%v", r, f)
	}
	return handoff, r
}

func waitHelper(t *testing.T, pid int) {
	t.Helper()
	p, err := watchProcess(processIdentity{PID: uint32(pid)})
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	// Folder and metadata retries plus process creation can outlast ten seconds
	// under concurrent host activity. This is a harness deadline, not a relaxed
	// health deadline: the exact terminal status and original bytes remain strict.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := p.wait(ctx); err != nil {
		t.Fatalf("own helper PID %d did not finish: %v", pid, err)
	}
}

func TestARealUpdateSucceedsHealthPassesShippedHelperOutsideInstall(t *testing.T) {
	f, s := integrationPackage(t, "", "good")
	_, r := handoffAndWait(t, f.client, s)
	if r.Status != "installed" || r.Failure != nil {
		t.Fatalf("%+v", r)
	}
	data, err := os.ReadFile(filepath.Join(f.client.cfg.InstallDir, "fixture-mode"))
	if err != nil || string(data) != "good" {
		t.Fatalf("new install=%q %v", data, err)
	}
	if exists(f.client.cfg.InstallDir + ".previous") {
		t.Fatal("healthy update retained old package unexpectedly")
	}
	if exists(filepath.Join(f.client.cfg.InstallDir, "old-package.txt")) {
		t.Fatal("old package not replaced")
	}
	if exists(filepath.Join(f.client.cfg.InstallDir, HealthFileName)) {
		t.Fatal("health marker not consumed")
	}
}

func TestBNewGUIFailsHealthRollbackOldFolderByteIdentical(t *testing.T) {
	for _, mode := range []string{"exit", "timeout", "stale"} {
		t.Run(mode, func(t *testing.T) {
			f, s := integrationPackage(t, "", mode)
			f.client.cfg.HealthTimeout = 150 * time.Millisecond
			before, err := treeSeal(f.client.cfg.InstallDir)
			if err != nil {
				t.Fatal(err)
			}
			_, r := handoffAndWait(t, f.client, s)
			if r.Status != "rolled_back" || r.Failure == nil || r.Failure.Code != CodeHealth {
				t.Fatalf("%+v", r)
			}
			after, err := treeSeal(f.client.cfg.InstallDir)
			if err != nil || after != before {
				t.Fatalf("old folder changed: before=%s after=%s err=%v", before, after, err)
			}
			if exists(f.client.cfg.InstallDir + ".previous") {
				t.Fatal("previous was not restored")
			}
		})
	}
}

func TestCLockedFileRetriesThenRollbackOldFolderIntact(t *testing.T) {
	f, s := integrationPackage(t, "", "good")
	before, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil {
		t.Fatal(err)
	}
	name, err := windows.UTF16PtrFromString(filepath.Join(f.client.cfg.InstallDir, "old-package.txt"))
	if err != nil {
		t.Fatal(err)
	}
	h, err := windows.CreateFile(name, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(h)
	start := time.Now()
	_, r := handoffAndWait(t, f.client, s)
	if r.Status != "failed" || r.Failure == nil || r.Failure.Code != CodeSwap {
		t.Fatalf("%+v", r)
	}
	if time.Since(start) < 1500*time.Millisecond {
		t.Fatal("sharing violation was not retried with backoff")
	}
	after, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil || after != before {
		t.Fatalf("old folder changed: %s %s %v", before, after, err)
	}
}

func TestGUnicodeAndSpaceInstallPath(t *testing.T) {
	install := filepath.Join(t.TempDir(), "Bubo 日本語 café space")
	f, s := integrationPackage(t, install, "good")
	_, r := handoffAndWait(t, f.client, s)
	if r.Status != "installed" {
		t.Fatalf("%+v", r)
	}
}

func TestUncooperativeCandidateRetainsPreviousThenRecoverRollsBack(t *testing.T) {
	f, s := integrationPackage(t, "", "slow")
	f.client.cfg.HealthTimeout = 50 * time.Millisecond
	f.client.cfg.WaitTimeout = 300 * time.Millisecond
	before, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil {
		t.Fatal(err)
	}
	_, r := handoffAndWait(t, f.client, s)
	if r.Status != "rollback_pending" || r.Failure == nil || r.Failure.Code != CodeRollbackPending || !exists(f.client.cfg.InstallDir+".previous") {
		t.Fatalf("%+v", r)
	}
	data, err := readRegular(filepath.Join(s.root, journalName))
	if err != nil {
		t.Fatal(err)
	}
	var j journal
	if err := json.Unmarshal(data, &j); err != nil {
		t.Fatal(err)
	}
	p, err := watchProcess(j.NewProcess)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := p.wait(ctx); err != nil {
		t.Fatal(err)
	}
	p.close()
	waiter := processFixture(t, 25*time.Millisecond)
	f.client.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
	handoff, failure := f.client.Recover()
	if failure != nil {
		t.Fatal(failure)
	}
	waitHelper(t, handoff.HelperPID)
	r, failure = f.client.LastResult()
	if failure != nil || r.Status != "rolled_back" {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
	after, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil || after != before {
		t.Fatalf("old folder changed: %s %s %v", before, after, err)
	}
}

func TestInstallRejectsModifiedStagedFiles(t *testing.T) {
	f, s := integrationPackage(t, "", "good")
	if err := os.WriteFile(filepath.Join(s.Directory, "fixture-mode"), []byte("tampered"), 0600); err != nil {
		t.Fatal(err)
	}
	if handoff, failure := f.client.Install(s); handoff != nil || failure == nil || failure.Code != CodeHash {
		t.Fatalf("handoff=%v failure=%v", handoff, failure)
	}
}

func TestPIDReuseIdentityIsNotWaited(t *testing.T) {
	p := processFixture(t, 250*time.Millisecond)
	wrong := p.identity
	wrong.Created++
	reused, err := watchProcess(wrong)
	if err != nil {
		t.Fatal(err)
	}
	defer reused.close()
	if done, err := reused.exited(); err != nil || !done {
		t.Fatalf("waited unrelated identity: %v %v", done, err)
	}
}

func TestHelperRefusesStagingOutsideTransactionFolder(t *testing.T) {
	_, f := validateHelper(HelperOptions{InstallDir: filepath.Join(t.TempDir(), "Bubo"), StagedDir: filepath.Join(t.TempDir(), "unrelated"), Version: "0.1.2", Seal: strings.Repeat("a", 64), HealthTimeout: time.Second, WaitTimeout: time.Second})
	if f == nil || f.Code != CodeConfig {
		t.Fatalf("%v", f)
	}
}

func TestInterruptedRenameJournalRecovery(t *testing.T) {
	for _, phase := range []string{"before_journal", "before_first_rename", "between_renames", "before_gui_start", "after_restore", "committed_cleanup"} {
		t.Run(phase, func(t *testing.T) {
			f, s := integrationPackage(t, "", "good")
			c := f.client
			oldSeal, err := treeSeal(c.cfg.InstallDir)
			if err != nil {
				t.Fatal(err)
			}
			helperDir := filepath.Join(s.root, "recovery-helper")
			if err := ensureDir(helperDir); err != nil {
				t.Fatal(err)
			}
			helper := filepath.Join(helperDir, "bubo-update.exe")
			hash, err := copyHelper(filepath.Join(c.cfg.InstallDir, filepath.FromSlash(helperRelative)), helper)
			if err != nil {
				t.Fatal(err)
			}
			a := activeTransaction{Root: s.root, Version: s.Version, Seal: s.seal, Helper: helper, HelperHash: hash}
			if err := writeJSON(c.activePath(), a); err != nil {
				t.Fatal(err)
			}
			j := journal{Phase: "moving_old", Version: s.Version, Seal: s.seal, OldSeal: oldSeal, Token: "fixture-token"}
			switch phase {
			case "between_renames":
				if err := moveDirectory(c.cfg.InstallDir, c.cfg.InstallDir+".previous"); err != nil {
					t.Fatal(err)
				}
				j.Phase = "moving_new"
			case "before_gui_start", "committed_cleanup":
				if err := moveDirectory(c.cfg.InstallDir, c.cfg.InstallDir+".previous"); err != nil {
					t.Fatal(err)
				}
				if err := moveDirectory(s.Directory, c.cfg.InstallDir); err != nil {
					t.Fatal(err)
				}
				j.Phase = "starting"
				if phase == "committed_cleanup" {
					j.Phase = "committed"
				}
			case "after_restore":
				j.Phase = "health"
			}
			if phase != "before_journal" {
				if err := writeJSON(filepath.Join(s.root, journalName), j); err != nil {
					t.Fatal(err)
				}
			}
			// New must work even when the current folder is absent after power loss.
			restarted, failure := New(c.cfg)
			if failure != nil {
				t.Fatal(failure)
			}
			defer restarted.Close()
			waiter := processFixture(t, 25*time.Millisecond)
			restarted.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
			h, failure := restarted.Recover()
			if failure != nil {
				t.Fatal(failure)
			}
			waitHelper(t, h.HelperPID)
			r, failure := restarted.LastResult()
			if failure != nil || r == nil {
				t.Fatalf("result=%v failure=%v", r, failure)
			}
			want := "rolled_back"
			if phase == "before_first_rename" || phase == "before_journal" {
				want = "failed"
			}
			if phase == "committed_cleanup" {
				want = "installed"
			}
			if r.Status != want {
				t.Fatalf("%+v", r)
			}
			if want != "installed" {
				restored, err := treeSeal(c.cfg.InstallDir)
				if err != nil || restored != oldSeal {
					t.Fatalf("old tree changed: %s %s %v", oldSeal, restored, err)
				}
			}
			if exists(filepath.Join(s.root, journalName)) {
				t.Fatal("terminal journal not cleared")
			}
		})
	}
}

func TestSecondRenameFailureRestoresOriginalFolder(t *testing.T) {
	f, s := integrationPackage(t, "", "good")
	before, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil {
		t.Fatal(err)
	}
	name, err := windows.UTF16PtrFromString(filepath.Join(s.Directory, "fixture-mode"))
	if err != nil {
		t.Fatal(err)
	}
	h, err := windows.CreateFile(name, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	var closeOnce sync.Once
	release := func() { closeOnce.Do(func() { windows.CloseHandle(h) }) }
	defer release()
	gui, server := processFixture(t, 150*time.Millisecond), processFixture(t, 200*time.Millisecond)
	f.client.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{gui.identity.PID, server.identity.PID}, nil }
	handoff, failure := f.client.Install(s)
	if failure != nil {
		t.Fatal(failure)
	}
	diagnostic := time.AfterFunc(5*time.Second, func() {
		data, err := readRegular(filepath.Join(s.root, journalName))
		fmt.Printf("second-rename diagnostic for own helper PID %d before releasing file lock: journal=%s err=%v\n", handoff.HelperPID, data, err)
		release()
	})
	defer diagnostic.Stop()
	waitHelper(t, handoff.HelperPID)
	r, failure := f.client.LastResult()
	if failure != nil || r == nil {
		t.Fatalf("result=%+v failure=%v", r, failure)
	}
	if r.Status != "rolled_back" || r.Failure == nil || r.Failure.Code != CodeSwap {
		t.Fatalf("%+v", r)
	}
	after, err := treeSeal(f.client.cfg.InstallDir)
	if err != nil || after != before {
		t.Fatalf("old tree changed: %s %s %v", before, after, err)
	}
}

func TestInstallMutexSerializesDifferentTransactions(t *testing.T) {
	install := filepath.Join(t.TempDir(), "Bubo")
	unlock, err := lockInstall(install)
	if err != nil {
		t.Fatal(err)
	}
	blocked := make(chan bool, 1)
	go func() {
		release, err := lockInstall(install)
		if err == nil {
			release()
		}
		blocked <- err != nil
	}()
	if !<-blocked {
		unlock()
		t.Fatal("second OS thread acquired the install mutex")
	}
	unlock()
	release, err := lockInstall(install)
	if err != nil {
		t.Fatal("released mutex cannot be reacquired:", err)
	}
	release()
}

func TestDownloadRefusesPendingDifferentAssetHandoff(t *testing.T) {
	f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
	root := filepath.Join(f.client.root, "earlier-transaction")
	if err := ensureDir(root); err != nil {
		t.Fatal(err)
	}
	a := activeTransaction{Root: root, Version: "0.1.2", Seal: strings.Repeat("a", 64), Helper: filepath.Join(root, "helper.exe")}
	if err := writeJSON(f.client.activePath(), a); err != nil {
		t.Fatal(err)
	}
	if _, failure := f.client.Download(context.Background(), nil); failure == nil || failure.Code != CodeBusy || f.requests != 0 {
		t.Fatalf("%v", failure)
	}
}

func TestLastResultDuringHealthHandshakeWaitsForLiveHelper(t *testing.T) {
	f, s := integrationPackage(t, "", "slow")
	c := f.client
	c.cfg.HealthTimeout = 200 * time.Millisecond
	c.cfg.WaitTimeout = 300 * time.Millisecond
	waiter := processFixture(t, 25*time.Millisecond)
	c.cfg.ProcessIDs = func() ([]uint32, error) { return []uint32{waiter.identity.PID}, nil }
	h, failure := c.Install(s)
	if failure != nil {
		t.Fatal(failure)
	}
	var j journal
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if data, err := readRegular(filepath.Join(s.root, journalName)); err == nil {
			if err := json.Unmarshal(data, &j); err != nil {
				t.Fatal(err)
			}
			if j.Phase == "health" {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if j.Phase != "health" {
		waitHelper(t, h.HelperPID)
		result, resultFailure := c.LastResult()
		t.Fatalf("health journal did not appear: journal=%+v result=%+v result failure=%v", j, result, resultFailure)
	}
	if r, failure := c.LastResult(); r != nil || failure != nil {
		t.Fatalf("live transaction treated as interrupted: result=%+v failure=%v", r, failure)
	}
	waitHelper(t, h.HelperPID)
	p, err := watchProcess(j.NewProcess)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := p.wait(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestMetadataAtomicReplaceRetriesReaderWithoutDeleteSharing(t *testing.T) {
	name := filepath.Join(t.TempDir(), "metadata.json")
	if err := atomicWrite(name, []byte("old")); err != nil {
		t.Fatal(err)
	}
	r, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			r.Close()
		}
	}()
	done := make(chan error, 1)
	go func() { done <- atomicWrite(name, []byte("new")) }()
	time.Sleep(125 * time.Millisecond)
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	data, err := readRegular(name)
	if err != nil || string(data) != "new" {
		t.Fatalf("metadata=%q err=%v", data, err)
	}
}
