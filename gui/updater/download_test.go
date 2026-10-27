package updater

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type archiveEntry struct {
	name string
	data []byte
}

func packageZip(t *testing.T, entries []archiveEntry) ([]byte, int64) {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	var size int64
	for _, e := range entries {
		f, err := w.CreateHeader(&zip.FileHeader{Name: e.name, Method: zip.Store})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write(e.data); err != nil {
			t.Fatal(err)
		}
		size += int64(len(e.data))
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes(), size
}

func fixtureEntries(gui, helper []byte) []archiveEntry {
	const top = "bubo-v0.1.2-windows-x64-gfx1151/"
	return []archiveEntry{{top + "bin/bubo.exe", bytes.Repeat([]byte("fixture"), 32768)}, {top + guiRelative, gui}, {top + helperRelative, helper}}
}

type downloadFixture struct {
	client      *Client
	server      *httptest.Server
	manifest    Manifest
	payload     []byte
	requests    int
	ranges      []string
	interrupt   bool
	ignoreRange bool
	badRange    bool
}

func servePackage(t *testing.T, install string, entries []archiveEntry) *downloadFixture {
	t.Helper()
	f := &downloadFixture{}
	var unpacked int64
	f.payload, unpacked = packageZip(t, entries)
	f.server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "bubo-release.json") {
			json.NewEncoder(w).Encode(f.manifest)
			return
		}
		f.requests++
		f.ranges = append(f.ranges, r.Header.Get("Range"))
		offset := 0
		if r.Header.Get("Range") != "" && !f.ignoreRange {
			if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-", &offset); err != nil {
				t.Errorf("Range: %v", err)
			}
			start := offset
			if f.badRange {
				start++
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(f.payload)-1, len(f.payload)))
			w.Header().Set("Content-Length", fmt.Sprint(len(f.payload)-offset))
			w.WriteHeader(http.StatusPartialContent)
		} else {
			w.Header().Set("Content-Length", fmt.Sprint(len(f.payload)))
		}
		end := len(f.payload)
		if f.interrupt {
			end /= 2
			f.interrupt = false
		}
		w.Write(f.payload[offset:end])
	}))
	t.Cleanup(f.server.Close)
	f.manifest = testManifest(f.server.URL, f.payload)
	hash := sha256.Sum256(f.payload)
	f.manifest.Assets[0].SHA256 = hex.EncodeToString(hash[:])
	f.manifest.Assets[0].UnpackedBytes = unpacked
	f.client = testClient(t, f.server, install)
	f.client.freeSpace = func(string) (uint64, error) { return math.MaxUint64, nil }
	if r := f.client.Check(context.Background()); r.Status != Available {
		t.Fatalf("check: %+v", r)
	}
	return f
}

func TestDSHAMismatchNothingInstalledTypedError(t *testing.T) {
	f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
	f.client.release.Asset.SHA256 = strings.Repeat("0", 64)
	staged, err := f.client.Download(context.Background(), nil)
	if staged != nil || err == nil || err.Code != CodeHash || err.ExpectedHash != strings.Repeat("0", 64) || err.ReceivedHash == "" || err.Partial != "removed" || err.URL != f.manifest.Assets[0].URL {
		t.Fatalf("staged=%v failure=%+v", staged, err)
	}
	entries, e := os.ReadDir(f.client.cfg.InstallDir)
	if e != nil || len(entries) != 0 {
		t.Fatalf("install changed: %v %v", entries, e)
	}
	if exists(filepath.Join(f.client.assetRoot(f.client.release.Asset), "archive.zip")) {
		t.Fatal("hash mismatch marked complete")
	}
}

func TestESpaceCheckCountsExistingPartialFiles(t *testing.T) {
	f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
	if err := os.WriteFile(filepath.Join(f.client.cfg.InstallDir, "old"), []byte("rollback"), 0600); err != nil {
		t.Fatal(err)
	}
	a := f.client.release.Asset
	root := f.client.assetRoot(a)
	if err := ensureDir(root); err != nil {
		t.Fatal(err)
	}
	partial := len(f.payload) / 3
	if err := os.WriteFile(filepath.Join(root, "archive.part"), f.payload[:partial], 0600); err != nil {
		t.Fatal(err)
	}
	want := uint64(a.Size - int64(partial) + a.UnpackedBytes + 8)
	f.client.freeSpace = func(string) (uint64, error) { return want - 1, nil }
	p, err := f.client.PlanSpace(a, int64(partial))
	if err == nil || err.Code != CodeSpace || p.AdditionalBytes != want || p.ExistingPartialBytes != int64(partial) || p.RollbackBytes != 8 {
		t.Fatalf("plan=%+v failure=%v", p, err)
	}
	if _, err := f.client.Download(context.Background(), nil); err == nil || err.Code != CodeSpace || err.Partial != "kept" || f.requests != 0 {
		t.Fatalf("failure=%v requests=%d", err, f.requests)
	}
	f.client.freeSpace = func(string) (uint64, error) { return want, nil }
	if _, err := f.client.Download(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	if f.ranges[0] != fmt.Sprintf("bytes=%d-", partial) {
		t.Fatalf("ranges=%v", f.ranges)
	}
}

func TestFResumeAfterInterruptedDownload(t *testing.T) {
	f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
	f.interrupt = true
	_, err := f.client.Download(context.Background(), nil)
	if err == nil || err.Code != CodeNetwork || err.Partial != "kept" || err.ReceivedBytes <= 0 {
		t.Fatalf("failure=%+v", err)
	}
	partial := err.ReceivedBytes
	var phases []string
	staged, err := f.client.Download(context.Background(), func(p Progress) { phases = append(phases, p.Phase) })
	if err != nil || staged == nil {
		t.Fatalf("staged=%v failure=%v", staged, err)
	}
	if f.requests != 2 || f.ranges[1] != fmt.Sprintf("bytes=%d-", partial) {
		t.Fatalf("requests=%d ranges=%v", f.requests, f.ranges)
	}
	if phases[len(phases)-1] != "ready" {
		t.Fatalf("phases=%v", phases)
	}
	if data, e := os.ReadFile(filepath.Join(staged.Directory, filepath.FromSlash(guiRelative))); e != nil || string(data) != "gui" {
		t.Fatalf("staged GUI %q %v", data, e)
	}
}

func TestResumeRestartsIfRangeIgnoredAndRejectsBadRange(t *testing.T) {
	for _, mode := range []string{"ignored", "bad", "corrupt_partial"} {
		t.Run(mode, func(t *testing.T) {
			f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
			f.interrupt = true
			_, err := f.client.Download(context.Background(), nil)
			if err == nil {
				t.Fatal("expected interruption")
			}
			switch mode {
			case "ignored":
				f.ignoreRange = true
			case "bad":
				f.badRange = true
			case "corrupt_partial":
				p := filepath.Join(f.client.assetRoot(f.client.release.Asset), "archive.part")
				data, e := os.ReadFile(p)
				if e != nil {
					t.Fatal(e)
				}
				data[0] ^= 1
				if e := os.WriteFile(p, data, 0600); e != nil {
					t.Fatal(e)
				}
			}
			_, err = f.client.Download(context.Background(), nil)
			if mode == "ignored" && err != nil {
				t.Fatal(err)
			}
			if mode == "bad" && (err == nil || err.Code != CodeResume) {
				t.Fatalf("%v", err)
			}
			if mode == "corrupt_partial" && (err == nil || err.Code != CodeHash) {
				t.Fatalf("%v", err)
			}
		})
	}
}

func TestDownloadCancelRetainsResumablePartial(t *testing.T) {
	f := servePackage(t, "", fixtureEntries([]byte("gui"), []byte("helper")))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := f.client.Download(ctx, func(p Progress) {
		if p.Phase == "downloading" && p.ReceivedBytes > 0 {
			cancel()
		}
	})
	if err == nil || err.Code != CodeCanceled || err.Partial != "kept" {
		t.Fatalf("%+v", err)
	}
	if _, err := f.client.Download(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
}

func TestStageRejectsZipSlipTopFolderAndLongPaths(t *testing.T) {
	for _, bad := range []string{"../escape", "bubo-v0.1.2-windows-x64-gfx1151/../../escape", "other/file", "bubo-v0.1.2-windows-x64-gfx1151/bin/CON", "bubo-v0.1.2-windows-x64-gfx1151/bin/a.", "bubo-v0.1.2-windows-x64-gfx1151/" + strings.Repeat("x", 250)} {
		t.Run(fmt.Sprint(len(bad), bad[:min(len(bad), 12)]), func(t *testing.T) {
			entries := append(fixtureEntries([]byte("gui"), []byte("helper")), archiveEntry{bad, []byte("bad")})
			f := servePackage(t, "", entries)
			if staged, err := f.client.Download(context.Background(), nil); staged != nil || err == nil || err.Code != CodeArchive {
				t.Fatalf("staged=%v failure=%v", staged, err)
			}
			if exists(filepath.Join(filepath.Dir(f.client.cfg.InstallDir), "escape")) {
				t.Fatal("escaped staging")
			}
		})
	}
}
