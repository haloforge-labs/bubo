//go:build windows

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestMVPRuntimePackageEntry(t *testing.T) {
	for _, fault := range []string{"cli-hash", "engine-hash", "missing-engine-hash", "duplicate-cli-hash", "valid-but-locked", "int-engine-hash", "int-wrong-folder", "int-valid-but-locked"} {
		t.Run(fault, func(t *testing.T) {
			root := t.TempDir()
			runtime := filepath.Join(root, "runtime")
			if err := os.MkdirAll(filepath.Join(runtime, "bin"), 0o700); err != nil {
				t.Fatal(err)
			}
			type member struct {
				Path   string `json:"path"`
				SHA256 string `json:"sha256"`
			}
			var members []member
			for _, name := range []string{"bubo.exe", "bubo-engine.dll"} {
				// Synthetic, non-executable bytes: no real CLI/runtime is copied.
				data := []byte("R2 runtime provenance fixture " + name)
				if err := os.WriteFile(filepath.Join(runtime, "bin", name), data, 0o600); err != nil {
					t.Fatal(err)
				}
				hash := sha256.Sum256(data)
				members = append(members, member{"bin/" + name, hex.EncodeToString(hash[:])})
			}
			wantFiles := append([]member(nil), members...)
			wantRefusal := "Runtime package SHA256 mismatch for bin/bubo.exe"
			switch fault {
			case "cli-hash":
				members[0].SHA256 = strings.Repeat("0", 64)
			case "engine-hash":
				members[1].SHA256 = strings.Repeat("0", 64)
				wantRefusal = "Runtime package SHA256 mismatch for bin/bubo-engine.dll"
			case "missing-engine-hash":
				members = members[:1]
				wantRefusal = "exactly one valid SHA256 for bin/bubo-engine.dll"
			case "duplicate-cli-hash":
				members = append(members, members[0])
				wantRefusal = "exactly one valid SHA256 for bin/bubo.exe"
			}
			const commit = "0123456789abcdef0123456789abcdef01234567"
			wantCommit, wantProvenance := commit, "BUILD-MANIFEST.json"
			if strings.HasPrefix(fault, "int-") {
				// Integration layout: a copied base BUILD-MANIFEST whose hashes no
				// longer match, plus MANIFEST-int.txt for the replaced CLI/engine.
				const intCommit = "89abcdef0123456789abcdef0123456789abcdef"
				wantCommit, wantProvenance = intCommit, "MANIFEST-int.txt"
				folder, engine := filepath.Base(runtime), strings.ToUpper(wantFiles[1].SHA256)
				for i := range members {
					members[i].SHA256 = strings.Repeat("0", 64)
				}
				switch fault {
				case "int-engine-hash":
					engine = strings.Repeat("A", 64)
					wantRefusal = "Runtime package SHA256 mismatch for bin/bubo-engine.dll"
				case "int-wrong-folder":
					folder = "bubo-int-other"
					wantRefusal = "MANIFEST-int.txt must name this folder"
				}
				text := folder + ` = C:\base with bubo.exe + bubo-engine.dll built from C:\tree (` + intCommit + " fixture merge)\r\n" +
					"bubo.exe sha256 " + strings.ToUpper(wantFiles[0].SHA256) + "\r\n" +
					"bubo-engine.dll sha256 " + engine + "\r\nbuilt fixture\r\n"
				if err := os.WriteFile(filepath.Join(runtime, "MANIFEST-int.txt"), []byte(text), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			manifest, err := json.Marshal(struct {
				Source struct {
					Commit string `json:"commit"`
				} `json:"source"`
				Files []member `json:"files"`
			}{Source: struct {
				Commit string `json:"commit"`
			}{commit}, Files: members})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(runtime, "BUILD-MANIFEST.json"), manifest, 0o600); err != nil {
				t.Fatal(err)
			}
			modelManifest := filepath.Join(root, "bubo-package.json")
			if err := os.WriteFile(modelManifest, []byte("{}"), 0o600); err != nil {
				t.Fatal(err)
			}
			out := filepath.Join(root, "out")
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			// Require both user locks with an owner absent from each file.
			// The matching-hash case proves the entry/driver handoff, then must
			// refuse before any process scan, GPU query or socket operation.
			owner := "r2-fixture-" + filepath.Base(root)
			for _, lock := range []string{defaultLock, defaultBuildLock} {
				if data, err := os.ReadFile(lock); err == nil && strings.Contains(string(data), owner) {
					t.Fatal("fixture owner unexpectedly present in user lock")
				}
			}
			cmd := exec.CommandContext(ctx, "pwsh", "-NoProfile", "-File", "run-mvp.ps1", "-RuntimePackage", runtime, "-Manifest", modelManifest, "-Out", out, "-AllowLockOwner", owner)
			cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
			console, err := cmd.CombinedOutput()
			if ctx.Err() != nil || err == nil {
				t.Fatalf("expected refusal, error=%v context=%v\n%s", err, ctx.Err(), console)
			}
			if !strings.HasSuffix(fault, "valid-but-locked") {
				if !strings.Contains(string(console), wantRefusal) {
					t.Fatalf("missing %q\n%s", wantRefusal, console)
				}
				if _, err := os.Stat(out); !os.IsNotExist(err) {
					t.Fatalf("hash refusal created output/builds: %v", err)
				}
				return
			}
			if exit, ok := err.(*exec.ExitError); !ok || exit.ExitCode() != ExitRefused {
				t.Fatalf("lock refusal exit code lost: %v\n%s", err, console)
			}
			const label = "pre-package smoke (CLI + supervisor on the GPU runtime; GUI exe not involved)"
			for _, name := range []string{"entry.json", filepath.Join("result", "report.json")} {
				data, err := os.ReadFile(filepath.Join(out, name))
				if err != nil {
					t.Fatalf("%s: %v\n%s", name, err, console)
				}
				var evidence struct {
					Qualification string `json:"qualification"`
					Overall       string `json:"overall"`
					Runtime       struct {
						Folder       string   `json:"folder"`
						SourceCommit string   `json:"source_commit"`
						Provenance   string   `json:"provenance_file"`
						ManifestHash string   `json:"build_manifest_sha256"`
						Files        []member `json:"files"`
					} `json:"runtime_package"`
				}
				if err := json.Unmarshal(data, &evidence); err != nil {
					t.Fatal(err)
				}
				hash := sha256.Sum256(manifest)
				if evidence.Qualification != label || evidence.Runtime.Folder != runtime || evidence.Runtime.SourceCommit != wantCommit || evidence.Runtime.Provenance != wantProvenance || evidence.Runtime.ManifestHash != hex.EncodeToString(hash[:]) || len(evidence.Runtime.Files) != 2 {
					t.Fatalf("incorrect provenance: %s", data)
				}
				for i, file := range evidence.Runtime.Files {
					if file != wantFiles[i] {
						t.Fatalf("incorrect binary hash: %+v", file)
					}
				}
				if name != "entry.json" && evidence.Overall != "REFUSED" {
					t.Fatalf("lock refusal lost: %s", data)
				}
			}
			if !strings.Contains(string(console), "REFUSED: GPU lock") || !strings.Contains(string(console), label) {
				t.Fatalf("missing lock refusal/scope label\n%s", console)
			}
			if !strings.Contains(string(console), "COUNTS: steps=0 checks=0 passed=0 failed=0; verdict=REFUSED") {
				t.Fatalf("missing refusal counts\n%s", console)
			}
			if _, err := os.Stat(filepath.Join(out, "tools")); !os.IsNotExist(err) {
				t.Fatalf("entry left temporary binaries: %v", err)
			}
		})
	}
}
