//go:build windows

package replay

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestS24CaptureIntegrity(t *testing.T) {
	// Validate the shipped catalogue captures; E23's lower-memory records are retired.
	root := filepath.Join("..", "recordings", "v12")
	data, err := os.ReadFile(filepath.Join(root, "metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Commit     string `json:"candidate_commit"`
		Executable string `json:"executable_sha256"`
		Commands   []struct {
			File  string `json:"file"`
			Bytes int    `json:"stdout_bytes"`
			Hash  string `json:"stdout_sha256"`
		} `json:"commands"`
	}
	if err := json.Unmarshal(data, &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Commit != "1a0cd4305a87a527c6bac453910349d754e5944b" || metadata.Executable != "36393435c2db5ca9ae54c1434f74a8f62c0ea0de2718de46b5a7f399ed647ad2" || len(metadata.Commands) != 72 {
		t.Fatalf("capture provenance: %+v", metadata)
	}
	seen := map[string]bool{}
	for _, command := range metadata.Commands {
		t.Run(command.File, func(t *testing.T) {
			if seen[command.File] {
				t.Fatal("duplicate capture")
			}
			seen[command.File] = true
			data, err := os.ReadFile(filepath.Join(root, command.File))
			if err != nil {
				t.Fatal(err)
			}
			hash := sha256.Sum256(data)
			if len(data) != command.Bytes || hex.EncodeToString(hash[:]) != command.Hash {
				t.Fatal("capture bytes or SHA-256 changed")
			}
		})
	}
	data, err = os.ReadFile(filepath.Join(root, "models-with-warnings.metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var derived struct {
		Bytes int    `json:"fixture_bytes"`
		Hash  string `json:"fixture_sha256"`
	}
	if err := json.Unmarshal(data, &derived); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(filepath.Join(root, "models-with-warnings.json"))
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(data)
	if len(data) != derived.Bytes || hex.EncodeToString(hash[:]) != derived.Hash {
		t.Fatal("derived warnings fixture changed")
	}
}
