//go:build windows

package replay

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

func TestV12CaptureIntegrityAndRuntimeReuse(t *testing.T) {
	root := filepath.Join("..", "recordings", "v12")
	raw, err := os.ReadFile(filepath.Join(root, "metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Commit     string `json:"candidate_commit"`
		Executable string `json:"executable_sha256"`
		Commands   []struct {
			File  string
			Bytes int    `json:"stdout_bytes"`
			Hash  string `json:"stdout_sha256"`
		} `json:"commands"`
	}
	if err := json.Unmarshal(raw, &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Commit != "1a0cd4305a87a527c6bac453910349d754e5944b" || metadata.Executable != "36393435c2db5ca9ae54c1434f74a8f62c0ea0de2718de46b5a7f399ed647ad2" || len(metadata.Commands) != 72 {
		t.Fatal("merged CLI provenance changed")
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
				t.Fatal("capture bytes or hash changed")
			}
		})
	}
	for _, name := range []string{"ready", "refresh"} {
		t.Run(name, func(t *testing.T) {
			file, err := os.Open(filepath.Join(root, "plan-runtime-"+name+".jsonl"))
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			scanner := bufio.NewScanner(file)
			scanner.Buffer(make([]byte, 4096), 1<<20)
			parts, sources, approval := 0, 0, false
			for scanner.Scan() {
				var event struct {
					Type    string
					Payload struct {
						Ready   bool
						Kind    string   `json:"record_kind"`
						States  []string `json:"file_states"`
						Sources []json.RawMessage
					}
				}
				if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
					t.Fatal(err)
				}
				if event.Type == "approval_required" {
					approval = true
				}
				if event.Type != "plan" {
					continue
				}
				parts++
				if event.Payload.Ready != (name == "ready") || len(event.Payload.States) != 3 {
					t.Fatal("runtime readiness/state count changed")
				}
				for _, state := range event.Payload.States {
					if state != "reuse" {
						t.Fatal("payload would be downloaded again")
					}
				}
				sources += len(event.Payload.Sources)
			}
			if err := scanner.Err(); err != nil {
				t.Fatal(err)
			}
			if parts != 76 || sources != 3 || approval != (name == "refresh") {
				t.Fatalf("plan parts=%d sources=%d approval=%v", parts, sources, approval)
			}
		})
	}
}

func TestV12HistoricalFixturesOnlyReplaceCapturedMetadata(t *testing.T) {
	root := filepath.Join("..", "recordings")
	read := func(file string) map[string]any {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(root, file))
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err := json.Unmarshal(raw, &value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	current := read("v12/status-main.json")["payload"].(map[string]any)["capabilities"].(map[string]any)["profiles"]
	modelNames := map[string]any{}
	for _, raw := range read("v12/models-list-empty.json")["payload"].(map[string]any)["catalogue"].([]any) {
		row := raw.(map[string]any)
		modelNames[row["id"].(string)] = row["display_name"]
	}
	allowed := map[string]any{}
	for _, raw := range read("v12/switches.json")["payload"].(map[string]any)["switches"].([]any) {
		row := raw.(map[string]any)
		allowed[row["name"].(string)] = row["allowed"]
	}
	var restoreConstraints func(*testing.T, any, any)
	restoreConstraints = func(t *testing.T, before, after any) {
		t.Helper()
		switch value := after.(type) {
		case map[string]any:
			previous, ok := before.(map[string]any)
			if !ok {
				t.Fatal("historical object shape changed")
			}
			if id, ok := value["id"].(string); ok && modelNames[id] != nil {
				if _, exists := value["display_name"]; exists {
					if !reflect.DeepEqual(value["display_name"], modelNames[id]) {
						t.Fatal("model label differs from captured catalogue", id)
					}
					value["display_name"] = previous["display_name"]
				}
			}
			if name, ok := value["name"].(string); ok && allowed[name] != nil {
				if _, exists := value["allowed"]; exists {
					if !reflect.DeepEqual(value["allowed"], allowed[name]) {
						t.Fatal("named constraints differ from merged CLI", name)
					}
					value["allowed"] = previous["allowed"]
				}
			}
			for name, entry := range value {
				if setting, ok := entry.(map[string]any); ok && allowed[name] != nil {
					_, named := setting["name"]
					if _, exists := setting["allowed"]; exists && !named {
						if !reflect.DeepEqual(setting["allowed"], allowed[name]) {
							t.Fatal("constraints differ from merged CLI", name)
						}
						setting["allowed"] = previous[name].(map[string]any)["allowed"]
					}
				}
				restoreConstraints(t, previous[name], entry)
			}
		case []any:
			previous, ok := before.([]any)
			if !ok || len(previous) != len(value) {
				t.Fatal("historical list shape changed")
			}
			for i, entry := range value {
				restoreConstraints(t, previous[i], entry)
			}
		}
	}
	raw, err := os.ReadFile(filepath.Join(root, "v12", "derived-statuses.metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		File, Source         string
		SourceHash           string `json:"source_sha256"`
		FixtureHash          string `json:"fixture_sha256"`
		IdentityReplacements []struct {
			Path   []string `json:"path"`
			Before string   `json:"before"`
			After  string   `json:"after"`
		} `json:"identity_replacements"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil || len(rows) != 7 {
		t.Fatalf("derived provenance: %v (%d)", err, len(rows))
	}
	for _, row := range rows {
		t.Run(row.File, func(t *testing.T) {
			for file, want := range map[string]string{row.Source: row.SourceHash, "v12/" + row.File: row.FixtureHash} {
				data, err := os.ReadFile(filepath.Join(root, file))
				if err != nil {
					t.Fatal(err)
				}
				hash := sha256.Sum256(data)
				if hex.EncodeToString(hash[:]) != want {
					t.Fatal("provenance hash mismatch", file)
				}
			}
			original, derived := read(row.Source), read("v12/"+row.File)
			for _, replacement := range row.IdentityReplacements {
				var before, after any = original, derived
				for _, key := range replacement.Path[:len(replacement.Path)-1] {
					if object, ok := before.(map[string]any); ok {
						before = object[key]
						after = after.(map[string]any)[key]
					} else {
						index, err := strconv.Atoi(key)
						if err != nil {
							t.Fatal(err)
						}
						before = before.([]any)[index]
						after = after.([]any)[index]
					}
				}
				key := replacement.Path[len(replacement.Path)-1]
				beforeObject, afterObject := before.(map[string]any), after.(map[string]any)
				if beforeObject[key] != replacement.Before || afterObject[key] != replacement.After {
					t.Fatal("derived identity differs from its exact replacement ledger")
				}
				afterObject[key] = beforeObject[key]
			}
			currentStatus := read("v12/status-main.json")
			if !reflect.DeepEqual(derived["product"], currentStatus["product"]) || !reflect.DeepEqual(derived["payload"].(map[string]any)["versions"], currentStatus["payload"].(map[string]any)["versions"]) {
				t.Fatal("derived product differs from final CLI")
			}
			derived["product"] = original["product"]
			derived["payload"].(map[string]any)["versions"] = original["payload"].(map[string]any)["versions"]
			caps := derived["payload"].(map[string]any)["capabilities"].(map[string]any)
			if !reflect.DeepEqual(caps["profiles"], current) {
				t.Fatal("profiles differ from merged CLI")
			}
			caps["profiles"] = original["payload"].(map[string]any)["capabilities"].(map[string]any)["profiles"]
			restoreConstraints(t, original, derived)
			if !reflect.DeepEqual(original, derived) {
				t.Fatal("historical observation changed beyond copied profile/constraint/model-name metadata")
			}
		})
	}
}
