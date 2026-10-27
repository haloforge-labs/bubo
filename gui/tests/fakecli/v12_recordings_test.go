//go:build windows

package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestV12FakeDefaultsFollowMergedCLI(t *testing.T) {
	previous := os.Args
	defer func() { os.Args = previous }()
	os.Args = []string{"bubo.exe", "serve"}
	if got := effectiveFromArgs()["context"].(map[string]any)["effective"]; got != 200000 {
		t.Fatalf("default context: %v", got)
	}
	if got := idleSlot(0)["context_tokens"]; got != 200000 {
		t.Fatalf("default slot context: %v", got)
	}
	payload := map[string]any{}
	uiStatus(payload, map[string]any{"endpoint": "http://127.0.0.1:19939/v1"})
	if got := payload["memory_estimate"].(map[string]any)["context_tokens"]; got != 200000 {
		t.Fatalf("UI fixture default context: %v", got)
	}
	_, status := recordedOne("status-main.json")
	if !reflect.DeepEqual(capabilities, status["capabilities"]) {
		t.Fatal("fake capabilities differ from captured CLI capabilities")
	}
	os.Args = []string{"bubo.exe", "serve", "--profile", "stock"}
	if got := effectiveFromArgs()["profile"].(map[string]any)["effective"]; got != "vanilla" {
		t.Fatalf("legacy profile resolves to %v", got)
	}
}

func TestV12FakeServedNameUsesCapturedPackageOrFile(t *testing.T) {
	previous := os.Args
	defer func() { os.Args = previous }()
	_, listing := recordedOne("models-list-gui.json")
	for _, raw := range listing["prepared_models"].([]any) {
		model := raw.(map[string]any)
		path, ok := model["model_path"].(string)
		if !ok || path == "" {
			continue
		}
		os.Args = []string{"bubo.exe", "serve", "--model", path}
		if got := servedModelName(); got != model["id"] {
			t.Fatalf("prepared model %s: served %q", path, got)
		}
	}
	os.Args = []string{"bubo.exe", "serve", "--model", filepath.Join(t.TempDir(), "example.gguf")}
	if got := servedModelName(); got != "example" {
		t.Fatalf("loose model: %q", got)
	}
	os.Args = append(os.Args, "--served-model-name", "custom")
	if got := servedModelName(); got != "custom" {
		t.Fatalf("explicit name: %q", got)
	}
}
