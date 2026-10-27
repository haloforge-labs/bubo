//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// keyEnv is the variable through which the GUI hands the API key to serve.
const keyEnv = "BUBO_API_KEY"

func init() {
	if os.Getenv("BUBO_FAKE_RECORD_ALL_ENV") == "1" {
		recordEnv()
	}
}

// keyFromEnv returns every value the child environment holds for keyEnv.
func keyFromEnv() []string {
	var out []string
	for _, kv := range os.Environ() {
		if name, value, ok := strings.Cut(kv, "="); ok && strings.EqualFold(name, keyEnv) {
			out = append(out, value)
		}
	}
	return out
}

// recordEnv writes env.json: whether the key variable arrived, how often, and
// the SHA-256 of its value. It never writes the value itself, so a test can
// prove delivery without putting the key in an artifact.
func recordEnv() {
	values := keyFromEnv()
	rec := map[string]any{"present": len(values) > 0, "count": len(values)}
	if len(values) > 0 {
		sum := sha256.Sum256([]byte(values[0]))
		rec["sha256"] = hex.EncodeToString(sum[:])
	}
	type variable struct {
		Present bool   `json:"present"`
		Count   int    `json:"count"`
		SHA256  string `json:"sha256"`
	}
	seen := map[string]variable{}
	for _, kv := range os.Environ() {
		name, value, ok := strings.Cut(kv, "=")
		if !ok || !strings.HasPrefix(strings.ToUpper(name), "BUBO_") {
			continue
		}
		sum := sha256.Sum256([]byte(value))
		seen[name] = variable{Present: true, Count: 1, SHA256: hex.EncodeToString(sum[:])}
	}
	if b, err := json.Marshal(map[string]any{keyEnv: rec, "bubo_variables": seen}); err == nil {
		os.WriteFile(filepath.Join(dir, "env.json"), b, 0600)
	}
}

// leakEnvValue is the leak-key scenario's environment half: a misbehaving CLI
// that prints the key value it received.
func leakEnvValue() string {
	if v := keyFromEnv(); len(v) > 0 {
		return keyEnv + "=" + v[0]
	}
	return ""
}
