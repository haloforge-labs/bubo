//go:build windows

package main

import "testing"

// The CLI's pi_snippet (src/cli/settings_support.hpp PiSnippet): loopback only;
// an authenticated loopback server names the env variable BUBO_API_KEY, never the
// dummy key; a LAN or wildcard bind has no snippet. The fake must say the same.
func TestFakePiSnippetFollowsTheCLIRules(t *testing.T) {
	key := func(i map[string]any) any {
		p := map[string]any{}
		uiStatus(p, i)
		snippet, ok := p["pi_snippet"].(map[string]any)
		if !ok {
			return nil
		}
		return snippet["providers"].(map[string]any)["bubo"].(map[string]any)["apiKey"]
	}
	local := map[string]any{"endpoint": "http://127.0.0.1:8086/v1", "served_model": "bubo", "authenticated": false}
	if got := key(local); got != "bubo-loopback" {
		t.Fatalf("keyless loopback: %v", got)
	}
	local["authenticated"] = true
	if got := key(local); got != "BUBO_API_KEY" {
		t.Fatalf("keyed loopback must name the env variable: %v", got)
	}
	lan := map[string]any{"endpoint": "http://0.0.0.0:8086/v1", "served_model": "bubo", "authenticated": true}
	if got := key(lan); got != nil {
		t.Fatalf("a LAN bind has no Pi snippet: %v", got)
	}
}
