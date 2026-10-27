//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestEngineDefaultsRemainAbsent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	for _, source := range []string{"missing", "saved"} {
		t.Run(source, func(t *testing.T) {
			s, warning := LoadSettings(path)
			if warning != "" || s.Port != 0 || s.Profile != "" || s.Context != 0 || s.Sessions != 0 || len(startArgs(s)) != 0 {
				t.Fatalf("invented an engine override: %+v, %s", s, warning)
			}
			if err := SaveSettings(path, s); err != nil {
				t.Fatal(err)
			}
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var doc map[string]json.RawMessage
			if err := json.Unmarshal(b, &doc); err != nil {
				t.Fatal(err)
			}
			for _, key := range []string{"port", "profile", "context", "sessions"} {
				if _, exists := doc[key]; exists {
					t.Fatal("stored an absent override", key)
				}
			}
		})
	}
}

func TestEngineOverridesAndAdditiveCodesSurviveSettings(t *testing.T) {
	s := DefaultSettings()
	s.Profile, s.Context, s.Sessions, s.Port = "future-cli-profile", 17, 2, 8082
	s.LastError = &Failure{Code: "future_cli_error", Source: "cli", Remedy: "private path"}
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	got, warning := LoadSettings(path)
	if warning != "" || got.Profile != s.Profile || got.Context != s.Context || got.Sessions != s.Sessions || got.Port != s.Port || got.LastError.Code != s.LastError.Code || strings.Contains(got.LastError.Remedy, "private") {
		t.Fatalf("override/code lost: %+v, %s", got, warning)
	}
	want := []string{"--profile", s.Profile, "--context", "17", "--sessions", "2"}
	if !reflect.DeepEqual(startArgs(got), want) {
		t.Fatalf("%v != %v", startArgs(got), want)
	}
}

func TestReadyBindingWithCLIDefaultPort(t *testing.T) {
	endpoint := "http://127.0.0.1:19099/cli-chosen-path"
	if !matchesBind(&endpoint, "127.0.0.1", 0) || !matchesBind(&endpoint, "127.0.0.1", 19099) || matchesBind(&endpoint, "127.0.0.1", 19098) || matchesBind(&endpoint, "0.0.0.0", 0) {
		t.Fatal("network binding verification ignored CLI/default or requested binding")
	}
}

func TestCLIRejectsInvalidPortAndBackendCanRetry(t *testing.T) {
	b := backendFor(t, "reject-port")
	s := b.settings
	s.Port = -1
	if err := b.Save(s); err != nil {
		t.Fatal("GUI revalidated an engine setting", err)
	}
	reply(t, b.Action("start", s, ""))
	v := waitBackend(t, b, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	if v.LastError == nil || v.LastError.Code != "invalid_arguments" || v.LastError.Source != "cli" {
		t.Fatalf("CLI validation was not preserved: %+v", v.LastError)
	}
	s.Port = 19091
	if err := b.Save(s); err != nil {
		t.Fatal(err)
	}
	reply(t, b.Action("start", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
	reply(t, b.Action("stop", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped })
}
