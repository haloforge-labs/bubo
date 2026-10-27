//go:build windows

package supervisor

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/windows/registry"
)

func loginTestKey(t *testing.T) (registry.Key, string) {
	t.Helper()
	var id [8]byte
	if _, err := rand.Read(id[:]); err != nil {
		t.Fatal(err)
	}
	path := `Software\Bubo-test-` + hex.EncodeToString(id[:])
	key, _, err := registry.CreateKey(registry.CURRENT_USER, path, registry.ALL_ACCESS)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		key.Close()
		if err := registry.DeleteKey(registry.CURRENT_USER, path); err != nil {
			t.Error(err)
		}
	})
	return key, path
}

func TestLoginRunAddRemovePreservesOtherValuesAndMigratesShortcut(t *testing.T) {
	key, path := loginTestKey(t)
	if err := key.SetStringValue("OtherApp", "untouched"); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	exe := filepath.Join(dir, "GUI with spaces 日本語.exe")
	if err := os.WriteFile(exe, nil, 0600); err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(dir, "Bubo.Control.lnk")
	other := filepath.Join(dir, "OtherApp.lnk")
	for _, p := range []string{old, other} {
		if err := os.WriteFile(p, []byte("shortcut"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 { // migration is idempotent; no marker file is needed
		if err := setLoginStartAt(true, exe, registry.CURRENT_USER, path, dir); err != nil {
			t.Fatal(err)
		}
		if got, _, err := key.GetStringValue(loginRunValue); err != nil || got != `"`+exe+`"` {
			t.Fatalf("launch command/arguments changed: %q, %v", got, err)
		}
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatal("legacy shortcut remains", err)
	}
	if err := setLoginStartAt(false, "", registry.CURRENT_USER, path, dir); err != nil {
		t.Fatal(err)
	}
	if _, _, err := key.GetStringValue(loginRunValue); !errors.Is(err, registry.ErrNotExist) {
		t.Fatal("Run value remains", err)
	}
	if got, _, err := key.GetStringValue("OtherApp"); err != nil || got != "untouched" {
		t.Fatal("another value changed", got, err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatal("another shortcut changed", err)
	}
}

func TestLoginMigrationFailureRestoresOwnRunValue(t *testing.T) {
	key, path := loginTestKey(t)
	if err := key.SetStringValue(loginRunValue, "old command"); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	// A nonempty directory cannot be removed as a shortcut. It models a
	// migration failure without writing anywhere outside the test folder.
	old := filepath.Join(dir, "Bubo.Control.lnk")
	if err := os.Mkdir(old, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(old, "keep"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if err := setLoginStartAt(false, "", registry.CURRENT_USER, path, dir); err == nil {
		t.Fatal("migration failure was hidden")
	}
	if got, _, err := key.GetStringValue(loginRunValue); err != nil || got != "old command" {
		t.Fatal("Run value was not restored", got, err)
	}
}
