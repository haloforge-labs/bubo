//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type memStore struct {
	mu                  sync.Mutex
	m                   map[string]string
	failWrite, failRead bool
}

func (s *memStore) Read(t string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failRead {
		return "", errors.New("offline")
	}
	v, ok := s.m[t]
	if !ok {
		return "", ErrCredentialNotFound
	}
	return v, nil
}
func (s *memStore) Write(t, v string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failWrite {
		return errors.New("offline")
	}
	s.m[t] = v
	return nil
}
func (s *memStore) Delete(t string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, t)
	return nil
}
func (s *memStore) key() string { s.mu.Lock(); defer s.mu.Unlock(); return s.m[KeyTarget] }

func lanApp(b *fakeBackend, store *memStore) *AppService {
	a := newTestApp(b, DefaultSettings(), nil, "")
	a.keys = NewKeyManager(store)
	return a
}
func viewJSON(t *testing.T, a *AppService) string {
	t.Helper()
	b, err := json.Marshal(a.GetView())
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestLoopbackViewHasNoKeyAndNoCopyWithKey(t *testing.T) {
	a := lanApp(&fakeBackend{}, &memStore{m: map[string]string{}})
	f := a.GetView()
	if f.Key.LANAccess || f.Key.Present || f.Key.CopyWithKey || f.Key.Mask != "" {
		t.Fatalf("%+v", f.Key)
	}
	if result := a.RevealKey(); result.Error == nil || result.Error.Remedy != ErrNoKey.Error() || result.Value != "" {
		t.Fatal("RevealKey without a key:", result)
	}
}

func TestTurningLANOnStoresKeyAndViewShowsOnlyThePresenceFlag(t *testing.T) {
	store := &memStore{m: map[string]string{}}
	b := &fakeBackend{}
	a := lanApp(b, store)
	s := DefaultSettings()
	s.LanAccess = true
	if err := patchTestPreferences(a, s); err != nil {
		t.Fatal(err)
	}
	key := store.key()
	if len(key) != 64 {
		t.Fatal("no key stored")
	}
	f := a.GetView()
	if !f.Key.LANAccess || !f.Key.Present || !f.Key.CopyWithKey || f.Key.Mask != KeyMask {
		t.Fatalf("%+v", f.Key)
	}
	if strings.Contains(viewJSON(t, a), key) {
		t.Fatal("key in frame")
	}
	saved, _ := json.Marshal(b.settings)
	if strings.Contains(string(saved), key) || !strings.Contains(string(saved), `"lan_access":true`) {
		t.Fatalf("saved settings: %s", saved)
	}
	if result := a.RevealKey(); result.Error != nil || result.Value != key {
		t.Fatal("explicit call did not return the key", result.Error)
	}
	// LAN off keeps the key and hides "Copy address and key".
	s.LanAccess = false
	if err := patchTestPreferences(a, s); err != nil {
		t.Fatal(err)
	}
	f = a.GetView()
	if f.Key.CopyWithKey || !f.Key.Present || store.key() != key {
		t.Fatalf("after LAN off: %+v", f.Key)
	}
}

func TestKeyStoreFailureLeavesLANOffWithMessage(t *testing.T) {
	store := &memStore{m: map[string]string{}, failWrite: true}
	b := &fakeBackend{}
	a := lanApp(b, store)
	s := DefaultSettings()
	s.LanAccess = true
	err := patchTestPreferences(a, s)
	if err == nil || err.Error() != "LAN access unavailable: cannot store the key" {
		t.Fatal(err)
	}
	if b.settings.LanAccess || a.GetView().Key.LANAccess {
		t.Fatal("LAN access stayed on")
	}
	if a.GetView().Key.Unavailable != "LAN access unavailable: cannot store the key" {
		t.Fatal("no message in the frame")
	}
	if len(store.m) != 0 {
		t.Fatal("a plaintext copy was stored")
	}
}

func TestResetKeyChangesKeyAndFlagsRunningLANServer(t *testing.T) {
	store := &memStore{m: map[string]string{}}
	ep := "http://0.0.0.0:18110/v1"
	b := &fakeBackend{view: Snapshot{State: Ready, Generation: 4, Instance: &Instance{Endpoint: &ep}}}
	a := lanApp(b, store)
	s := DefaultSettings()
	s.LanAccess = true
	if err := patchTestPreferences(a, s); err != nil {
		t.Fatal(err)
	}
	old := store.key()
	if a.GetView().Key.RestartNeeded {
		t.Fatal("restart flagged before a reset")
	}
	if err := actionTestError(a.ResetKey()); err != nil {
		t.Fatal(err)
	}
	if store.key() == old || len(store.key()) != 64 {
		t.Fatal("key did not change")
	}
	f := a.GetView()
	if !f.Key.RestartNeeded || !f.Key.LANActive {
		t.Fatalf("%+v", f.Key)
	}
	if strings.Contains(viewJSON(t, a), store.key()) || strings.Contains(viewJSON(t, a), old) {
		t.Fatal("key in frame")
	}
}

// This exercises the production backend and the sole read binding. A deliberately
// leaking child is scrubbed before text reaches any GUI-owned artifact.
func TestPlantedKeyInBackendTextNeverReachesView(t *testing.T) {
	planted := strings.Repeat("cd", 32)
	store := &memStore{m: map[string]string{KeyTarget: planted}}
	keys := NewKeyManager(store)
	dir := t.TempDir()
	t.Setenv("BUBO_FAKE_DIR", dir)
	t.Setenv("BUBO_FAKE_MODE", "leak-key")
	fixtures, err := filepath.Abs(filepath.Join("..", "..", "tests", "cli", "contract"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_FIXTURES", fixtures)
	s := DefaultSettings()
	s.Port, s.LanAccess = 19097, true
	path := filepath.Join(dir, "settings.json")
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	exe := fakeCLI(t)
	b, err := NewBackend(Config{Exe: exe, Port: s.Port, SettingsPath: path, Keys: keys, LogDir: filepath.Join(dir, "logs"), InitialPrepared: true, PollInterval: 40 * time.Millisecond}, s, exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if r := <-b.Action("exit", s, ""); r.Err != nil {
			t.Error(r.Err)
		}
	})
	a := newTestApp(b, s, nil, "")
	a.keys = keys
	// This test targets backend ingest/redaction. The fake has no planning
	// estimate; lifecycle availability is proved separately with explicit facts.
	if err := (<-b.Action("start", s, "")).Err; err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for b.Snapshot().State != Ready && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if b.Snapshot().State != Ready {
		t.Fatalf("not ready: %+v", b.Snapshot())
	}
	got := viewJSON(t, a)
	if strings.Contains(got, planted) {
		t.Fatal("key reached the View")
	}
	if !strings.Contains(got, KeyMask) {
		t.Fatal("child did not exercise ingest scrubbing")
	}

	if err := actionTestError(a.Stop()); err != nil {
		t.Fatal(err)
	}
	deadline = time.Now().Add(10 * time.Second)
	for b.Snapshot().State != Stopped && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if b.Snapshot().State != Stopped {
		t.Fatal("stop did not complete")
	}
}
