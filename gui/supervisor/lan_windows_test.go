//go:build windows

package supervisor

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/danieljoos/wincred"
)

type fakeStore struct {
	mu                  sync.Mutex
	m                   map[string]string
	writes, deletes     int
	failRead, failWrite bool
	dropWrites          bool // accepts a write but stores nothing
	failDelete          bool
}

func newFakeStore() *fakeStore { return &fakeStore{m: map[string]string{}} }
func (f *fakeStore) Read(t string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failRead {
		return "", errors.New("store offline")
	}
	v, ok := f.m[t]
	if !ok {
		return "", ErrCredentialNotFound
	}
	return v, nil
}
func (f *fakeStore) Write(t, s string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failWrite {
		return errors.New("store offline")
	}
	f.writes++
	if !f.dropWrites {
		f.m[t] = s
	}
	return nil
}
func (f *fakeStore) Delete(t string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failDelete {
		return errors.New("store offline")
	}
	if _, ok := f.m[t]; !ok {
		return ErrCredentialNotFound
	}
	f.deletes++
	delete(f.m, t)
	return nil
}
func (f *fakeStore) get() string { f.mu.Lock(); defer f.mu.Unlock(); return f.m[KeyTarget] }

var keyShape = regexp.MustCompile(`^[0-9a-f]{64}$`)

// lanSupervisor builds a supervisor with a settings file and a fake store.
func lanSupervisor(t *testing.T, mode string, lan bool, store CredentialStore) (*Supervisor, Config) {
	t.Helper()
	c := configFor(t, mode)
	c.SettingsPath = filepath.Join(filepath.Dir(c.LogDir), "settings.json")
	c.Keys = NewKeyManager(store)
	st := DefaultSettings()
	st.LanAccess = lan
	if err := SaveSettings(c.SettingsPath, st); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.finished:
			return
		default:
		}
		select {
		case <-s.Exit():
		case <-time.After(5 * time.Second):
			t.Error("Exit did not finish")
		}
	})
	return s, c
}
func fakeArgv(t *testing.T, c Config) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(filepath.Dir(c.LogDir), "argv.json"))
	if err != nil {
		t.Fatal(err)
	}
	var a []string
	if err = json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	return a
}

// fakeEnv is the fake CLI's record of the key variable it received: presence,
// how many entries, and the SHA-256 of the value (never the value).
type envSeen struct {
	Present bool   `json:"present"`
	Count   int    `json:"count"`
	SHA256  string `json:"sha256"`
}

func fakeEnv(t *testing.T, c Config) envSeen {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(filepath.Dir(c.LogDir), "env.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]envSeen
	if err = json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m["BUBO_API_KEY"]
}
func sum(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

// noKeyInArgv fails when the child command line holds the flag or the key.
func noKeyInArgv(t *testing.T, argv []string, key string) {
	t.Helper()
	for _, a := range argv {
		if strings.HasPrefix(a, "--api-key") || (key != "" && strings.Contains(a, key)) {
			t.Fatalf("the child command line carries the API key or its flag: %v", argv)
		}
	}
}
func count(a []string, v string) int {
	n := 0
	for _, x := range a {
		if x == v {
			n++
		}
	}
	return n
}
func argAfter(a []string, name string) string {
	for i, x := range a {
		if x == name && i+1 < len(a) {
			return a[i+1]
		}
	}
	return ""
}

func TestLoopbackStartHasNoAPIKey(t *testing.T) {
	t.Setenv("BUBO_API_KEY", "inherited-decoy-key") // must not reach the child
	store := newFakeStore()
	s, c := lanSupervisor(t, "split", false, store)
	reply(t, s.Start())
	waitState(t, s, Ready)
	argv := fakeArgv(t, c)
	noKeyInArgv(t, argv, "inherited-decoy-key")
	if argAfter(argv, "--host") != "127.0.0.1" {
		t.Fatalf("loopback argv: %v", argv)
	}
	if e := fakeEnv(t, c); e.Present || e.Count != 0 {
		t.Fatalf("loopback child environment holds a key: %+v", e)
	}
	if store.writes != 0 || len(store.m) != 0 {
		t.Fatal("loopback start created a key")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}

func TestLANStartGeneratesOnceAndReusesKey(t *testing.T) {
	store := newFakeStore()
	s, c := lanSupervisor(t, "split", true, store)
	reply(t, s.Start())
	v := waitState(t, s, Ready)
	argv := fakeArgv(t, c)
	key := store.get()
	if !keyShape.MatchString(key) || argAfter(argv, "--host") != "0.0.0.0" {
		t.Fatalf("LAN argv shape: host %q, key shape ok %v", argAfter(argv, "--host"), keyShape.MatchString(key))
	}
	noKeyInArgv(t, argv, key)
	if e := fakeEnv(t, c); !e.Present || e.Count != 1 || e.SHA256 != sum(key) {
		t.Fatalf("the key did not reach the child environment exactly once: %+v", e)
	}
	if store.get() != key || store.writes != 1 {
		t.Fatal("key was not stored exactly once")
	}
	if v.Instance == nil || v.Instance.Endpoint == nil || *v.Instance.Endpoint != "http://0.0.0.0:19091/v1" {
		t.Fatalf("LAN instance endpoint: %+v", v.Instance)
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.Start())
	waitState(t, s, Ready)
	noKeyInArgv(t, fakeArgv(t, c), key)
	if e := fakeEnv(t, c); e.SHA256 != sum(key) || e.Count != 1 || store.writes != 1 {
		t.Fatal("second start did not reuse the stored key")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}

// artifacts returns every text artifact the GUI keeps after a run.
func artifacts(t *testing.T, s *Supervisor, c Config) map[string]string {
	t.Helper()
	out := map[string]string{}
	snap, _ := json.Marshal(s.Snapshot())
	out["snapshot.json"] = string(snap)
	out["ring"] = strings.Join(s.Logs(), "\n")
	out["label"] = s.Snapshot().Label
	root := filepath.Dir(c.LogDir)
	filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		// argv.json is the fake CLI's own record of what it received: the
		// child side of the hand-over. Everything else is the GUI side.
		if filepath.Base(p) == "argv.json" {
			return nil
		}
		b, _ := os.ReadFile(p)
		out[strings.TrimPrefix(p, root)] = string(b)
		return nil
	})
	return out
}

func TestPlantedKeyNeverAppearsInAnyGUIArtifact(t *testing.T) {
	store := newFakeStore()
	planted := strings.Repeat("ab", 32)
	store.m[KeyTarget] = planted // the key the child will receive
	s, c := lanSupervisor(t, "leak-key", true, store)
	reply(t, s.Start())
	waitState(t, s, Ready)
	noKeyInArgv(t, fakeArgv(t, c), planted)
	if e := fakeEnv(t, c); e.SHA256 != sum(planted) || e.Count != 1 {
		t.Fatalf("fake CLI did not receive the planted key in its environment: %+v", e)
	}
	time.Sleep(150 * time.Millisecond) // let the log writer drain
	got := artifacts(t, s, c)
	masked := false
	for name, text := range got {
		if strings.Contains(text, planted) {
			t.Errorf("planted key found in %s", name)
		}
		if strings.Contains(text, KeyMask) {
			masked = true
		}
	}
	if !masked {
		t.Error("the leaking CLI line was never masked: the test did not exercise the scrubber")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	for name, text := range artifacts(t, s, c) {
		if strings.Contains(text, planted) {
			t.Errorf("planted key found in %s after stop", name)
		}
	}
}

func TestSettingsFileNeverHoldsKeyAndLANFlagPersists(t *testing.T) {
	store := newFakeStore()
	s, c := lanSupervisor(t, "split", true, store)
	reply(t, s.Start())
	waitState(t, s, Ready)
	b, _ := os.ReadFile(c.SettingsPath)
	if strings.Contains(string(b), store.get()) || !strings.Contains(string(b), `"lan_access": true`) {
		t.Fatalf("settings file: %s", b)
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}

func TestResetKeyChangesKey(t *testing.T) {
	store := newFakeStore()
	k := NewKeyManager(store)
	first, err := k.Ensure()
	if err != nil || !keyShape.MatchString(first) {
		t.Fatal(err, first)
	}
	if err = k.Reset(); err != nil {
		t.Fatal(err)
	}
	second, err := k.Reveal()
	if err != nil || second == first || !keyShape.MatchString(second) {
		t.Fatal("Reset did not change the key", err)
	}
	if store.get() != second {
		t.Fatal("old key still stored")
	}
}

func TestEnsureReturnsStoredKeyAndPresentNeverReturnsIt(t *testing.T) {
	store := newFakeStore()
	k := NewKeyManager(store)
	if k.Present() {
		t.Fatal("present before any key")
	}
	if _, err := k.Reveal(); !errors.Is(err, ErrNoKey) {
		t.Fatal("Reveal without a key:", err)
	}
	key, _ := k.Ensure()
	if !k.Present() {
		t.Fatal("not present after Ensure")
	}
	again, _ := k.Ensure()
	if again != key || store.writes != 1 {
		t.Fatal("Ensure created a second key")
	}
	if err := k.Delete(); err != nil || k.Present() {
		t.Fatal("Delete failed", err)
	}
	if err := k.Delete(); err != nil {
		t.Fatal("deleting a missing key must not fail:", err)
	}
}

func TestKeyStoreUnavailableLeavesLANOffAndStartsNothing(t *testing.T) {
	for name, store := range map[string]*fakeStore{
		"write fails":       {m: map[string]string{}, failWrite: true},
		"read fails":        {m: map[string]string{}, failRead: true},
		"write is not kept": {m: map[string]string{}, dropWrites: true},
	} {
		t.Run(name, func(t *testing.T) {
			s, c := lanSupervisor(t, "split", true, store)
			reply(t, s.Start())
			v := waitState(t, s, Failed)
			if v.Label != LANUnavailable {
				t.Fatalf("label %q", v.Label)
			}
			if _, err := os.Stat(filepath.Join(filepath.Dir(c.LogDir), "argv.json")); err == nil {
				t.Fatal("a server was launched without a stored key")
			}
			st, _ := LoadSettings(c.SettingsPath)
			if st.LanAccess {
				t.Fatal("lan_access stayed on")
			}
			b, _ := os.ReadFile(c.SettingsPath)
			if len(store.m) != 0 && !store.dropWrites {
				t.Fatal("something was stored")
			}
			_ = b
		})
	}
}

func TestSupervisorRejectsCallerAPIKeyArgument(t *testing.T) {
	for _, a := range []string{"--api-key", "--api-key=abc"} {
		if validateArgs([]string{a, "x"}) == nil {
			t.Fatalf("%s accepted", a)
		}
	}
}

func TestRestartNeededReportsLANChange(t *testing.T) {
	ep := "http://127.0.0.1:19091/v1"
	st := DefaultSettings()
	st.Port = 19091
	st.LanAccess = true
	got := RestartNeeded(st, &Instance{Endpoint: &ep}, nil, fixtureEvent(t, "switches").Value.(*Switches))
	if len(got) != 1 || got[0].Key != "lan_access" || got[0].Running != "127.0.0.1" || got[0].Next != "0.0.0.0" {
		t.Fatalf("%+v", got)
	}
	st.LanAccess = false
	if len(RestartNeeded(st, &Instance{Endpoint: &ep}, nil, fixtureEvent(t, "switches").Value.(*Switches))) != 0 {
		t.Fatal("reported a change with none")
	}
}

// One test against the real Windows Credential Manager. It uses a throwaway
// target and deletes it; it touches nothing else in the user's store.
func TestRealCredentialManagerRoundTrip(t *testing.T) {
	var r [6]byte
	rand.Read(r[:])
	target := "Bubo/test-" + hex.EncodeToString(r[:])
	store := WindowsCredentialStore{}
	t.Cleanup(func() { store.Delete(target) })
	if _, err := store.Read(target); !errors.Is(err, ErrCredentialNotFound) {
		t.Fatal("fresh target is not 'not found':", err)
	}
	if err := store.Write(target, "first-secret"); err != nil {
		t.Fatal(err)
	}
	if c, err := wincred.GetGenericCredential(target); err != nil || c.Persist != 2 {
		t.Fatalf("credential persistence must stay local-machine (2): %+v, %v", c, err)
	}
	if v, err := store.Read(target); err != nil || v != "first-secret" {
		t.Fatal("read back", v, err)
	}
	k := NewKeyManagerFor(store, target)
	if _, err := k.Reveal(); err != nil {
		t.Fatal(err)
	}
	if err := k.Reset(); err != nil {
		t.Fatal(err)
	}
	second, err := k.Reveal()
	if err != nil || !keyShape.MatchString(second) {
		t.Fatal("reset key shape", err)
	}
	if err = store.Delete(target); err != nil {
		t.Fatal(err)
	}
	if _, err = store.Read(target); !errors.Is(err, ErrCredentialNotFound) {
		t.Fatal("still present after delete:", err)
	}
	if err = store.Delete(target); !errors.Is(err, ErrCredentialNotFound) {
		t.Fatal("second delete:", err)
	}
}

func TestChildEnvironmentCarriesKeyOnceAndDropsInherited(t *testing.T) {
	parent := []string{"PATH=C:x", "bubo_api_key=stale", "BUBO_API_KEY=older", "A=b=c"}
	got := childEnvironment(parent, "", "")
	if len(got) != 2 || got[0] != parent[0] || got[1] != "A=b=c" {
		t.Fatalf("loopback environment: %v", got)
	}
	got = childEnvironment(parent, "new-key", "")
	n := 0
	for _, kv := range got {
		if strings.HasPrefix(strings.ToUpper(kv), "BUBO_API_KEY=") {
			n++
			if kv != "BUBO_API_KEY=new-key" {
				t.Fatalf("wrong key entry %q", kv)
			}
		}
	}
	if n != 1 || len(got) != 3 {
		t.Fatalf("LAN environment: %v", got)
	}
}

func TestCredentialConstructionPreservesTargetAndPersistence(t *testing.T) {
	c, err := keyCredential(KeyTarget, "test-secret")
	if err != nil || c.TargetName != "Bubo/serve-api-key" || c.Persist != 2 || c.UserName != "Bubo" || string(c.CredentialBlob) != "test-secret" {
		t.Fatalf("credential contract changed: %+v, %v", c, err)
	}
	for _, secret := range []string{"", strings.Repeat("x", 2561)} {
		if _, err := keyCredential(KeyTarget, secret); err == nil {
			t.Fatal("invalid credential size accepted")
		}
	}
}
