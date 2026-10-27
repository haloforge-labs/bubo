package updater

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testManifest(origin string, payload []byte) Manifest {
	return Manifest{
		SchemaVersion: 1, Product: "bubo", Channel: "stable", Version: "0.1.2", Tag: "v0.1.2",
		Commit: strings.Repeat("a", 40), Released: time.Date(2026, 10, 14, 0, 0, 0, 0, time.UTC), MinUpdatableFrom: "0.1.1",
		Assets:  []Asset{{Name: "bubo-v0.1.2-windows-x64-gfx1151.zip", Kind: "package", URL: origin + releasePath + "download/v0.1.2/bubo-v0.1.2-windows-x64-gfx1151.zip", Size: int64(len(payload)), SHA256: strings.Repeat("a", 64), UnpackedBytes: 100, TopFolder: "bubo-v0.1.2-windows-x64-gfx1151"}},
		Notes:   Notes{Summary: "A release", URL: origin + releasePath + "tag/v0.1.2"},
		Runtime: Runtime{ID: "fixture-runtime"}, Models: Models{NeedsPrepare: []string{}}, Advisories: []Advisory{}, RestartRequired: true,
	}
}

func testClient(t *testing.T, s *httptest.Server, install string) *Client {
	t.Helper()
	if install == "" {
		install = filepath.Join(t.TempDir(), "Bubo")
	}
	if err := os.MkdirAll(install, 0700); err != nil {
		t.Fatal(err)
	}
	pool := x509.NewCertPool()
	pool.AddCert(s.Certificate())
	c, f := New(Config{InstalledVersion: "0.1.1", InstallDir: install, ManifestURL: s.URL + releasePath + "latest/download/bubo-release.json", AllowedHosts: []string{"127.0.0.1"}, RootCAs: pool, HealthTimeout: 2 * time.Second})
	if f != nil {
		t.Fatal(f)
	}
	t.Cleanup(c.Close)
	return c
}

func TestHManifestFromNonAllowlistedHostRefused(t *testing.T) {
	var calls atomic.Int32
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1) }))
	defer s.Close()
	install := t.TempDir()
	_, f := New(Config{InstalledVersion: "0.1.1", InstallDir: install, ManifestURL: s.URL + releasePath + "latest/download/bubo-release.json"})
	if f == nil || f.Code != CodeURL || calls.Load() != 0 {
		t.Fatalf("failure=%v requests=%d", f, calls.Load())
	}
	// The constructor check above happens before any network access.
}

func TestCheckAndManifestValidation(t *testing.T) {
	var m Manifest
	var requests int
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Method != "GET" || r.Header.Get("User-Agent") != "bubo/0.1.1" || r.Header.Get("Cookie") != "" {
			t.Errorf("unexpected request: %v", r.Header)
		}
		json.NewEncoder(w).Encode(m)
	}))
	defer s.Close()
	c := testClient(t, s, "")
	m = testManifest(s.URL, []byte("archive"))
	r := c.Check(context.Background())
	if r.Status != Available || r.Available.Version != "0.1.2" || requests != 1 {
		t.Fatalf("result=%+v requests=%d", r, requests)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Manifest)
		code   Code
	}{
		{"schema", func(m *Manifest) { m.SchemaVersion = 2 }, CodeUnsupported},
		{"tag", func(m *Manifest) { m.Tag = "v9.0.0" }, CodeManifest},
		{"asset_path", func(m *Manifest) { m.Assets[0].URL = s.URL + "/wrong" }, CodeManifest},
		{"hash", func(m *Manifest) { m.Assets[0].SHA256 = "bad" }, CodeManifest},
		{"summary", func(m *Manifest) { m.Notes.Summary = "<script>" }, CodeManifest},
		{"range", func(m *Manifest) { m.Advisories = []Advisory{{Versions: ">0.0.0", Text: "bad"}} }, CodeManifest},
		{"intermediate", func(m *Manifest) {
			m.MinUpdatableFrom = "0.1.2"
			m.Version = "0.1.3"
			m.Tag = "v0.1.3"
			m.Assets[0].Name = "bubo-v0.1.3-windows-x64-gfx1151.zip"
			m.Assets[0].TopFolder = "bubo-v0.1.3-windows-x64-gfx1151"
			m.Assets[0].URL = s.URL + releasePath + "download/v0.1.3/" + m.Assets[0].Name
			m.Notes.URL = s.URL + releasePath + "tag/v0.1.3"
		}, CodeIntermediateRequired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m = testManifest(s.URL, []byte("archive"))
			tc.mutate(&m)
			r := c.Check(context.Background())
			if r.Status != Failed || r.Failure.Code != tc.code {
				t.Fatalf("%+v", r)
			}
		})
	}
}

func TestVersionAndAdvisoryRanges(t *testing.T) {
	for _, s := range []string{"1.2", "1.2.3.4", "1.2.3-beta", "01.2.3", "1.-2.3", "+1.2.3"} {
		if _, err := parseVersion(s); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
	v, _ := parseVersion("0.1.1")
	for _, s := range []string{"<=0.1.1", "0.1.1", "0.1.0-0.1.2"} {
		if ok, err := advisoryMatches(s, v); !ok || err != nil {
			t.Errorf("%s: %v %v", s, ok, err)
		}
	}
	a, _ := parseVersion("0.1.10")
	b, _ := parseVersion("0.1.9")
	if compare(a, b) != 1 {
		t.Fatal("lexicographic comparison")
	}
}

func TestRedirectHostAndHTTPSPolicy(t *testing.T) {
	for _, target := range []string{"https://localhost:1/not-contacted", "http://127.0.0.1:1/not-contacted"} {
		t.Run(target, func(t *testing.T) {
			s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, target, http.StatusFound) }))
			defer s.Close()
			c := testClient(t, s, "")
			r := c.Check(context.Background())
			if r.Status != Failed || r.Failure.Code != CodeURL {
				t.Fatalf("%+v", r)
			}
		})
	}

}

func TestCheckUpToDateDowngradeAndPersistedHistory(t *testing.T) {
	var m Manifest
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { json.NewEncoder(w).Encode(m) }))
	defer s.Close()
	c := testClient(t, s, "")
	m = testManifest(s.URL, []byte("archive"))
	if r := c.Check(context.Background()); r.Status != Available {
		t.Fatalf("%+v", r)
	}
	// A fresh client reads the same previously accepted release history.
	c2 := testClient(t, s, c.cfg.InstallDir)
	m.Released = m.Released.Add(-time.Hour)
	if r := c2.Check(context.Background()); r.Status != Failed || r.Failure.Code != CodeStale {
		t.Fatalf("%+v", r)
	}
	m = testManifest(s.URL, []byte("archive"))
	m.Released = m.Released.Add(time.Hour)
	c3 := testClient(t, s, "")
	c3.installed, _ = parseVersion("0.1.2")
	if r := c3.Check(context.Background()); r.Status != UpToDate {
		t.Fatalf("%+v", r)
	}
	c3.installed, _ = parseVersion("0.1.3")
	if r := c3.Check(context.Background()); r.Status != Failed || r.Failure.Code != CodeDowngrade {
		t.Fatalf("%+v", r)
	}
}

func TestIntermediateReleaseUsesOneExplicitManifestGET(t *testing.T) {
	var latest, intermediate Manifest
	var requests int
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if strings.Contains(r.URL.Path, "/latest/") {
			json.NewEncoder(w).Encode(latest)
		} else {
			json.NewEncoder(w).Encode(intermediate)
		}
	}))
	defer s.Close()
	intermediate = testManifest(s.URL, []byte("archive"))
	encoded, err := json.Marshal(intermediate)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(strings.ReplaceAll(string(encoded), "0.1.2", "0.1.3")), &latest); err != nil {
		t.Fatal(err)
	}
	latest.MinUpdatableFrom = "0.1.2"
	via := "0.1.2"
	latest.Via = &via
	latest.Released = latest.Released.Add(time.Hour)
	c := testClient(t, s, "")
	r := c.Check(context.Background())
	if r.Status != Failed || r.Failure.Code != CodeIntermediateRequired || requests != 1 {
		t.Fatalf("%+v requests=%d", r, requests)
	}
	cfg := c.cfg
	cfg.ManifestURL = r.Failure.URL
	hop, f := New(cfg)
	if f != nil {
		t.Fatal(f)
	}
	defer hop.Close()
	r = hop.Check(context.Background())
	if r.Status != Available || r.Available.Version != "0.1.2" || requests != 2 {
		t.Fatalf("%+v requests=%d", r, requests)
	}
	data, err := readRegular(filepath.Join(c.root, "seen.json"))
	if err != nil {
		t.Fatal(err)
	}
	var seen seenRelease
	if err := json.Unmarshal(data, &seen); err != nil || seen.Version != "0.1.3" {
		t.Fatalf("history lowered: %+v %v", seen, err)
	}
}

func TestFutureSchemaIsUnsupportedBeforeDecodingFutureFields(t *testing.T) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"schema_version":2,"runtime":"a future field shape"}`))
	}))
	defer s.Close()
	c := testClient(t, s, "")
	r := c.Check(context.Background())
	if r.Status != Failed || r.Failure.Code != CodeUnsupported {
		t.Fatalf("%+v", r)
	}
}
