package updater

import (
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func mustURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func TestManualProxyRules(t *testing.T) {
	for _, tc := range []struct {
		name             string
		enabled          bool
		server, override string
		target           string
		want             string
	}{
		{"off", false, "proxy:8080", "", "https://github.com/x", ""},
		{"one proxy for every scheme", true, "proxy:8080", "", "https://github.com/x", "http://proxy:8080"},
		{"per scheme picks https", true, "http=a:1;https=b:2;ftp=c:3", "", "https://github.com/x", "http://b:2"},
		{"per scheme without https goes direct", true, "http=a:1", "", "https://github.com/x", ""},
		{"scheme in the setting", true, "https://secure:443", "", "https://github.com/x", "https://secure:443"},
		{"empty server", true, " ", "", "https://github.com/x", ""},
		{"socks is not an HTTP proxy", true, "socks=s:1080", "", "https://github.com/x", ""},
		{"loopback is never proxied", true, "proxy:8080", "", "https://127.0.0.1:9/x", ""},
		{"localhost is never proxied", true, "proxy:8080", "", "https://localhost/x", ""},
		{"local names skip the proxy", true, "proxy:8080", "<local>", "https://intranet/x", ""},
		{"dotted names use it with <local>", true, "proxy:8080", "<local>", "https://github.com/x", "http://proxy:8080"},
		{"wildcard exception", true, "proxy:8080", "*.githubusercontent.com;10.*", "https://release-assets.githubusercontent.com/x", ""},
		{"wildcard exception, other host", true, "proxy:8080", "*.githubusercontent.com;10.*", "https://github.com/x", "http://proxy:8080"},
		{"exception with a scheme and a port", true, "proxy:8080", "https://github.com:443", "https://github.com/x", ""},
		{"exceptions are case blind", true, "proxy:8080", "GitHub.COM", "https://github.com/x", ""},
		{"address range pattern", true, "proxy:8080", "192.168.*", "https://192.168.1.5/x", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := manualProxy(tc.enabled, tc.server, tc.override, mustURL(t, tc.target))
			text := ""
			if got != nil {
				text = got.String()
			}
			if text != tc.want {
				t.Fatalf("proxy = %q, want %q", text, tc.want)
			}
		})
	}
}

func TestEnvironmentProxyWinsOverTheSystemProxy(t *testing.T) {
	saved := systemProxyFor
	t.Cleanup(func() { systemProxyFor = saved })
	systemProxyFor = func(*url.URL) (*url.URL, error) { return mustURL(t, "http://system.invalid:1"), nil }
	req, _ := http.NewRequest(http.MethodGet, "https://github.com/x", nil)
	for _, name := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"} {
		t.Setenv(name, "")
	}
	got, err := releaseProxy(req)
	if err != nil || got == nil || got.Host != "system.invalid:1" {
		t.Fatalf("without an environment proxy the system proxy is used: %v %v", got, err)
	}
	// net/http reads its environment once per process, so this is the only test
	// that sets a proxy variable, with a fixed value.
	t.Setenv("HTTPS_PROXY", "http://env.invalid:3128")
	got, err = releaseProxy(req)
	if err != nil || got == nil || got.Host != "env.invalid:3128" {
		t.Fatalf("the environment proxy must win: %v %v", got, err)
	}
}

// connectProxy is a CONNECT-only proxy that sends every tunnel to target and
// records which host each client asked for.
type connectProxy struct {
	*httptest.Server
	mu      sync.Mutex
	tunnels []string
}

func newConnectProxy(t *testing.T, target string) *connectProxy {
	t.Helper()
	p := &connectProxy{}
	p.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			http.Error(w, "CONNECT only", http.StatusMethodNotAllowed)
			return
		}
		p.mu.Lock()
		p.tunnels = append(p.tunnels, r.Host)
		p.mu.Unlock()
		upstream, err := net.Dial("tcp", target)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		hijacker, _ := w.(http.Hijacker)
		client, _, err := hijacker.Hijack()
		if err != nil {
			upstream.Close()
			return
		}
		client.Write([]byte("HTTP/1.1 200 Connection established\r\n\r\n"))
		go func() { io.Copy(upstream, client); upstream.Close() }()
		go func() { io.Copy(client, upstream); client.Close() }()
	}))
	t.Cleanup(p.Close)
	return p
}

func (p *connectProxy) seen() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.tunnels...)
}

// A named origin (the TLS test certificate covers example.com), reached only
// through the proxy: Go never proxies loopback hosts.
func proxiedClient(t *testing.T, origin *httptest.Server, proxy *connectProxy) (*Client, string) {
	t.Helper()
	_, port, _ := net.SplitHostPort(origin.Listener.Addr().String())
	base := "https://example.com:" + port
	saved := systemProxyFor
	t.Cleanup(func() { systemProxyFor = saved })
	systemProxyFor = func(target *url.URL) (*url.URL, error) {
		return manualProxy(true, strings.TrimPrefix(proxy.URL, "http://"), "", target), nil
	}
	pool := x509.NewCertPool()
	pool.AddCert(origin.Certificate())
	install := filepath.Join(t.TempDir(), "Bubo")
	if err := os.MkdirAll(install, 0700); err != nil {
		t.Fatal(err)
	}
	c, f := New(Config{InstalledVersion: "0.1.1", InstallDir: install, ManifestURL: base + releasePath + "latest/download/bubo-release.json", AllowedHosts: []string{"example.com"}, RootCAs: pool, HealthTimeout: 2 * time.Second})
	if f != nil {
		t.Fatal(f)
	}
	c.freeSpace = func(string) (uint64, error) { return math.MaxUint64, nil }
	t.Cleanup(c.Close)
	return c, base
}

func TestCheckAndDownloadGoThroughAnHTTPProxy(t *testing.T) {
	payload, unpacked := packageZip(t, fixtureEntries([]byte("gui"), []byte("helper")))
	var manifest Manifest
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "bubo-release.json") {
			json.NewEncoder(w).Encode(manifest)
			return
		}
		w.Write(payload)
	}))
	defer origin.Close()
	proxy := newConnectProxy(t, origin.Listener.Addr().String())
	c, base := proxiedClient(t, origin, proxy)
	manifest = testManifest(base, payload)
	hash := sha256.Sum256(payload)
	manifest.Assets[0].SHA256 = hex.EncodeToString(hash[:])
	manifest.Assets[0].UnpackedBytes = unpacked
	if r := c.Check(context.Background()); r.Status != Available {
		t.Fatalf("check through the proxy: %+v %+v", r, r.Failure)
	}
	staged, f := c.Download(context.Background(), nil)
	if f != nil || staged == nil {
		t.Fatalf("download through the proxy: %+v", f)
	}
	tunnels := proxy.seen()
	if len(tunnels) == 0 {
		t.Fatal("nothing went through the proxy")
	}
	for _, host := range tunnels {
		if host != strings.TrimPrefix(base, "https://") {
			t.Fatalf("tunnel to %q, want only the release origin", host)
		}
	}
}

// The proxy carries bytes and decides nothing: a redirect to a host outside the
// allowlist is refused by the client and never reaches the proxy.
func TestAllowlistStillJudgesRedirectsThroughAProxy(t *testing.T) {
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://evil.example.org/bubo-release.json", http.StatusFound)
	}))
	defer origin.Close()
	proxy := newConnectProxy(t, origin.Listener.Addr().String())
	c, base := proxiedClient(t, origin, proxy)
	r := c.Check(context.Background())
	if r.Status != Failed || r.Failure == nil || r.Failure.Code != CodeURL {
		t.Fatalf("redirect off the allowlist: %+v %+v", r, r.Failure)
	}
	for _, host := range proxy.seen() {
		if host != strings.TrimPrefix(base, "https://") {
			t.Fatalf("the proxy was asked to tunnel to %q", host)
		}
	}
}

// With a proxy that refuses, the failure is a network failure, not a silent
// fall back to a direct connection.
func TestRefusingProxyIsANetworkFailureNotADirectConnection(t *testing.T) {
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("the origin was reached without the proxy")
	}))
	defer origin.Close()
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no", http.StatusForbidden)
	}))
	defer refusing.Close()
	proxy := &connectProxy{Server: refusing}
	c, _ := proxiedClient(t, origin, proxy)
	r := c.Check(context.Background())
	if r.Status != Failed || r.Failure == nil || r.Failure.Code != CodeNetwork {
		t.Fatalf("refused CONNECT: %+v %+v", r, r.Failure)
	}
}
