package updater

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"bubo/gui/updater/jsonwire"
)

type Client struct {
	cfg       Config
	installed version
	origin    *url.URL
	targetTag string
	http      *http.Client
	root      string
	mu        sync.Mutex
	release   *Release
	staged    *Staged
	// Platform probes are private so production callers cannot bypass them.
	freeSpace func(string) (uint64, error)
}

func New(cfg Config) (*Client, *Failure) {
	v, err := parseVersion(cfg.InstalledVersion)
	if err != nil {
		return nil, fail(CodeConfig, err.Error())
	}
	if cfg.ManifestURL == "" {
		cfg.ManifestURL = ManifestURL
	}
	if len(cfg.AllowedHosts) == 0 {
		cfg.AllowedHosts = []string{"github.com", "*.githubusercontent.com"}
	} else {
		cfg.AllowedHosts = append([]string(nil), cfg.AllowedHosts...)
	}
	if cfg.RootCAs != nil {
		cfg.RootCAs = cfg.RootCAs.Clone()
	}
	cfg.InstallDir = filepath.Clean(cfg.InstallDir)
	if !filepath.IsAbs(cfg.InstallDir) || filepath.Dir(cfg.InstallDir) == cfg.InstallDir {
		return nil, fail(CodeConfig, "install directory must be an absolute package folder, not a volume root")
	}
	if err := safeTree(cfg.InstallDir); err != nil {
		// A power loss between renames can leave only .previous. Recovery must
		// still be constructible for that exact original install path.
		if !os.IsNotExist(err) || safeTree(cfg.InstallDir+".previous") != nil {
			code := CodeConfig
			if strings.Contains(err.Error(), "reparse point") {
				code = CodePathRedirected
			}
			return nil, fail(code, "install directory: "+err.Error())
		}
	}
	if cfg.HealthTimeout == 0 {
		cfg.HealthTimeout = DefaultHealthTimeout
	}
	if cfg.WaitTimeout == 0 {
		cfg.WaitTimeout = DefaultWaitTimeout
	}
	if cfg.HealthTimeout < time.Millisecond || cfg.WaitTimeout < time.Millisecond {
		return nil, fail(CodeConfig, "health and process wait timeouts must be positive")
	}
	u, err := url.Parse(cfg.ManifestURL)
	if err != nil {
		return nil, fail(CodeURL, "invalid manifest URL")
	}
	c := &Client{cfg: cfg, installed: v, origin: u, root: filepath.Join(filepath.Dir(cfg.InstallDir), "."+filepath.Base(cfg.InstallDir)+".bubo-update"), freeSpace: diskFree}
	if f := c.allowedURL(u); f != nil {
		return nil, f
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawPath != "" {
		return nil, fail(CodeURL, "manifest must use an exact HTTPS release download path")
	}
	if u.Path != releasePath+"latest/download/bubo-release.json" {
		tagPath, ok := strings.CutPrefix(u.Path, releasePath+"download/")
		tag, tail, found := strings.Cut(tagPath, "/")
		_, err := parseVersion(strings.TrimPrefix(tag, "v"))
		if !ok || !found || tail != "bubo-release.json" || !strings.HasPrefix(tag, "v") || err != nil {
			return nil, fail(CodeURL, "manifest must use latest or an exact stable release tag")
		}
		c.targetTag = tag
	}
	c.http = &http.Client{
		Timeout: 30 * time.Minute,
		Transport: &http.Transport{
			TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12, RootCAs: cfg.RootCAs},
			DisableCompression:    true,
			ResponseHeaderTimeout: 30 * time.Second,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fail(CodeURL, "too many release redirects")
			}
			if f := c.allowedURL(req.URL); f != nil {
				return f
			}
			if req.URL.User != nil || req.URL.Fragment != "" {
				return fail(CodeURL, "redirect credentials or fragments refused")
			}
			// No credentials, cookies, referrer, or identifying headers cross hosts.
			req.Header.Del("Referer")
			req.Header.Del("Authorization")
			req.Header.Del("Cookie")
			return nil
		},
	}
	return c, nil
}

func (c *Client) Close() { c.http.CloseIdleConnections() }

func (c *Client) allowedURL(u *url.URL) *Failure {
	if u.Scheme != "https" || u.Hostname() == "" || u.User != nil {
		return fail(CodeURL, "only allowlisted HTTPS URLs are permitted")
	}
	host := strings.ToLower(u.Hostname())
	for _, pattern := range c.cfg.AllowedHosts {
		pattern = strings.ToLower(pattern)
		if host == pattern || (strings.HasPrefix(pattern, "*.") && strings.HasSuffix(host, pattern[1:]) && host != pattern[2:]) {
			return nil
		}
	}
	return fail(CodeURL, "release host is not allowlisted: "+host)
}

func (c *Client) request(ctx context.Context, raw, rangeHeader string) (*http.Response, *Failure) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return nil, fail(CodeURL, "invalid release URL")
	}
	if f := c.allowedURL(req.URL); f != nil {
		return nil, f
	}
	req.Header.Set("User-Agent", "bubo/"+c.cfg.InstalledVersion)
	if rangeHeader != "" {
		req.Header.Set("Range", rangeHeader)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		code := CodeNetwork
		if ctx.Err() == context.Canceled {
			code = CodeCanceled
		}
		// Unwrap policy errors without exposing expiring signed redirect URLs.
		for e := err; e != nil; {
			if f, ok := e.(*Failure); ok {
				return nil, f
			}
			unwrapper, ok := e.(interface{ Unwrap() error })
			if !ok {
				break
			}
			e = unwrapper.Unwrap()
		}
		f := fail(code, "release request failed; check connectivity and retry")
		f.URL = raw
		return nil, f
	}
	return resp, nil
}

type seenRelease struct {
	Version  string    `json:"version"`
	Released time.Time `json:"released"`
	Via      *string   `json:"via"`
}

func (c *Client) Check(ctx context.Context) CheckResult {
	if !c.mu.TryLock() {
		return CheckResult{Status: Failed, Failure: fail(CodeBusy, "another updater operation is running")}
	}
	defer c.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	c.release = nil
	bad := func(f *Failure) CheckResult { return CheckResult{Status: Failed, Failure: f} }
	resp, f := c.request(ctx, c.cfg.ManifestURL, "")
	if f != nil {
		return bad(f)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		code := CodeUpdateServer
		if resp.StatusCode == http.StatusNotFound {
			code = CodeUpdateInfoMissing
		}
		f := fail(code, fmt.Sprintf("manifest returned HTTP %d", resp.StatusCode))
		f.URL = c.cfg.ManifestURL
		return bad(f)
	}
	const limit = 1 << 20
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil || len(data) > limit {
		return bad(fail(CodeManifest, "manifest is incomplete or exceeds 1 MiB"))
	}
	m, a, f := c.decodeManifest(data)
	if f != nil {
		f.URL = c.cfg.ManifestURL
		return bad(f)
	}
	if c.targetTag != "" && m.Tag != c.targetTag {
		return bad(fail(CodeManifest, "manifest tag does not match the requested intermediate release"))
	}
	v, _ := parseVersion(m.Version)
	if compare(v, c.installed) < 0 {
		return bad(fail(CodeDowngrade, "release is older than the installed version"))
	}
	if err := ensureDir(c.root); err != nil {
		return bad(fail(CodeIO, err.Error()))
	}
	seenPath := filepath.Join(c.root, "seen.json")
	var seen seenRelease
	viaHop := false
	if data, err := readRegular(seenPath); err == nil {
		if err := jsonwire.Unmarshal(data, &seen); err != nil {
			return bad(fail(CodeIO, "saved release history is invalid"))
		}
		sv, err := parseVersion(seen.Version)
		if err != nil {
			return bad(fail(CodeIO, "saved release version is invalid"))
		}
		// An explicitly requested intermediate is the one deliberate exception
		// to newest-seen ordering. It must have been named by that saved feed,
		// and still be newer than the installed version; never an arbitrary tag.
		viaHop = c.targetTag != "" && seen.Via != nil && m.Version == *seen.Via && compare(v, c.installed) > 0
		if (compare(v, sv) < 0 || m.Released.Before(seen.Released)) && !viaHop {
			return bad(fail(CodeStale, "manifest is older than an already observed release"))
		}
	} else if !os.IsNotExist(err) {
		return bad(fail(CodeIO, "cannot read release history: "+err.Error()))
	}
	if !viaHop {
		if err := writeJSON(seenPath, seenRelease{Version: m.Version, Released: m.Released, Via: m.Via}); err != nil {
			return bad(fail(CodeIO, "cannot save release history: "+err.Error()))
		}
	}
	if compare(v, c.installed) == 0 {
		return CheckResult{Status: UpToDate}
	}
	min, _ := parseVersion(m.MinUpdatableFrom)
	if compare(c.installed, min) < 0 {
		f := fail(CodeIntermediateRequired, "installed version cannot update directly to "+m.Version)
		f.NotesURL = m.Notes.URL
		f.Retry = "install the intermediate version first, or install manually from the release page"
		if m.Via != nil {
			f.Via = *m.Via
			f.URL = c.origin.Scheme + "://" + c.origin.Host + releasePath + "download/v" + *m.Via + "/bubo-release.json"
		}
		return bad(f)
	}
	r := Release{Version: m.Version, Notes: m.Notes, Size: a.Size, Asset: a, Manifest: m, Advisories: []Advisory{}}
	for _, advisory := range m.Advisories {
		if matches, _ := advisoryMatches(advisory.Versions, c.installed); matches {
			r.Advisories = append(r.Advisories, advisory)
		}
	}
	c.release = &r
	// Return an independent value so the view layer cannot mutate a checked asset.
	copy := r
	return CheckResult{Status: Available, Available: &copy}
}
