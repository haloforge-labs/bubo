package updater

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
)

// releaseProxy chooses the proxy for one release request: the environment when
// it names a proxy (HTTPS_PROXY, HTTP_PROXY, NO_PROXY, read as net/http reads
// them), otherwise the user's manual system proxy. A proxy only carries the
// connection (an HTTPS request goes through a CONNECT tunnel and its TLS is
// still verified against the origin): it never changes the request URL, so the
// host allowlist and the exact release path still judge the origin on every hop.
// PAC scripts and automatic proxy detection are not supported.
func releaseProxy(req *http.Request) (*url.URL, error) {
	if envProxyNamed() {
		return http.ProxyFromEnvironment(req)
	}
	return systemProxyFor(req.URL)
}

// systemProxyFor is a variable so that a test can stand in for the registry.
var systemProxyFor = systemProxy

func envProxyNamed() bool {
	for _, name := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"} {
		if os.Getenv(name) != "" {
			return true
		}
	}
	return false
}

// manualProxy applies a manual Windows (Internet Options) proxy setting. server
// is "host:port" for every scheme or "http=h:p;https=h:p" per scheme; override is
// the exception list ("<local>" and host patterns with * wildcards, separated by
// semicolons). A request to a loopback host is never proxied.
func manualProxy(enabled bool, server, override string, target *url.URL) *url.URL {
	if !enabled || strings.TrimSpace(server) == "" || target == nil {
		return nil
	}
	host := strings.ToLower(target.Hostname())
	if host == "" || host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return nil
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return nil
	}
	for _, pattern := range strings.FieldsFunc(override, func(r rune) bool { return r == ';' || r == ',' || r == ' ' }) {
		pattern = strings.ToLower(pattern)
		if pattern == "<local>" {
			if !strings.Contains(host, ".") {
				return nil
			}
			continue
		}
		pattern = strings.TrimPrefix(strings.TrimPrefix(pattern, "https://"), "http://")
		if i := strings.LastIndex(pattern, ":"); i >= 0 && !strings.Contains(pattern, "]") {
			pattern = pattern[:i]
		}
		if wildcardMatch(pattern, host) {
			return nil
		}
	}
	chosen := ""
	if !strings.Contains(server, "=") {
		chosen = strings.TrimSpace(server)
	} else {
		for _, entry := range strings.Split(server, ";") {
			scheme, address, found := strings.Cut(strings.TrimSpace(entry), "=")
			if found && strings.EqualFold(scheme, target.Scheme) {
				chosen = strings.TrimSpace(address)
			}
		}
	}
	if chosen == "" {
		return nil
	}
	if !strings.Contains(chosen, "://") {
		chosen = "http://" + chosen
	}
	proxy, err := url.Parse(chosen)
	if err != nil || proxy.Hostname() == "" || (proxy.Scheme != "http" && proxy.Scheme != "https") {
		return nil
	}
	return proxy
}

// wildcardMatch matches host against a pattern whose * stands for any run of
// characters (so "*.example.com" and "10.*" work).
func wildcardMatch(pattern, host string) bool {
	if pattern == "" {
		return false
	}
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return pattern == host
	}
	if !strings.HasPrefix(host, parts[0]) {
		return false
	}
	host = host[len(parts[0]):]
	for _, part := range parts[1 : len(parts)-1] {
		i := strings.Index(host, part)
		if i < 0 {
			return false
		}
		host = host[i+len(part):]
	}
	return strings.HasSuffix(host, parts[len(parts)-1])
}
