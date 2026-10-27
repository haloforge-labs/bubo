package supervisor

import (
	"net"
	"net/url"
	"sort"
	"strconv"
	"strings"
)

// addressesUnavailableText is shown when LAN access is on but no usable address exists.
const addressesUnavailableText = "Could not find this computer's network address."

// LanAddress is one address other devices can use to reach the server.
// It never holds the API key.
type LanAddress = NetworkAddress

// netAdapter is the part of an adapter the filter needs.
type netAdapter struct {
	Name        string
	Description string
	IfType      uint32
	Up          bool
	IPv4        []net.IP
	HasGateway  bool
}

// adapterSource lists the adapters of this computer. Tests use a fake.
type adapterSource interface {
	Adapters() ([]netAdapter, error)
}

const (
	ifTypeLoopback = 24
	ifTypeTunnel   = 131
)

var virtualMarkers = []string{
	"vethernet", "hyper-v", "wsl", "vpn", "vmware", "virtualbox", "virtual", "bluetooth",
	"tap-", "tap windows", "tunnel", "wintun", "wireguard", "tailscale", "zerotier", "loopback", "pseudo", "teredo", "isatap",
}

func isVirtualAdapter(a netAdapter) bool {
	if a.IfType == ifTypeLoopback || a.IfType == ifTypeTunnel {
		return true
	}
	s := strings.ToLower(a.Name + " " + a.Description)
	for _, m := range virtualMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

func usableIPv4(ip net.IP) (string, bool) {
	v4 := ip.To4()
	if v4 == nil || v4.IsUnspecified() || v4.IsLoopback() || v4.IsLinkLocalUnicast() || v4.IsMulticast() {
		return "", false
	}
	return v4.String(), true
}

// buildAddresses applies the filter and ordering rules. The result is never nil.
// Physical adapters with a gateway come first; virtual adapters are used only when
// no physical adapter has a gateway (then a virtual adapter with a gateway is kept).
func buildAddresses(adapters []netAdapter, endpoint string) []LanAddress {
	base, err := url.Parse(endpoint)
	if err != nil || base.Scheme == "" || endpointPort(endpoint) == 0 {
		return []LanAddress{}
	}
	var phys, virt []netAdapter
	physGateway := false
	for _, a := range adapters {
		if !a.Up || a.IfType == ifTypeLoopback || a.IfType == ifTypeTunnel {
			continue
		}
		if isVirtualAdapter(a) {
			virt = append(virt, a)
			continue
		}
		phys = append(phys, a)
		if a.HasGateway {
			physGateway = true
		}
	}
	use := phys
	if !physGateway {
		for _, a := range virt {
			if a.HasGateway {
				use = append(use, a)
			}
		}
	}
	sort.SliceStable(use, func(i, j int) bool { return use[i].HasGateway && !use[j].HasGateway })
	out := []LanAddress{}
	seen := map[string]bool{}
	for _, a := range use {
		for _, ip := range a.IPv4 {
			s, ok := usableIPv4(ip)
			if !ok || seen[s] {
				continue
			}
			seen[s] = true
			address := *base
			address.Host = net.JoinHostPort(s, base.Port())
			out = append(out, LanAddress{Adapter: a.Name, IPv4: s, URL: address.String()})
		}
	}
	return out
}

// endpointPort returns the port of an endpoint such as http://0.0.0.0:8080, or 0.
func endpointPort(endpoint string) int {
	u, err := url.Parse(endpoint)
	if err != nil {
		return 0
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil || p < 1 || p > 65535 {
		return 0
	}
	return p
}
