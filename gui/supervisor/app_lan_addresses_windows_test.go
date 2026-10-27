//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"regexp"
	"strings"
	"testing"
)

type fakeAdapters struct {
	list []netAdapter
	err  error
}

func (f fakeAdapters) Adapters() ([]netAdapter, error) { return f.list, f.err }

func ad(name, desc string, typ uint32, gw bool, ips ...string) netAdapter {
	a := netAdapter{Name: name, Description: desc, IfType: typ, Up: true, HasGateway: gw}
	for _, s := range ips {
		a.IPv4 = append(a.IPv4, net.ParseIP(s))
	}
	return a
}

func lanFrame(t *testing.T, src adapterSource, lan bool, port int) LanState {
	t.Helper()
	b := &fakeBackend{view: Snapshot{Status: &Status{Endpoints: &Endpoints{OpenAIBaseURL: fmt.Sprintf("http://0.0.0.0:%d/v1", port)}}}}
	s := DefaultSettings()
	s.LanAccess, s.Port = lan, port
	a := newTestApp(b, s, nil, "")
	a.keys = NewKeyManager(&memStore{m: map[string]string{}})
	a.adapters = src
	return a.lanState(b.Snapshot(), s)
}

func TestLanAddressesGatewayFirstAndPort(t *testing.T) {
	l := lanFrame(t, fakeAdapters{list: []netAdapter{
		ad("Ethernet 2", "Realtek PCIe", 6, false, "10.0.0.5"),
		ad("Wi-Fi", "MediaTek Wi-Fi 7", 71, true, "192.168.1.23"),
	}}, true, 18110)
	if len(l.Addresses) != 2 || l.Addresses[0].Adapter != "Wi-Fi" || l.Addresses[0].IPv4 != "192.168.1.23" || l.Addresses[0].URL != "http://192.168.1.23:18110/v1" || l.Addresses[1].Adapter != "Ethernet 2" {
		t.Fatalf("%+v", l.Addresses)
	}
	if l.AddressesUnavailable != "" {
		t.Fatal(l.AddressesUnavailable)
	}
}

func TestLanAddressesUseRunningEndpointPort(t *testing.T) {
	b := &fakeBackend{}
	snap := b.Snapshot()
	ep := "http://0.0.0.0:18222"
	snap.Instance = &Instance{Endpoint: &ep}
	snap.Status = &Status{Endpoints: &Endpoints{OpenAIBaseURL: ep + "/v1"}}
	s := DefaultSettings()
	s.LanAccess = true
	a := newTestApp(b, s, nil, "")
	a.keys = NewKeyManager(&memStore{m: map[string]string{}})
	a.adapters = fakeAdapters{list: []netAdapter{ad("Wi-Fi", "x", 71, true, "192.168.1.9")}}
	l := a.lanState(snap, s)
	if len(l.Addresses) != 1 || l.Addresses[0].URL != "http://192.168.1.9:18222/v1" {
		t.Fatalf("%+v", l.Addresses)
	}
}

func TestLanAddressesSkipVirtualAndLinkLocal(t *testing.T) {
	down := ad("Ethernet", "Intel", 6, true, "10.1.1.1")
	down.Up = false
	l := lanFrame(t, fakeAdapters{list: []netAdapter{
		ad("vEthernet (WSL)", "Hyper-V Virtual Ethernet Adapter", 6, false, "172.20.0.1"),
		ad("VMware Network Adapter VMnet8", "VMware Virtual Ethernet Adapter", 6, false, "192.168.50.1"),
		ad("VirtualBox Host-Only Network", "VirtualBox Host-Only Ethernet Adapter", 6, false, "192.168.56.1"),
		ad("Bluetooth Network Connection", "Bluetooth Device (Personal Area Network)", 6, false, "192.168.44.2"),
		ad("Proton VPN", "WireGuard Tunnel", 131, true, "10.2.0.2"),
		ad("Loopback", "Software Loopback", 24, false, "127.0.0.1"),
		down,
		ad("Ethernet 3", "Intel", 6, false, "169.254.7.7", "0.0.0.0"),
		ad("Wi-Fi", "MediaTek", 71, true, "192.168.1.23", "169.254.1.2"),
	}}, true, 8080)
	if len(l.Addresses) != 1 || l.Addresses[0].IPv4 != "192.168.1.23" {
		t.Fatalf("%+v", l.Addresses)
	}
}

func TestLanAddressesVirtualGatewayKeptOnlyWhenNoPhysicalGateway(t *testing.T) {
	l := lanFrame(t, fakeAdapters{list: []netAdapter{
		ad("Ethernet", "Intel", 6, false, "10.0.0.5"),
		ad("vEthernet (External)", "Hyper-V Virtual Ethernet Adapter", 6, true, "192.168.1.77"),
	}}, true, 8080)
	if len(l.Addresses) != 2 || l.Addresses[0].Adapter != "vEthernet (External)" {
		t.Fatalf("%+v", l.Addresses)
	}
	l = lanFrame(t, fakeAdapters{list: []netAdapter{
		ad("Wi-Fi", "MediaTek", 71, true, "192.168.1.23"),
		ad("vEthernet (External)", "Hyper-V Virtual Ethernet Adapter", 6, true, "192.168.1.77"),
	}}, true, 8080)
	if len(l.Addresses) != 1 || l.Addresses[0].Adapter != "Wi-Fi" {
		t.Fatalf("%+v", l.Addresses)
	}
}

func TestLanAddressesNoneOrErrorGivesUnavailable(t *testing.T) {
	for _, src := range []adapterSource{fakeAdapters{}, fakeAdapters{err: errors.New("boom")}, fakeAdapters{list: []netAdapter{ad("Loopback", "", 24, false, "127.0.0.1")}}} {
		l := lanFrame(t, src, true, 8080)
		if l.Addresses == nil || len(l.Addresses) != 0 || l.AddressesUnavailable != "Could not find this computer's network address." {
			t.Fatalf("%+v", l)
		}
		b, _ := json.Marshal(l)
		if !strings.Contains(string(b), `"addresses":[]`) || !strings.Contains(string(b), `"addresses_unavailable":"Could not find this computer's network address."`) {
			t.Fatal(string(b))
		}
	}
}

func TestLanAddressesLanOffIsEmptyWithoutText(t *testing.T) {
	l := lanFrame(t, fakeAdapters{list: []netAdapter{ad("Wi-Fi", "x", 71, true, "192.168.1.23")}}, false, 8080)
	b, _ := json.Marshal(l)
	if len(l.Addresses) != 0 || l.AddressesUnavailable != "" || !strings.Contains(string(b), `"addresses":[]`) || strings.Contains(string(b), "addresses_unavailable") {
		t.Fatalf("%s", b)
	}
}

func TestLanAddressesNeverCarryKey(t *testing.T) {
	st := &memStore{m: map[string]string{}}
	km := NewKeyManager(st)
	key, err := km.Ensure()
	if err != nil {
		t.Fatal(err)
	}
	b := &fakeBackend{}
	s := DefaultSettings()
	s.LanAccess = true
	a := newTestApp(b, s, nil, "")
	a.keys = km
	a.adapters = fakeAdapters{list: []netAdapter{ad("Wi-Fi", "x", 71, true, "192.168.1.23")}}
	j, _ := json.Marshal(a.lanState(b.Snapshot(), s))
	if key == "" || strings.Contains(string(j), key) {
		t.Fatal("key leaked into lan state")
	}
}

func TestEndpointPort(t *testing.T) {
	for in, want := range map[string]int{"http://0.0.0.0:8080": 8080, "http://127.0.0.1:18110": 18110, "http://x": 0, "": 0, "http://h:0": 0, "http://h:99999": 0} {
		if got := endpointPort(in); got != want {
			t.Errorf("%q: %d want %d", in, got, want)
		}
	}
}

func TestRealAdaptersWellFormed(t *testing.T) {
	list, err := winAdapters{}.Adapters()
	if err != nil {
		t.Fatalf("real detector: %v", err)
	}
	for _, a := range list {
		t.Logf("adapter %q (%s) type=%d up=%v gateway=%v ipv4=%v", a.Name, a.Description, a.IfType, a.Up, a.HasGateway, a.IPv4)
	}
	addrs := buildAddresses(list, "http://0.0.0.0:8080/v1")
	re := regexp.MustCompile(`^http://\d+\.\d+\.\d+\.\d+:8080/v1$`)
	for _, x := range addrs {
		t.Logf("address: %s -> %s", x.Adapter, x.URL)
		ip := net.ParseIP(x.IPv4)
		if x.Adapter == "" || ip == nil || ip.To4() == nil || ip.IsLoopback() || ip.IsLinkLocalUnicast() || !re.MatchString(x.URL) || x.URL != "http://"+x.IPv4+":8080/v1" {
			t.Fatalf("malformed: %+v", x)
		}
	}
	if addrs == nil {
		t.Fatal("nil slice")
	}
}
