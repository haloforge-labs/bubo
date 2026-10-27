//go:build windows

package supervisor

import (
	"net"
	"net/url"
)

// Keep the CLI endpoint's scheme and SDK path. A bare listener has no SDK path
// and is usable only for bind inspection, never for a guessed API address.
func boundEndpoint(st *Status, instance *Instance) *url.URL {
	if st != nil && st.Endpoints != nil {
		if u, err := url.Parse(st.Endpoints.OpenAIBaseURL); err == nil && u.Hostname() != "" && endpointPort(u.String()) != 0 {
			return u
		}
	}
	if instance != nil && instance.Endpoint != nil {
		if u, err := url.Parse(*instance.Endpoint); err == nil && endpointPort(u.String()) != 0 {
			return u
		}
	}
	return nil
}

func (a *AppService) lanState(snap Snapshot, saved Settings) LanState {
	present := a.keys.Present()
	a.mu.RLock()
	note, stale, gen := a.lanNote, a.keyStale, a.keyStaleGen
	a.mu.RUnlock()
	l := LanState{LanAccess: saved.LanAccess, APIKeyPresent: present, Unavailable: note, Addresses: []LanAddress{}}
	if present {
		l.APIKeyMask = KeyMask
		l.CopyWithKey = saved.LanAccess
	}
	endpoint := boundEndpoint(snap.Status, nil)
	// SDK URLs can use a reachable loopback host even when the listener binds
	// all interfaces. Read the selected instance's bind endpoint for LAN mode.
	if snap.Instance != nil {
		listener := boundEndpoint(nil, snap.Instance)
		if listener != nil {
			ip := net.ParseIP(listener.Hostname())
			l.LanActive = ip != nil && ip.IsUnspecified()
			l.KeyRestartNeeded = stale && l.LanActive && snap.Generation == gen
		}
	}
	if saved.LanAccess {
		// A saved zero port means CLI default. Only a CLI-bound SDK endpoint
		// supplies both the actual port and the endpoint path.
		if snap.Status != nil && snap.Status.Endpoints != nil && endpoint != nil && a.adapters != nil {
			if list, err := a.adapters.Adapters(); err == nil {
				l.Addresses = buildAddresses(list, endpoint.String())
			}
		}
		if len(l.Addresses) == 0 {
			l.AddressesUnavailable = addressesUnavailableText
		}
	}
	return l
}
