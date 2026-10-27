//go:build windows

package supervisor

import (
	"errors"
	"net"
	"unsafe"

	"golang.org/x/sys/windows"
)

// winAdapters reads the adapter list with the Windows IP helper (read-only).
type winAdapters struct{}

func (winAdapters) Adapters() ([]netAdapter, error) {
	const flags = windows.GAA_FLAG_INCLUDE_GATEWAYS | windows.GAA_FLAG_SKIP_ANYCAST | windows.GAA_FLAG_SKIP_MULTICAST | windows.GAA_FLAG_SKIP_DNS_SERVER
	size := uint32(15000)
	var buf []byte
	for try := 0; try < 5; try++ {
		buf = make([]byte, size)
		err := windows.GetAdaptersAddresses(windows.AF_INET, flags, 0, (*windows.IpAdapterAddresses)(unsafe.Pointer(&buf[0])), &size)
		if err == nil {
			break
		}
		if !errors.Is(err, windows.ERROR_BUFFER_OVERFLOW) || try == 4 {
			if errors.Is(err, windows.ERROR_NO_DATA) {
				return nil, nil
			}
			return nil, err
		}
	}
	var out []netAdapter
	for aa := (*windows.IpAdapterAddresses)(unsafe.Pointer(&buf[0])); aa != nil; aa = aa.Next {
		a := netAdapter{
			Name:        windows.UTF16PtrToString(aa.FriendlyName),
			Description: windows.UTF16PtrToString(aa.Description),
			IfType:      aa.IfType,
			Up:          aa.OperStatus == windows.IfOperStatusUp,
		}
		for u := aa.FirstUnicastAddress; u != nil; u = u.Next {
			if ip := u.Address.IP(); ip != nil {
				a.IPv4 = append(a.IPv4, ip)
			}
		}
		for g := aa.FirstGatewayAddress; g != nil; g = g.Next {
			if ip := g.Address.IP(); ip != nil && !net.IP.IsUnspecified(ip) {
				a.HasGateway = true
			}
		}
		out = append(out, a)
	}
	return out, nil
}
