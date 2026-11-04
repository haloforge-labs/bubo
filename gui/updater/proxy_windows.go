//go:build windows

package updater

import (
	"net/url"

	"golang.org/x/sys/windows/registry"
)

// systemProxy reads the manual proxy of the Windows Internet Options of the
// current user on every request (a changed setting applies at once). A setting
// that cannot be read means no proxy.
func systemProxy(target *url.URL) (*url.URL, error) {
	key, err := registry.OpenKey(registry.CURRENT_USER, `Software\Microsoft\Windows\CurrentVersion\Internet Settings`, registry.QUERY_VALUE)
	if err != nil {
		return nil, nil
	}
	defer key.Close()
	enabled, _, err := key.GetIntegerValue("ProxyEnable")
	if err != nil {
		return nil, nil
	}
	server, _, _ := key.GetStringValue("ProxyServer")
	override, _, _ := key.GetStringValue("ProxyOverride")
	return manualProxy(enabled != 0, server, override, target), nil
}
