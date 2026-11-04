//go:build !windows

package updater

import "net/url"

// Only the environment names a proxy off Windows.
func systemProxy(*url.URL) (*url.URL, error) { return nil, nil }
