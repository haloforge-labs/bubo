package updater

import (
	"fmt"
	"strconv"
	"strings"
)

// Stable feed versions use exactly X.Y.Z. No prereleases, aliases, or arbitrary
// dotted-number ordering: GitHub latest already excludes prereleases.
type version [3]uint64

func parseVersion(s string) (version, error) {
	var v version
	parts := strings.Split(s, ".")
	if len(parts) != len(v) {
		return v, fmt.Errorf("version %q must be X.Y.Z", s)
	}
	for i, part := range parts {
		if part == "" || (len(part) > 1 && part[0] == '0') || strings.IndexFunc(part, func(r rune) bool { return r < '0' || r > '9' }) >= 0 {
			return v, fmt.Errorf("invalid stable version %q", s)
		}
		n, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			return v, fmt.Errorf("invalid stable version %q", s)
		}
		v[i] = n
	}
	return v, nil
}

func compare(a, b version) int {
	for i := range a {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	return 0
}

func advisoryMatches(s string, v version) (bool, error) {
	if strings.HasPrefix(s, "<=") {
		end, err := parseVersion(strings.TrimPrefix(s, "<="))
		return compare(v, end) <= 0, err
	}
	if a, b, ok := strings.Cut(s, "-"); ok {
		start, err := parseVersion(a)
		if err != nil {
			return false, err
		}
		end, err := parseVersion(b)
		if err != nil || compare(start, end) > 0 {
			return false, fmt.Errorf("invalid advisory range %q", s)
		}
		return compare(v, start) >= 0 && compare(v, end) <= 0, nil
	}
	exact, err := parseVersion(s)
	return compare(v, exact) == 0, err
}
