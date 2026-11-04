//go:build windows && !bindings

package main

import "testing"

func TestWindowOpensAtTheSavedSizeOrTheLargeDefaultNeverBeyondTheScreen(t *testing.T) {
	cases := []struct {
		name           string
		savedW, savedH int
		workW, workH   int
		wantW, wantH   int
	}{
		{"first start on a large screen", 0, 0, 2560, 1500, 1280, 860},
		{"the size the user left", 1500, 1000, 2560, 1500, 1500, 1000},
		{"a smaller screen clamps the default", 0, 0, 1366, 728, 1280, 728},
		{"a saved size from a larger screen clamps", 3000, 2000, 1920, 1040, 1920, 1040},
		{"never below the minimum", 300, 200, 2560, 1500, 560, 360},
		{"an unknown work area keeps the size", 0, 0, 0, 0, 1280, 860},
	}
	for _, c := range cases {
		w, h := clampWindowSize(c.savedW, c.savedH, c.workW, c.workH)
		if w != c.wantW || h != c.wantH {
			t.Errorf("%s: %dx%d, want %dx%d", c.name, w, h, c.wantW, c.wantH)
		}
	}
}
