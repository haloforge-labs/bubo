//go:build windows

package supervisor

import "strings"

// A part file (vision projector, MTP, twin, draft head) belongs to one model.
// When the chosen model changes, an override that points at such a file would
// load the old model's part into the new one. The CLI marks these rows in the
// registry (`requires` carries `artefact:<kind>`); the GUI drops exactly those
// overrides and knows no part name of its own.
func dropModelFileOverrides(overrides map[string]string, registry *Switches) map[string]string {
	if registry == nil || len(overrides) == 0 {
		return overrides
	}
	points := map[string]bool{}
	for _, row := range registry.Switches {
		for _, need := range row.Requires {
			if strings.HasPrefix(need, "artefact:") {
				points[row.Name] = true
			}
		}
	}
	next := make(map[string]string, len(overrides))
	for name, value := range overrides {
		if !points[name] {
			next[name] = value
		}
	}
	return next
}

// modelChanged reports whether a patch chose another model (or none).
func modelChanged(current, next Settings) bool {
	return current.ModelPath != next.ModelPath || current.ManifestPath != next.ManifestPath
}
