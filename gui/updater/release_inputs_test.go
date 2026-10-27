package updater

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// Failure cases before implementation: omitted decisions silently claim reuse;
// an empty summary ships developer text or nothing; explicit changed facts are lost.
func TestReleaseRequiresExplicitDecisions(t *testing.T) {
	archive := os.Getenv("BUBO_RELEASE_ZIP")
	if archive == "" {
		t.Skip("set BUBO_RELEASE_ZIP to a real package")
	}
	raw, err := os.ReadFile("../tests/scenarios/recordings/release-update/bubo-release.json")
	if err != nil {
		t.Fatal(err)
	}
	var release Manifest
	if err := json.Unmarshal(raw, &release); err != nil {
		t.Fatal(err)
	}
	text, err := os.ReadFile("../WHATSNEW.md")
	if err != nil {
		t.Fatal(err)
	}
	summary := ""
	for _, line := range strings.Split(string(text), "\n") {
		if strings.HasPrefix(line, "- ") {
			summary = strings.ReplaceAll(strings.TrimSpace(line[2:]), "**", "")
			break
		}
	}
	base := map[string]any{"MinUpdatableFrom": release.MinUpdatableFrom, "Released": release.Released, "InstallPathLength": 60, "Notes": summary, "RuntimeChanged": false, "RuntimeDownloadBytes": 0, "ModelsChanged": false, "ModelsNeedsPrepare": []string{}}
	for _, missing := range []string{"RuntimeChanged", "RuntimeDownloadBytes", "ModelsChanged", "ModelsNeedsPrepare", "Notes"} {
		t.Run(missing, func(t *testing.T) {
			values := map[string]any{}
			for key, value := range base {
				if key != missing {
					values[key] = value
				}
			}
			data, _ := json.Marshal(values)
			var opts ReleaseOptions
			_ = json.Unmarshal(data, &opts)
			if _, _, err := GenerateReleaseManifest(archive, opts); err == nil {
				t.Fatalf("missing %s accepted", missing)
			}
		})
	}
	data, _ := json.Marshal(base)
	var opts ReleaseOptions
	_ = json.Unmarshal(data, &opts)
	got, _, err := GenerateReleaseManifest(archive, opts)
	if err != nil || got.Notes.Summary != summary || got.Runtime.Changed || got.Runtime.DownloadBytes != 0 || got.Models.Changed {
		t.Fatalf("explicit release facts changed: %+v %v", got, err)
	}
}
