package updater

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf16"

	"bubo/gui/updater/jsonwire"
)

// ReleaseOptions contains release decisions; binary facts come from the ZIP.
type ReleaseOptions struct {
	Version, Tag, Commit, MinUpdatableFrom, Notes string
	RuntimeChanged, ModelsChanged                 *bool
	RuntimeDownloadBytes                          *int64
	ModelsNeedsPrepare                            []string
	Released                                      time.Time
	InstallPathLength                             int
}

// GenerateReleaseManifest uses the same types and validator as update checks.
// It performs no network requests and never executes packaged code.
func GenerateReleaseManifest(archive string, opts ReleaseOptions) (Manifest, string, error) {
	bad := func(err error) (Manifest, string, error) { return Manifest{}, "", err }
	if opts.RuntimeChanged == nil || opts.ModelsChanged == nil || opts.RuntimeDownloadBytes == nil || opts.ModelsNeedsPrepare == nil {
		return bad(fmt.Errorf("runtime-changed, runtime-download-bytes, models-changed and models-needs-prepare are required release decisions"))
	}
	if strings.TrimSpace(opts.Notes) == "" {
		return bad(fmt.Errorf("a nonempty user-facing release summary is required"))
	}
	z, err := zip.OpenReader(archive)
	if err != nil {
		return bad(err)
	}
	defer z.Close()
	if len(z.File) == 0 {
		return bad(fmt.Errorf("empty package ZIP"))
	}
	var build struct {
		Product struct {
			Version string `json:"product_version"`
		} `json:"product"`
		GUI struct {
			Commit    string `json:"commit"`
			RuntimeID string `json:"runtime_id"`
		} `json:"gui"`
	}
	top := strings.Split(z.File[0].Name, "/")[0]
	var total int64
	longest := ""
	seen := make(map[string]bool)
	for _, f := range z.File {
		rel, err := zipRelative(f.Name, top)
		if err != nil {
			return bad(err)
		}
		if rel == "" {
			continue
		}
		key := strings.ToLower(rel)
		if _, ok := seen[key]; ok {
			return bad(fmt.Errorf("duplicate ZIP path %s", f.Name))
		}
		seen[key] = f.FileInfo().IsDir()
		if f.Mode()&os.ModeType != 0 && !f.FileInfo().IsDir() {
			return bad(fmt.Errorf("special ZIP entry %s", f.Name))
		}
		if !f.FileInfo().IsDir() {
			if f.UncompressedSize64 > uint64(math.MaxInt64-total) {
				return bad(fmt.Errorf("unpacked size overflow"))
			}
			total += int64(f.UncompressedSize64)
			if len(utf16.Encode([]rune(rel))) > len(utf16.Encode([]rune(longest))) {
				longest = rel
			}
		}
		if filepath.ToSlash(rel) == "BUILD-MANIFEST.json" {
			r, err := f.Open()
			if err != nil {
				return bad(err)
			}
			data, readErr := io.ReadAll(io.LimitReader(r, 4<<20))
			err = readErr
			if err == nil {
				err = jsonwire.Unmarshal(data, &build)
			}
			r.Close()
			if err != nil {
				return bad(err)
			}
		}
	}
	for _, rel := range PackageExecutables() {
		dir, ok := seen[strings.ToLower(filepath.FromSlash(rel))]
		if !ok || dir {
			return bad(fmt.Errorf("package is missing %s", rel))
		}
	}
	if build.Product.Version == "" || build.GUI.RuntimeID == "" || build.GUI.Commit == "" {
		return bad(fmt.Errorf("package lacks version, GUI commit or CLI runtime metadata"))
	}
	if opts.Version == "" {
		opts.Version = build.Product.Version
	}
	if opts.Version != build.Product.Version {
		return bad(fmt.Errorf("version differs from packaged product"))
	}
	if opts.Commit == "" {
		opts.Commit = build.GUI.Commit
	}
	if opts.Commit != build.GUI.Commit {
		return bad(fmt.Errorf("commit differs from packaged GUI provenance"))
	}
	if opts.Tag == "" {
		opts.Tag = "v" + opts.Version
	}
	if opts.MinUpdatableFrom == "" {
		return bad(fmt.Errorf("min-updatable-from is required"))
	}
	if opts.Released.IsZero() {
		return bad(fmt.Errorf("released time is required"))
	}
	if opts.InstallPathLength < 1 {
		return bad(fmt.Errorf("install-path-length must be positive"))
	}
	// shortPath counts UTF-16 units, as Windows does, including the separator.
	if !shortPath(strings.Repeat("x", opts.InstallPathLength) + "/" + longest) {
		return bad(fmt.Errorf("package path exceeds Windows limit at install path length %d: %s", opts.InstallPathLength, longest))
	}
	f, err := os.Open(archive)
	if err != nil {
		return bad(err)
	}
	defer f.Close()
	h := sha256.New()
	size, err := io.Copy(h, f)
	if err != nil {
		return bad(err)
	}
	origin, _ := url.Parse(ManifestURL)
	base := origin.Scheme + "://" + origin.Host + releasePath
	m := Manifest{
		SchemaVersion: 1, Product: "bubo", Channel: "stable", Version: opts.Version, Tag: opts.Tag,
		Commit: opts.Commit, Released: opts.Released.UTC(), MinUpdatableFrom: opts.MinUpdatableFrom,
		Assets:  []Asset{{Name: filepath.Base(archive), Kind: "package", URL: base + "download/" + opts.Tag + "/" + filepath.Base(archive), Size: size, SHA256: hex.EncodeToString(h.Sum(nil)), UnpackedBytes: total, TopFolder: top}},
		Notes:   Notes{Summary: strings.TrimSpace(opts.Notes), URL: base + "tag/" + opts.Tag},
		Runtime: Runtime{ID: build.GUI.RuntimeID, Changed: *opts.RuntimeChanged, DownloadBytes: *opts.RuntimeDownloadBytes}, Models: Models{Changed: *opts.ModelsChanged, NeedsPrepare: opts.ModelsNeedsPrepare}, Advisories: []Advisory{}, RestartRequired: true,
	}
	data, err := json.Marshal(m)
	if err != nil {
		return bad(err)
	}
	c := Client{origin: origin, cfg: Config{AllowedHosts: []string{origin.Hostname()}}}
	if _, _, failure := c.decodeManifest(data); failure != nil {
		return bad(failure)
	}
	return m, filepath.ToSlash(longest), nil
}
