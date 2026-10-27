package updater

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"path"
	"strings"
	"unicode"
	"unicode/utf8"

	"bubo/gui/updater/jsonwire"
)

const releasePath = "/haloforge-labs/bubo/releases/"

func required(data []byte, keys ...string) error {
	var fields map[string]json.RawMessage
	if err := jsonwire.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, key := range keys {
		v, ok := fields[key]
		if !ok || (key != "via" && bytes.Equal(bytes.TrimSpace(v), []byte("null"))) {
			return fmt.Errorf("missing required field %s", key)
		}
	}
	return nil
}

func plain(s string, max int) bool {
	return utf8.ValidString(s) && utf8.RuneCountInString(s) <= max &&
		!strings.ContainsAny(s, "<>") && strings.IndexFunc(s, func(r rune) bool { return unicode.IsControl(r) && r != '\n' && r != '\t' }) < 0
}

func (c *Client) decodeManifest(data []byte) (Manifest, Asset, *Failure) {
	m := Manifest{RestartRequired: true}
	bad := func(message string) (Manifest, Asset, *Failure) {
		return Manifest{}, Asset{}, fail(CodeManifest, message)
	}
	if !utf8.Valid(data) {
		return bad("manifest must be UTF-8 JSON")
	}
	var header struct {
		SchemaVersion int `json:"schema_version"`
	}
	if err := jsonwire.Unmarshal(data, &header); err != nil {
		return bad("manifest JSON: " + err.Error())
	}
	if header.SchemaVersion > 1 {
		f := fail(CodeUnsupported, "manifest schema is newer than this updater")
		f.Retry = "install the GUI manually from the official release page"
		return Manifest{}, Asset{}, f
	}
	if err := jsonwire.Unmarshal(data, &m); err != nil {
		return bad("manifest JSON: " + err.Error())
	}
	if err := required(data, "schema_version", "product", "channel", "version", "tag", "commit", "released", "min_updatable_from", "via", "assets", "notes", "runtime", "models", "settings", "advisories"); err != nil {
		return bad(err.Error())
	}
	if m.SchemaVersion != 1 || m.Product != "bubo" || m.Channel != "stable" {
		return bad("expected schema_version 1, product bubo, channel stable")
	}
	v, err := parseVersion(m.Version)
	if err != nil {
		return Manifest{}, Asset{}, fail(CodeVersionInvalid, err.Error())
	}
	min, err := parseVersion(m.MinUpdatableFrom)
	if err != nil || compare(min, v) >= 0 || m.Tag != "v"+m.Version || m.Released.IsZero() {
		return bad("invalid minimum version, tag, or release time")
	}
	commit, err := hex.DecodeString(m.Commit)
	if err != nil || len(commit) != 20 {
		return bad("commit must be 40 hexadecimal characters")
	}
	if m.Via != nil {
		via, err := parseVersion(*m.Via)
		if err != nil || compare(via, v) >= 0 || compare(via, min) < 0 {
			return bad("via must be a valid intermediate stable version")
		}
	}
	var fields map[string]json.RawMessage
	_ = jsonwire.Unmarshal(data, &fields)
	for key, keys := range map[string][]string{
		"notes": {"summary", "url"}, "runtime": {"id", "changed", "download_bytes"},
		"models": {"changed", "needs_prepare"}, "settings": {"migrate"},
	} {
		if err := required(fields[key], keys...); err != nil {
			return bad(key + ": " + err.Error())
		}
	}
	if !plain(m.Notes.Summary, 600) || m.Runtime.ID == "" || m.Runtime.DownloadBytes < 0 || (!m.Runtime.Changed && m.Runtime.DownloadBytes != 0) {
		return bad("invalid notes or runtime metadata")
	}
	if f := c.releaseURL(m.Notes.URL, releasePath+"tag/"+m.Tag); f != nil {
		return bad("notes URL: " + f.Message)
	}
	for _, a := range m.Advisories {
		if _, err := advisoryMatches(a.Versions, c.installed); err != nil || a.Text == "" || !plain(a.Text, 4096) {
			return bad("invalid advisory version range or text")
		}
	}
	for _, id := range m.Models.NeedsPrepare {
		if id == "" || !plain(id, 256) {
			return bad("invalid needs_prepare model identifier")
		}
	}
	var rawAssets []json.RawMessage
	_ = jsonwire.Unmarshal(fields["assets"], &rawAssets)
	var selected Asset
	packages := 0
	for i, a := range m.Assets {
		if err := required(rawAssets[i], "name", "kind", "url", "size", "sha256", "unpacked_bytes", "top_folder"); err != nil {
			return bad("asset: " + err.Error())
		}
		hash, err := hex.DecodeString(a.SHA256)
		if a.Name == "" || path.Base(a.Name) != a.Name || strings.ContainsAny(a.Name, "\\:%?#") || a.Size <= 0 || a.UnpackedBytes <= 0 || err != nil || len(hash) != 32 {
			return bad("invalid asset name, size, unpacked_bytes, or sha256")
		}
		if f := c.releaseURL(a.URL, releasePath+"download/"+m.Tag+"/"+a.Name); f != nil {
			return bad("asset URL: " + f.Message)
		}
		if a.Kind == "package" {
			expected := "bubo-" + m.Tag + "-windows-x64-gfx1151"
			if a.Name != expected+".zip" || (a.TopFolder != "" && a.TopFolder != expected) {
				return bad("package name or top_folder does not match the release tag")
			}
			packages++
			selected = a
			selected.SHA256 = strings.ToLower(a.SHA256)
		}
	}
	if packages != 1 {
		return bad("manifest must contain exactly one Windows package asset")
	}
	return m, selected, nil
}

func (c *Client) releaseURL(raw, expectedPath string) *Failure {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.Host != c.origin.Host || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawPath != "" || u.Path != expectedPath {
		return fail(CodeURL, "URL must use the configured repository's exact HTTPS release path")
	}
	return c.allowedURL(u)
}
