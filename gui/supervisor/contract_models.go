package supervisor

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Typed view of the `models` and `model_estimate` events (docs/CLI-CONTRACT.md,
// "Models and the model catalogue"). The JSON schema checks the shape; the
// checks below add the rules a schema cannot state. Rows keep their raw CLI
// JSON (Raw) so the window gets every field the CLI sent, including additive
// ones. The GUI computes no estimate and no list entry.

// ModelEstimateInfo is the estimate object of one row, and the body of a
// model_estimate event.
type ModelEstimateInfo struct {
	State          string                     `json:"estimate_state"`
	Reason         *string                    `json:"reason"`
	EvidenceLabel  *string                    `json:"evidence_label"`
	FitVerified    bool                       `json:"fit_verified"`
	FitInputs      map[string]json.RawMessage `json:"fit_inputs"`
	Notes          []string                   `json:"notes"`
	MemoryEstimate *MemoryEstimate            `json:"memory_estimate"`
}

func (e ModelEstimateInfo) hasMemoryEstimate() bool {
	return e.MemoryEstimate != nil
}

// check states the rules between the fields: an estimate has numbers and no
// reason, and "none yet" has a reason and no numbers.
func (e ModelEstimateInfo) check(where string) error {
	if e.FitVerified {
		return protocol(where + ": fit_verified must be false")
	}
	switch e.State {
	case "estimated":
		if !e.hasMemoryEstimate() || e.Reason != nil {
			return protocol(where + ": an estimate needs memory_estimate and no reason")
		}
	case "none_yet":
		if e.hasMemoryEstimate() || e.Reason == nil || strings.TrimSpace(*e.Reason) == "" {
			return protocol(where + ": no estimate needs a reason and no memory_estimate")
		}
	default:
		return protocol(where + ": unknown estimate_state")
	}
	return nil
}

type ModelLicence struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	URL              string  `json:"url"`
	PromptLine       string  `json:"prompt_line"`
	Note             *string `json:"note"`
	VerifiedUpstream bool    `json:"verified_upstream"`
}
type ModelFile struct {
	Name         string  `json:"name"`
	Role         string  `json:"role"`
	Bytes        uint64  `json:"bytes"`
	SHA256       string  `json:"sha256"`
	URL          string  `json:"url"`
	Optional     bool    `json:"optional"`
	Downloadable bool    `json:"downloadable"`
	LicenceText  *string `json:"licence_text"`
}

// CatalogueEntry is one `catalogue` row.
type CatalogueEntry struct {
	ID                string            `json:"id"`
	DisplayName       string            `json:"display_name"`
	TechnicalName     string            `json:"technical_name"`
	About             string            `json:"about"`
	SelectionLabel    string            `json:"selection_label"`
	AboutLinks        []string          `json:"about_links"`
	GPUBytes          *uint64           `json:"gpu_bytes"`
	SystemMemoryBytes *uint64           `json:"system_memory_bytes"`
	SystemMemoryNote  *string           `json:"system_memory_note"`
	SizeSplitSource   *string           `json:"size_split_source"`
	Role              string            `json:"role"`
	Architecture      string            `json:"architecture"`
	Set               *string           `json:"set"`
	DefaultStatus     string            `json:"default_status"`
	FamilyGroup       *string           `json:"family_group"`
	DownloadEnabled   bool              `json:"download_enabled"`
	DisabledReason    *string           `json:"disabled_reason"`
	Packaging         string            `json:"packaging"`
	Licence           ModelLicence      `json:"licence"`
	Files             []ModelFile       `json:"files"`
	DownloadBytes     uint64            `json:"download_bytes"`
	PreparedState     string            `json:"prepared_state"`
	LocalFolder       string            `json:"local_folder"`
	Estimate          ModelEstimateInfo `json:"estimate"`
	raw               json.RawMessage
}

func (c *CatalogueEntry) UnmarshalJSON(b []byte) error {
	type plain CatalogueEntry
	if err := json.Unmarshal(b, (*plain)(c)); err != nil {
		return err
	}
	c.raw = append(json.RawMessage(nil), b...)
	return nil
}

// MarshalJSON preserves the row after ParseEvent's numeric normalization.
func (c CatalogueEntry) MarshalJSON() ([]byte, error) {
	if len(c.raw) > 0 {
		return c.raw, nil
	}
	type plain CatalogueEntry
	return json.Marshal(plain(c))
}

// ScannedModel is one `scanned` row (a GGUF file found in a chosen folder).
type ScannedModel struct {
	Path           string            `json:"path"`
	FileName       string            `json:"file_name"`
	Folder         string            `json:"folder"`
	SizeBytes      uint64            `json:"size_bytes"`
	ShardsFound    uint64            `json:"shards_found"`
	ShardsExpected uint64            `json:"shards_expected"`
	Architecture   *string           `json:"architecture"`
	ModelName      *string           `json:"model_name"`
	NativeContext  *uint64           `json:"native_context_tokens"`
	Kind           string            `json:"kind"`
	Runnable       bool              `json:"runnable"`
	Reason         *string           `json:"reason"`
	Route          *string           `json:"route"`
	Estimate       ModelEstimateInfo `json:"estimate"`
	raw            json.RawMessage
}

func (s *ScannedModel) UnmarshalJSON(b []byte) error {
	type plain ScannedModel
	if err := json.Unmarshal(b, (*plain)(s)); err != nil {
		return err
	}
	s.raw = append(json.RawMessage(nil), b...)
	return nil
}
func (s ScannedModel) MarshalJSON() ([]byte, error) {
	if len(s.raw) > 0 {
		return s.raw, nil
	}
	type plain ScannedModel
	return json.Marshal(plain(s))
}

type ScanWarning struct {
	Folder string `json:"folder"`
	Reason string `json:"reason"`
}
type ModelsListing struct {
	ModelsRoot          string          `json:"models_root"`
	AvailableBytes      *uint64         `json:"available_bytes"`
	ScanDepthLimit      uint64          `json:"scan_depth_limit"`
	EstimateDefaults    json.RawMessage `json:"estimate_defaults"`
	CatalogueVersion    string          `json:"catalogue_version"`
	PackageFilesSkipped uint64          `json:"package_files_skipped"`
	ScannedFolders      []string        `json:"scanned_folders"`
}

// Models is the payload of the `models` event.
type Models struct {
	Listing        ModelsListing    `json:"listing"`
	PreparedModels []PreparedModel  `json:"prepared_models"`
	Catalogue      []CatalogueEntry `json:"catalogue"`
	Scanned        []ScannedModel   `json:"scanned"`
	ScanWarnings   []ScanWarning    `json:"scan_warnings"`
}

// ModelEstimate is the payload of the `model_estimate` event.
type ModelEstimate struct {
	ModelEstimateInfo
	Subject     string `json:"subject"`
	SubjectKind string `json:"subject_kind"`
}

// check holds the rules the JSON schema cannot state. Every failure is a
// Protocol error: a list that breaks them is not shown.
func (m *Models) check() error {
	ids := map[string]bool{}
	for _, c := range m.Catalogue {
		where := "catalogue " + c.ID
		if ids[c.ID] {
			return protocol("duplicate catalogue id " + c.ID)
		}
		ids[c.ID] = true
		var sum uint64
		for _, f := range c.Files {
			if f.Downloadable {
				sum += f.Bytes
				if c.DownloadEnabled && (f.SHA256 == "" || !strings.HasPrefix(f.URL, "https://")) {
					return protocol(where + ": a downloadable file needs a sha256 and an https url")
				}
			}
		}
		if sum != c.DownloadBytes {
			return protocol(where + ": download_bytes is not the sum of the downloadable files")
		}
		if c.DownloadEnabled {
			if c.Licence.PromptLine == "" || !strings.HasPrefix(c.Licence.URL, "https://") || c.Licence.ID == "" {
				return protocol(where + ": an enabled entry needs the licence line and url")
			}
		} else if c.DisabledReason == nil || strings.TrimSpace(*c.DisabledReason) == "" {
			return protocol(where + ": a disabled entry needs a reason")
		}
		if err := c.Estimate.check(where); err != nil {
			return err
		}
	}
	paths := map[string]bool{}
	for _, s := range m.Scanned {
		where := "scanned " + s.Path
		if paths[strings.ToLower(s.Path)] {
			return protocol("duplicate scanned path " + s.Path)
		}
		paths[strings.ToLower(s.Path)] = true
		if s.ShardsFound > s.ShardsExpected {
			return protocol(where + ": more shards found than expected")
		}
		if !s.Runnable && (s.Reason == nil || strings.TrimSpace(*s.Reason) == "") {
			return protocol(where + ": a file that cannot run needs a reason")
		}
		if s.Runnable && s.Kind != "model" {
			return protocol(where + ": only a model file can be runnable")
		}
		if err := s.Estimate.check(where); err != nil {
			return err
		}
	}
	return nil
}

// FileByName returns the catalogue entry with this id.
func (m *Models) Entry(id string) (CatalogueEntry, bool) {
	for _, c := range m.Catalogue {
		if c.ID == id {
			return c, true
		}
	}
	return CatalogueEntry{}, false
}

func (m *ModelEstimate) check() error {
	if m.Subject == "" {
		return protocol("model_estimate has no subject")
	}
	return m.ModelEstimateInfo.check(fmt.Sprintf("estimate %s", m.Subject))
}
