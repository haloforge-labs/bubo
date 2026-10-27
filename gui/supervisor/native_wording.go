//go:build windows

package supervisor

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// Native wording is shared by the View and the tray. The phase is the only
// classification of the server lifecycle used by the desktop UI.
type ViewPhase string

const PhaseCLIMissing ViewPhase = "cli_missing"

const NativeCrashSentence = "Bubo hit a problem it can't recover from and must close."

// FirstStartLabel is shown when the first CLI call waits more than a few
// seconds (Windows scans a freshly unpacked program on its first run).
const FirstStartLabel = "Starting for the first time…"

var phaseLabels = map[ViewPhase]string{
	"checking":          "Checking this computer…",
	"cli_missing":       "Bubo needs to be reinstalled",
	"failed":            "Stopped after an error",
	"stopping":          "Stopping…",
	"starting":          "Starting the server…",
	"stalled":           "Running, but not responding",
	"running_external":  "Running (started outside Bubo)",
	"running":           "Running",
	"preparing_engine":  "Downloading engine files",
	"downloading_model": "Downloading model",
	"needs_engine":      "Setup needed: download the engine files",
	"needs_model":       "Setup needed: download a model",
	"stopped":           "Stopped",
}

var slotLabels = map[string]string{
	"idle": "Idle", "prefilling": "Reading the prompt", "generating": "Writing a response",
}

func nativePhase(s Snapshot, models *ModelsState, saved Settings) ViewPhase {
	// A selected model does not need the default package that status reports.
	custom := saved.ModelPath != "" || saved.ManifestPath != ""
	if !s.Initialized {
		return "checking"
	}
	if s.LastError != nil && s.LastError.Code == "cli_missing" {
		return PhaseCLIMissing
	}
	switch s.State {
	case Failed:
		return "failed"
	case Stopping:
		return "stopping"
	case Starting:
		return "starting"
	case Ready:
		if s.Stall != nil {
			return "stalled"
		}
		return "running"
	case AttachedExternal:
		return "running_external"
	case Preparing:
		if s.Approval != nil {
			return readinessPhase(s.Status, custom)
		}
		if s.Progress != nil && s.Progress.Component == "model" {
			return "downloading_model"
		}
		return "preparing_engine"
	}
	if models != nil && models.Download != nil && models.Download.State == "running" {
		return "downloading_model"
	}
	return readinessPhase(s.Status, custom)
}

func readinessPhase(status *Status, custom bool) ViewPhase {
	if status == nil || !status.Readiness.Runtime {
		return "needs_engine"
	}
	if !status.Readiness.Model && !custom {
		return "needs_model"
	}
	return "stopped"
}

type CurrentModel struct {
	CatalogueID  string `json:"catalogue_id"`
	DisplayName  string `json:"display_name"`
	ModelPath    string `json:"model_path"`
	ManifestPath string `json:"manifest_path"`
}

func currentModel(s Snapshot, models *ModelsState, saved Settings) *CurrentModel {
	var identity *ActiveModel
	if s.Status != nil {
		identity = s.Status.ActiveModel
	}
	if identity == nil && s.Instance != nil {
		identity = s.Instance.ActiveModel
	}
	m := &CurrentModel{}
	if identity != nil {
		if identity.CatalogueID != nil {
			m.CatalogueID = *identity.CatalogueID
		}
		if identity.ModelPath != nil {
			m.ModelPath = *identity.ModelPath
		}
		if identity.ManifestPath != nil {
			m.ManifestPath = *identity.ManifestPath
		}
	}
	if identity == nil && s.Status != nil && s.Status.ModelSelection != nil {
		selection := s.Status.ModelSelection
		if selection.Model != nil {
			m.ModelPath = *selection.Model
		}
		if selection.Manifest != nil {
			m.ManifestPath = *selection.Manifest
			for _, prepared := range s.Status.PreparedModels {
				if prepared.ManifestPath != nil && sameSelectedManifest(m.ManifestPath, *prepared.ManifestPath, prepared.ID) {
					m.CatalogueID, m.DisplayName = prepared.ID, prepared.DisplayName
					break
				}
			}
		}
	}
	if saved.ModelPath != "" && s.State != Ready && s.State != AttachedExternal {
		if m.ModelPath != saved.ModelPath {
			m.CatalogueID = ""
		}
		m.ModelPath = saved.ModelPath
		m.ManifestPath = ""
	}
	if saved.ManifestPath != "" && s.State != Ready && s.State != AttachedExternal {
		if !sameSelectedManifest(m.ManifestPath, saved.ManifestPath, m.CatalogueID) {
			m.CatalogueID = ""
		}
		m.ManifestPath = saved.ManifestPath
		m.ModelPath = ""
	}
	if models != nil {
		if m.ManifestPath != "" && m.CatalogueID == "" {
			for _, prepared := range models.PreparedModels {
				if prepared.ManifestPath != nil && sameSelectedManifest(m.ManifestPath, *prepared.ManifestPath, prepared.ID) {
					m.CatalogueID, m.DisplayName = prepared.ID, prepared.DisplayName
					break
				}
			}
		}
		if m.ManifestPath != "" && m.CatalogueID == "" {
			for _, entry := range models.Catalogue {
				if strings.EqualFold(filepath.Base(filepath.Dir(m.ManifestPath)), entry.ID) {
					m.CatalogueID, m.DisplayName = entry.ID, entry.DisplayName
					break
				}
			}
		}
		for _, entry := range models.Catalogue {
			if entry.ID == m.CatalogueID {
				m.DisplayName = entry.DisplayName
				break
			}
		}
		if m.CatalogueID == "" && m.ModelPath == "" && m.ManifestPath == "" {
			for _, entry := range models.Catalogue {
				if entry.Role == "recommended" && entry.DefaultStatus == "default" {
					m.CatalogueID, m.DisplayName = entry.ID, entry.DisplayName
					break
				}
			}
		}
	}
	if m.DisplayName == "" {
		path := m.ModelPath
		if path == "" {
			path = m.ManifestPath
		}
		if path != "" {
			m.DisplayName = filepath.Base(path)
		}
	}
	if m.CatalogueID == "" && m.ModelPath == "" && m.ManifestPath == "" {
		return nil
	}
	return m
}

// The CLI may report the same prepared package through a user-local link and
// its resolved model store. The package directory is its catalogue ID.
func sameSelectedManifest(selected, prepared, id string) bool {
	if strings.EqualFold(filepath.Clean(selected), filepath.Clean(prepared)) {
		return true
	}
	return id != "" &&
		strings.EqualFold(filepath.Base(selected), filepath.Base(prepared)) &&
		strings.EqualFold(filepath.Base(filepath.Dir(selected)), id) &&
		strings.EqualFold(filepath.Base(filepath.Dir(prepared)), id)
}

// FitVerdict projects only the CLI's estimate and its row for the requested
// session count. It never derives admission from byte counts.
type FitVerdict struct {
	Class                 string   `json:"class"`
	Sentence              string   `json:"sentence"`
	ContextTokens         uint64   `json:"context_tokens"`
	Sessions              uint64   `json:"sessions"`
	FreeAfterGiB          *float64 `json:"free_after_gib"`
	HostFreeAfterSpillGiB *float64 `json:"host_free_after_spill_gib"`
	SuggestedContext      *uint64  `json:"suggested_context"`
}

// NoVerdictSentence is the one line for a setup the CLI cannot judge before the
// server starts: serve checks memory at load, and exit 19 reports a refusal.
const NoVerdictSentence = "Not checked: you can still start the server."

// fitVerdict is the fit answer the app may state, or nil for no verdict. A "fits"
// from an estimate that is only a lower bound (fit_scope other than
// conservative_startup_estimate: a header estimate) is no verdict, because
// serve's load check uses the real numbers. A "does not fit" is a verdict at
// any scope: a lower bound that does not fit cannot fit. fit_verified is not
// used: it is always false in contract v1.
func (e *MemoryEstimate) fitVerdict() *bool {
	if e == nil || e.Fits == nil {
		return nil
	}
	if *e.Fits && e.FitScope != "conservative_startup_estimate" {
		return nil
	}
	return e.Fits
}

// noVerdictFit is the line for a finished estimate that gives no verdict. It
// never carries the CLI's reason text, which holds developer words.
func noVerdictFit() FitVerdict {
	return FitVerdict{Class: "fit_unknown", Sentence: NoVerdictSentence}
}

// buildFitFor is the fit of one `model_estimate` answer. A none_yet answer is
// finished: it has no numbers and no verdict. Only a missing answer is "in
// flight", and that is buildFit(nil).
func buildFitFor(info ModelEstimateInfo) FitVerdict {
	if info.State == "none_yet" {
		return noVerdictFit()
	}
	return buildFit(info.MemoryEstimate)
}

func buildFit(estimate *MemoryEstimate) FitVerdict {
	f := FitVerdict{Class: "unavailable", Sentence: "Checking memory…"}
	if estimate != nil && estimate.Fits != nil && *estimate.Fits && estimate.fitVerdict() == nil {
		// A lower bound that fits: nothing to claim before the server starts.
		return noVerdictFit()
	}
	if estimate == nil || estimate.Fits == nil {
		// The CLI could not say whether it fits: its reason is the statement.
		if estimate != nil {
			for _, reason := range []*string{estimate.FitReason, estimate.RecommendedContextReason} {
				if reason != nil && *reason != "" {
					f.Class, f.Sentence = "fit_unknown", *reason
					break
				}
			}
		}
		return f
	}
	f.ContextTokens, f.Sessions = uint64(estimate.ContextTokens), uint64(estimate.Sessions)
	var row *RecommendedContextRow
	if estimate.RecommendedContext != nil {
		for i := range estimate.RecommendedContext.BySessions {
			candidate := &estimate.RecommendedContext.BySessions[i]
			if candidate.Sessions == estimate.Sessions {
				row = candidate
				break
			}
		}
	}
	if row != nil {
		free := row.FreeAfterGiB
		f.FreeAfterGiB = &free
		f.HostFreeAfterSpillGiB = row.HostFreeAfterSpillGiB
		if row.FreeAfterGrowthGiB != nil {
			growth := *row.FreeAfterGrowthGiB
			f.FreeAfterGiB = &growth
		}
		if row.Fits && uint64(row.ContextPerSession) < f.ContextTokens {
			suggestion := uint64(row.ContextPerSession)
			f.SuggestedContext = &suggestion
		}
	}
	// The numbers of the context the user chose are the top level of the
	// estimate; the row is the recommended context's startup headroom.
	if estimate.FreeAfterGrowthGiB != nil {
		free := *estimate.FreeAfterGrowthGiB
		f.FreeAfterGiB = &free
	}
	if estimate.HostFreeAfterSpillGiB != nil {
		host := *estimate.HostFreeAfterSpillGiB
		f.HostFreeAfterSpillGiB = &host
	}
	if !*estimate.Fits {
		if row != nil && !row.Fits {
			f.Class = "memory_does_not_fit_any"
		} else {
			f.Class = "memory_does_not_fit"
		}
		f.Sentence = "These settings need more memory than this computer has, so Bubo can’t start them." + memoryFailureGuidance(estimate)
		return f
	}
	if estimate.Recommended != nil && !*estimate.Recommended || len(estimate.RecommendationFailures) > 0 {
		f.Class = "memory_tight"
		f.Sentence = "The server would start, but there is little room for the context cache (the KV cache), which grows as a conversation gets longer. If it runs out, the server stops with an out-of-memory error, or Windows slows down." + memoryFailureGuidance(estimate)
		return f
	}
	f.Class, f.Sentence = "fits", "Fits."
	if f.FreeAfterGiB != nil {
		f.Sentence = fmt.Sprintf("Fits. %.0f GiB of memory would be left.", *f.FreeAfterGiB)
	}
	return f
}

// Use the CLI's failure categories as reported; do not infer a verdict from
// the free-memory numbers or recalculate its recommendation thresholds.
func memoryFailureGuidance(estimate *MemoryEstimate) string {
	var host, gpu bool
	for _, failure := range estimate.RecommendationFailures {
		host = host || failure == RecommendationHostFloor
		gpu = gpu || failure == RecommendationGPUHeadroom
	}
	text := ""
	if host {
		text += " System memory is low; close other apps."
	}
	if gpu {
		text += " GPU memory is low."
	}
	return text
}

// sessionCount is "1 session" or "N sessions".
func sessionCount(n uint64) string {
	if n == 1 {
		return "1 session"
	}
	return strconv.FormatUint(n, 10) + " sessions"
}

func groupedCount(value uint64) string {
	digits := strconv.FormatUint(value, 10)
	for i := len(digits) - 3; i > 0; i -= 3 {
		digits = digits[:i] + "," + digits[i:]
	}
	return digits
}
