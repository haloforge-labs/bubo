package supervisor

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
)

// These types follow docs/CLI-CONTRACT.md and tests/cli/contract/event.schema.json
// at schema_version 1. The CLI is the single source of truth. A null on the wire
// is an unavailable value: it is a nil pointer, a nil slice or a nil map here and
// never a zero. Unknown keys are ignored (encoding/json). Required keys are
// checked by the embedded schema before any of these types is filled.

// SupportedSchemaVersion is the only major version this GUI reads.
const SupportedSchemaVersion = 1

type Product struct {
	Name          string `json:"name"`
	Version       string `json:"version"`
	Revision      string `json:"revision"`
	EngineVersion string `json:"engine_version"`
}
type Profile struct {
	Name  string `json:"name"`
	Label string `json:"label"`
	GUI   bool   `json:"gui"`
}
type Capabilities struct {
	SchemaVersion     int       `json:"schema_version"`
	PrepareJSONL      bool      `json:"prepare_jsonl"`
	DiagnoseJSON      bool      `json:"diagnose_json"`
	StatusJSON        bool      `json:"status_json"`
	InstanceControl   bool      `json:"instance_control"`
	StatusWatch       bool      `json:"status_watch"`
	Settings          bool      `json:"settings,omitempty"`
	Profiles          []Profile `json:"profiles,omitempty"`
	MemoryEstimate    bool      `json:"memory_estimate,omitempty"`
	ConnectionHelpers bool      `json:"connection_helpers,omitempty"`
	SupportBundle     bool      `json:"support_bundle,omitempty"`
	SwitchRegistry    bool      `json:"switch_registry,omitempty"`
	RequestHistory    bool      `json:"request_history,omitempty"`
	PerSlotContext    bool      `json:"per_slot_context,omitempty"`
}
type CLIError struct {
	Code      string `json:"code"`
	Label     string `json:"label,omitempty"`
	Component string `json:"component"`
	Stage     string `json:"stage"`
	Retryable bool   `json:"retryable"`
	Remedy    string `json:"remedy"`
	Switch    string `json:"switch,omitempty"`
	// Detail is the cause in the CLI's own words (an optional CLI-CONTRACT
	// field). It shows only under Details.
	Detail string `json:"detail,omitempty"`
	// The memory_insufficient error (serve exit 19) carries what did not fit.
	NeededBytes     *uint64 `json:"needed_bytes,omitempty"`
	AvailableBytes  *uint64 `json:"available_bytes,omitempty"`
	Sessions        *uint64 `json:"sessions,omitempty"`
	Context         *uint64 `json:"context,omitempty"`
	PerSessionBytes *uint64 `json:"per_session_bytes,omitempty"`
	FloorBytes      *uint64 `json:"floor_bytes,omitempty"`
}

// MemoryShortfall is the part of a memory_insufficient error the app shows:
// the sessions and context that were asked for, and the bytes they need and
// that were available.
type MemoryShortfall struct {
	NeededBytes    uint64 `json:"needed_bytes"`
	AvailableBytes uint64 `json:"available_bytes"`
	Sessions       uint64 `json:"sessions"`
	Context        uint64 `json:"context"`
}

// memoryShortfall reads the shortfall from an error record, or nil when the
// record does not carry all four numbers.
func memoryShortfall(c *CLIError) *MemoryShortfall {
	if c.Code != "memory_insufficient" || c.NeededBytes == nil || c.AvailableBytes == nil || c.Sessions == nil || c.Context == nil {
		return nil
	}
	return &MemoryShortfall{NeededBytes: *c.NeededBytes, AvailableBytes: *c.AvailableBytes, Sessions: *c.Sessions, Context: *c.Context}
}

type Preflight struct {
	Component string `json:"component"`
	Phase     string `json:"phase"`
}
type Source struct {
	FileID     string `json:"file_id"`
	URL        string `json:"url"`
	Size       uint64 `json:"size"`
	SHA256     string `json:"sha256"`
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Path       string `json:"path"`
	LicenseID  string `json:"license_id"`
}
type License struct {
	Path       string `json:"path"`
	Text       string `json:"text"`
	SHA256     string `json:"sha256"`
	TextOffset uint64 `json:"text_offset"`
	TextBytes  uint64 `json:"text_bytes"`
}
type Volume struct {
	Name                   string  `json:"name"`
	FreeBytes              uint64  `json:"free_bytes"`
	RequiredBytes          uint64  `json:"required_bytes"`
	BytesToWrite           *uint64 `json:"bytes_to_write,omitempty"`
	SafetyMarginBytes      *uint64 `json:"safety_margin_bytes,omitempty"`
	FreeAfterDownloadBytes *uint64 `json:"free_after_download_bytes,omitempty"`
}
type Plan struct {
	Component  string                     `json:"component"`
	Ready      bool                       `json:"ready"`
	Sources    []Source                   `json:"sources"`
	Licenses   []License                  `json:"licenses"`
	Paths      map[string]json.RawMessage `json:"paths"`
	Volumes    []Volume                   `json:"volumes"`
	Digest     string                     `json:"plan_digest"`
	RecordKind string                     `json:"record_kind"`
	TextOffset uint64                     `json:"text_offset"`
	TextBytes  uint64                     `json:"text_bytes"`
	Part       uint64                     `json:"part"`
	Parts      uint64                     `json:"parts"`
	FileStates []string                   `json:"file_states"`
	Replaces   []string                   `json:"replaces,omitempty"`
}

// Approval is the approval_required payload. Question is an additive CLI
// sentence for the consent dialog; the GUI shows it as sent.
type Approval struct {
	Digest     string   `json:"plan_digest"`
	Components []string `json:"components"`
	Question   string   `json:"question,omitempty"`
}
type Progress struct {
	Component    string  `json:"component"`
	Phase        string  `json:"phase"`
	FileID       string  `json:"file_id"`
	Done         uint64  `json:"bytes_done"`
	Total        *uint64 `json:"bytes_total"`
	ResumeOffset uint64  `json:"resume_offset"`
	Verification string  `json:"verification_method"`
	// Overall is the whole job. The supervisor adds it, so the progress event
	// and the View agree. Done and Total above describe the current file.
	Overall *JobProgress `json:"overall,omitempty"`
}
type Result struct {
	ExitCode     uint64  `json:"exit_code"`
	ManifestPath *string `json:"manifest_path"`
	Prepared     bool    `json:"prepared"`
	// PlanOnly is true for `prepare model --plan`: nothing was written.
	PlanOnly bool `json:"plan_only,omitempty"`
}
type Check struct {
	Name    string                     `json:"name"`
	Status  string                     `json:"status"`
	Message string                     `json:"message"`
	Details map[string]json.RawMessage `json:"details"`
}
type Diagnose struct {
	SchemaVersion  string   `json:"schemaVersion"`
	EngineRevision string   `json:"engineRevision"`
	Timestamp      string   `json:"timestamp"`
	Status         string   `json:"status"`
	Checks         []Check  `json:"checks"`
	Warnings       []string `json:"warnings"`
	Errors         []string `json:"errors"`
}
type Artifact map[string]json.RawMessage

// SupportBundle is the diagnose --bundle result (preview or saved).
type SupportBundle struct {
	SchemaVersion int               `json:"schema_version"`
	Saved         bool              `json:"saved"`
	BundleSHA256  string            `json:"bundle_sha256"`
	ZipBytes      uint64            `json:"zip_bytes"`
	Files         []string          `json:"files"`
	TextPreview   map[string]string `json:"text_preview"`
	SHA256        map[string]string `json:"sha256"`
}

// Identity names one server process. All four fields are required on the wire.
// CreationTime is the exact FILETIME as canonical decimal text.
type Identity struct {
	ID           string `json:"instance_id"`
	PID          uint32 `json:"pid"`
	CreationTime string `json:"process_creation_time"`
	ExePath      string `json:"exe_path"`
}

// Setting is one effective_settings row. Requested and Effective keep their
// raw JSON because the CLI types them per row (number, string, boolean, null).
type Setting struct {
	Requested json.RawMessage `json:"requested"`
	Effective json.RawMessage `json:"effective"`
	Source    string          `json:"source"`
	Reason    *string         `json:"reason"`
	Allowed   json.RawMessage `json:"allowed,omitempty"`
}

// SettingRow returns the named row. The profile row can be a plain string; then
// only Effective is set. ok is false when the row is absent or not a row.
func SettingRow(settings map[string]json.RawMessage, key string) (Setting, bool) {
	raw, ok := settings[key]
	if !ok || len(raw) == 0 || raw[0] == 'n' {
		return Setting{}, false
	}
	if raw[0] == '"' {
		return Setting{Requested: raw, Effective: raw}, true
	}
	var s Setting
	if json.Unmarshal(raw, &s) != nil || s.Effective == nil && s.Requested == nil && s.Source == "" {
		return Setting{}, false
	}
	return s, true
}

// Count is an unsigned integer that also reads a zero-fraction number such as
// 87790000000.0, which JSON Schema accepts as an integer.
type Count uint64

func (c *Count) UnmarshalJSON(b []byte) error {
	r, ok := new(big.Rat).SetString(string(b))
	if !ok || !r.IsInt() || r.Sign() < 0 || !r.Num().IsUint64() {
		return fmt.Errorf("not an unsigned integer: %s", b)
	}
	*c = Count(r.Num().Uint64())
	return nil
}

type MemoryAmount struct {
	Bytes    Count   `json:"bytes"`
	GiB      float64 `json:"gib"`
	Evidence string  `json:"evidence"`
}
type MemoryRange struct {
	Min MemoryAmount `json:"min"`
	Max MemoryAmount `json:"max"`
}

type MemoryGPUBudget struct {
	Total                      *MemoryAmount `json:"total,omitempty"`
	DedicatedVideoMemory       *MemoryAmount `json:"dedicated_video_memory,omitempty"`
	DedicatedVideoMemoryReason *string       `json:"dedicated_video_memory_reason,omitempty"`
	DedicatedVideoMemorySource string        `json:"dedicated_video_memory_source,omitempty"`
	Available                  *MemoryAmount `json:"available"`
	Need                       MemoryAmount  `json:"need"`
	NeedMax                    *MemoryAmount `json:"need_max,omitempty"`
	Reserve                    MemoryAmount  `json:"reserve"`
}

type MemoryHostBudget struct {
	Total     *MemoryAmount `json:"total,omitempty"`
	Available *MemoryAmount `json:"available"`
	Need      MemoryAmount  `json:"need"`
	Floor     MemoryAmount  `json:"floor"`
}

// RecommendationFailure is a CLI check identifier, never inferred from budgets.
type RecommendationFailure string

const (
	RecommendationGPUHeadroom RecommendationFailure = "gpu_headroom"
	RecommendationHostFloor   RecommendationFailure = "host_floor"
	RecommendationUnconfirmed RecommendationFailure = "unconfirmed"
)

type RecommendedContextRow struct {
	RecommendationFailures []RecommendationFailure `json:"recommendation_failures,omitempty"`
	CacheAtClampBytes      *Count                  `json:"cache_at_clamp_bytes,omitempty"`
	HostSpillBytes         *Count                  `json:"host_spill_bytes,omitempty"`
	FreeAfterGrowthGiB     *float64                `json:"free_after_growth_gib,omitempty"`
	HostFreeAfterSpillGiB  *float64                `json:"host_free_after_spill_gib,omitempty"`
	Sessions               Count                   `json:"sessions"`
	ContextPerSession      Count                   `json:"context_per_session"`
	Fits                   bool                    `json:"fits"`
	Recommended            *bool                   `json:"recommended,omitempty"`
	TotalMinBytes          Count                   `json:"total_min_bytes"`
	FreeAfterGiB           float64                 `json:"free_after_gib"`
	Reason                 *string                 `json:"reason,omitempty"`
}

type RecommendedContext struct {
	AvailableBytes       Count                   `json:"available_bytes"`
	AvailableGiB         *float64                `json:"available_gib,omitempty"`
	AvailableEvidence    string                  `json:"available_evidence,omitempty"`
	MarginBytes          Count                   `json:"margin_bytes"`
	MarginGiB            *float64                `json:"margin_gib,omitempty"`
	MarginRule           string                  `json:"margin_rule,omitempty"`
	RangeBasis           string                  `json:"range_basis"`
	StepTokens           Count                   `json:"step_tokens"`
	MinimumContextTokens Count                   `json:"minimum_context_tokens"`
	ContextMaxTokens     Count                   `json:"context_max_tokens"`
	MaxSessions          Count                   `json:"max_sessions"`
	BySessions           []RecommendedContextRow `json:"by_sessions"`
}

// MemoryEstimate is the CLI's planning estimate. FitVerified is always false in
// version 1. The GUI shows the CLI numbers and never computes its own.
type MemoryEstimate struct {
	RecommendationFailures       []RecommendationFailure `json:"recommendation_failures,omitempty"`
	CacheAtClampBytes            *Count                  `json:"cache_at_clamp_bytes,omitempty"`
	HostSpillBytes               *Count                  `json:"host_spill_bytes,omitempty"`
	FreeAfterGrowthGiB           *float64                `json:"free_after_growth_gib,omitempty"`
	HostFreeAfterSpillGiB        *float64                `json:"host_free_after_spill_gib,omitempty"`
	TableVersion                 int                     `json:"table_version"`
	Evidence                     string                  `json:"evidence"`
	ByteUnit                     string                  `json:"byte_unit"`
	DisplayUnit                  string                  `json:"display_unit"`
	FitVerified                  bool                    `json:"fit_verified"`
	Reason                       string                  `json:"reason"`
	Weights                      MemoryAmount            `json:"weights"`
	PerSessionState              MemoryAmount            `json:"per_session_state"`
	SessionStateTotal            MemoryAmount            `json:"session_state_total"`
	Sessions                     Count                   `json:"sessions"`
	ContextTokens                Count                   `json:"context_tokens"`
	ScratchPhaseAllowance        MemoryRange             `json:"scratch_phase_allowance"`
	HostCacheSnapshotAllowance   MemoryAmount            `json:"host_cache_snapshot_allowance"`
	Reserve                      MemoryAmount            `json:"reserve"`
	TotalRange                   MemoryRange             `json:"total_range"`
	CarveBudget                  *MemoryAmount           `json:"carve_budget"`
	HostBudget                   *MemoryAmount           `json:"host_budget"`
	ReferenceCoefficient         json.RawMessage         `json:"reference_coefficient"`
	Formula                      string                  `json:"formula"`
	GPU                          *MemoryGPUBudget        `json:"gpu,omitempty"`
	Host                         *MemoryHostBudget       `json:"host,omitempty"`
	Fits                         *bool                   `json:"fits"`
	FitReason                    *string                 `json:"fit_reason"`
	FitScope                     string                  `json:"fit_scope,omitempty"`
	CacheMode                    string                  `json:"cache_mode,omitempty"`
	DeviceCacheSnapshotAllowance *MemoryAmount           `json:"device_cache_snapshot_allowance,omitempty"`
	RecommendedContext           *RecommendedContext     `json:"recommended_context"`
	RecommendedContextReason     *string                 `json:"recommended_context_reason"`
	PerSessionVision             *MemoryAmount           `json:"per_session_vision,omitempty"`
	VisionSessionTotal           *MemoryAmount           `json:"vision_session_total,omitempty"`
	Recommended                  *bool                   `json:"recommended"`
	RecommendationReason         *string                 `json:"recommendation_reason"`
}

// MemoryUse is the status memory object (process and host bytes).
type MemoryUse struct {
	RSSBytes           *uint64 `json:"rss_bytes"`
	HostAvailableBytes *uint64 `json:"host_available_bytes"`
	Unit               string  `json:"unit"`
	Evidence           string  `json:"evidence,omitempty"`
}

type NumDefault struct {
	Value  *float64 `json:"value"`
	Source string   `json:"source"`
}
type StrDefault struct {
	Value  *string `json:"value"`
	Source string  `json:"source"`
}
type InferenceDefaults struct {
	Temperature     NumDefault `json:"temperature"`
	TopP            NumDefault `json:"top_p"`
	TopK            NumDefault `json:"top_k"`
	MinP            NumDefault `json:"min_p"`
	RepeatPenalty   NumDefault `json:"repeat_penalty"`
	PresencePenalty NumDefault `json:"presence_penalty"`
	Seed            NumDefault `json:"seed"`
	ReasoningEffort StrDefault `json:"reasoning_effort"`
	Think           StrDefault `json:"think"`
	MaxTokens       NumDefault `json:"max_tokens"`
}
type BiasEntry struct {
	Value  float64 `json:"value"`
	Source string  `json:"source"`
}
type EffortBiases struct {
	Low    BiasEntry `json:"low"`
	Medium BiasEntry `json:"medium"`
	High   BiasEntry `json:"high"`
	XHigh  BiasEntry `json:"xhigh"`
}
type ThinkMarkerBias struct {
	Value     float64         `json:"value"`
	Source    string          `json:"source"`
	Requested json.RawMessage `json:"requested"`
	ByEffort  EffortBiases    `json:"by_effort"`
}
type LoopBreaker struct {
	Value     bool            `json:"value"`
	Source    string          `json:"source"`
	Requested json.RawMessage `json:"requested"`
}
type EntropyGate struct {
	Available bool `json:"available"`
}

// RequestRow is one content-free request completion (status ring and the
// request_completed event). No prompt, text or client address exists here.
type RequestRow struct {
	ID               string   `json:"id"`
	Time             string   `json:"time"`
	Slot             *uint64  `json:"slot"`
	PromptTokens     *uint64  `json:"prompt_tokens"`
	CompletionTokens *uint64  `json:"completion_tokens"`
	TTFTMS           *float64 `json:"ttft_ms"`
	PrefillTPS       *float64 `json:"prefill_tps"`
	DecodeTPS        *float64 `json:"decode_tps"`
	CachedTokens     *uint64  `json:"cached_tokens"`
	CacheKind        *string  `json:"cache_kind"`
	FinishReason     *string  `json:"finish_reason"`
	Outcome          string   `json:"outcome"`
	// Status and ErrorCode say why a request that did not complete was
	// refused or failed (optional CLI-CONTRACT fields).
	Status    *uint64 `json:"status"`
	ErrorCode *string `json:"error_code"`
	// Refusal is present on rows with outcome "refused" (memory guard).
	Refusal *RequestRefusal `json:"refusal,omitempty"`
}

// RequestRefusal names why the memory guard refused a request. The byte
// counts are null for memory_pressure and for a stream stopped midway.
type RequestRefusal struct {
	Code        string  `json:"code"`
	NeededBytes *uint64 `json:"needed_bytes"`
	FreeBytes   *uint64 `json:"free_bytes"`
}
type PreparedCapabilities struct {
	ContextMax *uint64 `json:"context_max"`
	MTP        bool    `json:"mtp"`
	Twin       bool    `json:"twin"`
	MMProj     bool    `json:"mmproj"`
}
type PreparedModel struct {
	ID           string  `json:"id"`
	DisplayName  string  `json:"display_name"`
	ManifestPath *string `json:"manifest_path"`
	// ModelPath is the absolute path of the main model file (CLI-CONTRACT:
	// `models list`). Null for older CLIs, which report the manifest only.
	ModelPath     *string              `json:"model_path"`
	SizeBytes     uint64               `json:"size_bytes"`
	PreparedState string               `json:"prepared_state"`
	Capabilities  PreparedCapabilities `json:"capabilities"`
	// RepairOutput is the --output that checks and repairs this package. The CLI
	// names it (null when the package is not in prepare's layout, so the GUI
	// offers no repair). The GUI does no folder arithmetic.
	RepairOutput *string `json:"repair_output"`
}
type PiModel struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	ContextWindow uint64 `json:"contextWindow"`
}
type PiProvider struct {
	BaseURL string    `json:"baseUrl"`
	API     string    `json:"api"`
	APIKey  string    `json:"apiKey"`
	Models  []PiModel `json:"models"`
}

// PiSnippet is the models.json provider fragment. APIKey is the dummy key
// "bubo-loopback" or the environment variable name "BUBO_API_KEY"; the CLI
// never sends a real key.
type PiSnippet struct {
	Providers struct {
		Bubo PiProvider `json:"bubo"`
	} `json:"providers"`
}
type Connection struct {
	Endpoint      string  `json:"endpoint"`
	ModelID       string  `json:"model_id"`
	ContextTokens *uint64 `json:"context_tokens"`
	Authenticated bool    `json:"authenticated"`
}
type EffectiveSwitch struct {
	Name      string          `json:"name"`
	Label     string          `json:"label,omitempty"`
	Effective json.RawMessage `json:"effective"`
	Source    string          `json:"source"`
	Reason    *string         `json:"reason"`
	Audience  string          `json:"audience"`
	Group     string          `json:"group"`
}
type StateEntry struct {
	Type  string `json:"type"`
	Phase string `json:"phase"`
	Time  string `json:"time"`
}

// RuntimeDetail is the set of optional, additive runtime objects that status
// and the ready event share. Every pointer or slice can be nil (unavailable).
type RuntimeDetail struct {
	NPUPrefillRuntime    *NPUPrefillRuntime `json:"npu_prefill_runtime,omitempty"`
	ActiveModel          *ActiveModel       `json:"active_model"`
	ClientSnippets       []ClientSnippet    `json:"client_snippets"`
	Endpoints            *Endpoints         `json:"endpoints"`
	APIKeyEnv            string             `json:"api_key_env,omitempty"`
	InferenceDefaults    *InferenceDefaults `json:"inference_defaults"`
	EffectiveSwitches    []EffectiveSwitch  `json:"effective_switches"`
	MemoryEstimate       *MemoryEstimate    `json:"memory_estimate"`
	MemoryEstimateReason *string            `json:"memory_estimate_reason"`
	PiSnippet            *PiSnippet         `json:"pi_snippet"`
	Connection           *Connection        `json:"connection"`
	Requests             []RequestRow       `json:"requests"`
	PreparedModels       []PreparedModel    `json:"prepared_models"`
	ThinkMarkerBias      *ThinkMarkerBias   `json:"think_marker_bias"`
	LoopBreaker          *LoopBreaker       `json:"loop_breaker"`
	EntropyGate          *EntropyGate       `json:"think_marker_entropy_gate"`
	StateHistory         []StateEntry       `json:"state_history"`
}

// NPUPrefillRuntime is the CLI's live observation, distinct from the requested
// effective_settings.npu_prefill value. Absence means the CLI supplied no data.
// The canonical schema validates enums and required fields before decoding.
type NPUPrefillState string

const (
	NPUPrefillOn          NPUPrefillState = "on"
	NPUPrefillOff         NPUPrefillState = "off"
	NPUPrefillTripped     NPUPrefillState = "tripped"
	NPUPrefillFaulted     NPUPrefillState = "faulted"
	NPUPrefillUnavailable NPUPrefillState = "unavailable"
)

type NPUChipPowerState string

const (
	NPUChipPowerOK          NPUChipPowerState = "ok"
	NPUChipPowerUnavailable NPUChipPowerState = "unavailable"
	NPUChipPowerOff         NPUChipPowerState = "off"
)

type NPUPrefillRuntime struct {
	State         NPUPrefillState `json:"state"`
	Reason        *string         `json:"reason"`
	Trips         Count           `json:"trips"`
	HALFaults     Count           `json:"hal_faults"`
	WHEAEvents    Count           `json:"whea_events"`
	InvalidChunks Count           `json:"invalid_chunks"`
	ChipPower     NPUChipPower    `json:"chip_power"`
	Memory        *NPUMemory      `json:"memory,omitempty"`
	Scheme        *NPUPowerScheme `json:"scheme,omitempty"`
}

type NPUChipPower struct {
	State   NPUChipPowerState `json:"state"`
	WMean   *float64          `json:"w_mean"`
	WindowS float64           `json:"window_s"`
}

type NPUMemory struct {
	MinAvailableMiB float64 `json:"min_available_mib"`
	MaxCommitPct    float64 `json:"max_commit_pct"`
}

type NPUPowerScheme struct {
	GUID string `json:"guid"`
	Name string `json:"name"`
}

// ActiveModel is the CLI's selected identity; a custom served name is never
// used to infer a catalogue entry or a path.
type ActiveModel struct {
	CatalogueID  *string `json:"catalogue_id"`
	ModelPath    *string `json:"model_path"`
	ManifestPath *string `json:"manifest_path"`
}

// ClientSnippet preserves the CLI template and its key requirements verbatim.
type ClientSnippet struct {
	Kind          string  `json:"kind"`
	Title         string  `json:"title"`
	Template      string  `json:"template"`
	RequiresKey   bool    `json:"requires_key"`
	DefaultAPIKey *string `json:"default_api_key"`
}

// Endpoints are SDK base URLs emitted by the bound CLI listener. An absent
// object stays nil; neither Go nor the frontend appends endpoint paths.
type Endpoints struct {
	OpenAIBaseURL    string `json:"openai_base_url"`
	AnthropicBaseURL string `json:"anthropic_base_url"`
}

type Readiness struct {
	Runtime bool  `json:"runtime"`
	Model   bool  `json:"model"`
	Server  *bool `json:"server,omitempty"`
}

// Instance is one verified discovery descriptor (status instances[]). The flat
// identity keys duplicate the nested identity object; both are required on the
// wire and must agree (checked by ParseEvent).
type Instance struct {
	ActiveModel    *ActiveModel               `json:"active_model"`
	SchemaVersion  int                        `json:"schema_version"`
	ControlVersion int                        `json:"control_version"`
	ID             string                     `json:"instance_id"`
	PID            uint32                     `json:"pid"`
	CreationTime   string                     `json:"process_creation_time"`
	ExePath        string                     `json:"exe_path"`
	Identity       *Identity                  `json:"identity,omitempty"`
	Phase          string                     `json:"phase"`
	Endpoint       *string                    `json:"endpoint"`
	LogPath        *string                    `json:"log_path"`
	LastError      *CLIError                  `json:"last_error"`
	Ready          bool                       `json:"ready"`
	Effective      map[string]json.RawMessage `json:"effective_settings"`
	Model          string                     `json:"served_model"` // "" when the CLI sent null
}

// ServedModel returns the served model id, or "" when unavailable.
func (i Instance) ServedModel() string { return i.Model }

// Counters are the monotonic token and cache counters of one instance.
type Counters struct {
	PrefillTokens       *uint64  `json:"prefill_tokens_total"`
	DecodeTokens        *uint64  `json:"decode_tokens_total"`
	CachedTokens        *uint64  `json:"cached_tokens_total"`
	CacheDiskHits       *uint64  `json:"cache_disk_hits_total"`
	CacheMemoryHits     *uint64  `json:"cache_memory_hits_total"`
	CacheMisses         *uint64  `json:"cache_misses_total"`
	MaxSequenceTokens   *uint64  `json:"max_sequence_tokens"`
	DraftRounds         *uint64  `json:"draft_rounds_total"`
	DraftTokens         *uint64  `json:"draft_tokens_total"`
	DraftAcceptedTokens *uint64  `json:"draft_accepted_tokens_total"`
	DeviceLost          *uint64  `json:"device_lost_total"`
	PrefillSeconds      *float64 `json:"prefill_seconds_total"`
	DecodeSeconds       *float64 `json:"decode_seconds_total"`
	CompletedSuccessful *uint64  `json:"completed_successful_requests_total,omitempty"`
}
type LastCompletedRequest struct {
	PrefillTPS *float64 `json:"prefill_tokens_per_second"`
	DecodeTPS  *float64 `json:"decode_tokens_per_second"`
	AgeMS      *float64 `json:"age_ms"`
	RateUnit   string   `json:"rate_unit"`
	AgeUnit    string   `json:"age_unit"`
}

// Slot is a backend session state. State is idle, prefilling or generating.
type Slot struct {
	ID              uint64  `json:"id"`
	Processing      bool    `json:"processing"`
	State           string  `json:"state"`
	TokensUsed      *uint64 `json:"tokens_used"`
	Speculative     bool    `json:"speculative"`
	RequestID       *uint64 `json:"request_id"`
	ContextTokens   *uint64 `json:"context_tokens"`
	PromptTokens    *uint64 `json:"prompt_tokens"`
	CachedPrompt    *uint64 `json:"cached_prompt_tokens"`
	PrefilledTokens *uint64 `json:"prefilled_tokens"`
	GeneratedTokens *uint64 `json:"generated_tokens"`
	RemainingTokens *uint64 `json:"remaining_tokens"`
}
type LiveThroughput struct {
	Label      string   `json:"label"`
	Unit       string   `json:"unit"`
	IntervalMS *float64 `json:"interval_ms"`
	PrefillTPS *float64 `json:"prefill_tokens_per_second"`
	DecodeTPS  *float64 `json:"decode_tokens_per_second"`
}
type ModelSelection struct {
	Model    *string `json:"model"`
	Manifest *string `json:"manifest"`
	Source   string  `json:"source"`
	Pointer  *string `json:"pointer"`
}

// Status is the `bubo status --json` snapshot (status event payload).
// Machine is not a CLI v1 key and stays optional
// until the machine adapter ships its keys; an absent object means unavailable.
type Status struct {
	RuntimeDetail
	Prepared             bool                       `json:"prepared"`
	Phase                string                     `json:"phase"`
	Versions             Product                    `json:"versions"`
	Capabilities         Capabilities               `json:"capabilities"`
	Paths                map[string]json.RawMessage `json:"paths"`
	PreparedDefault      *string                    `json:"prepared_default"`
	Instances            []Instance                 `json:"instances"`
	Effective            map[string]json.RawMessage `json:"effective_settings"`
	Memory               map[string]json.RawMessage `json:"memory"` // raw CLI keys; MemoryUse() types them
	LastError            *CLIError                  `json:"last_error"`
	SampleTime           string                     `json:"sample_time"`
	Readiness            Readiness                  `json:"readiness"`
	ModelSelection       *ModelSelection            `json:"model_selection,omitempty"`
	Identity             *Identity                  `json:"identity"`
	Endpoint             *string                    `json:"endpoint"`
	LogPath              *string                    `json:"log_path"`
	ServedModel          *string                    `json:"served_model"`
	SelectionRequired    bool                       `json:"selection_required"`
	LastTransition       *string                    `json:"last_transition"`
	Counters             *Counters                  `json:"counters"`
	LastCompleted        *LastCompletedRequest      `json:"last_completed_request"`
	Sessions             *uint64                    `json:"sessions"`
	Queue                *uint64                    `json:"queue"`
	Slots                []Slot                     `json:"slots"`
	LiveThroughput       *LiveThroughput            `json:"live_throughput"`
	CacheHitRatio        *float64                   `json:"cache_hit_ratio,omitempty"`
	EnvironmentOverrides map[string]json.RawMessage `json:"environment_overrides,omitempty"`

	Machine map[string]json.RawMessage `json:"machine,omitempty"`
}

// Lifecycle is the payload of starting, loading, ready, warning, stopping and
// stopped. Identity is required. The ready event also carries the runtime
// detail objects. Stop control events add acknowledged and process_exited.
type Lifecycle struct {
	RuntimeDetail
	Identity       Identity                   `json:"identity"`
	Endpoint       *string                    `json:"endpoint"`
	Model          string                     `json:"served_model"`
	Effective      map[string]json.RawMessage `json:"effective_settings"`
	LogPath        *string                    `json:"log_path"`
	Message        string                     `json:"message"`
	Code           string                     `json:"code"`
	Remedy         string                     `json:"remedy"`
	Readiness      *Readiness                 `json:"readiness"`
	Acknowledged   *bool                      `json:"acknowledged"`
	ProcessExited  *bool                      `json:"process_exited"`
	AlreadyStopped *bool                      `json:"already_stopped"`
}

// Switch is one `serve --list-switches --json` registry row.
type Switch struct {
	Name            string             `json:"name"`
	Label           string             `json:"label,omitempty"`
	Kind            string             `json:"kind"`
	Class           string             `json:"class"`
	ModelFamily     string             `json:"model_family"`
	Platform        string             `json:"platform"`
	Requires        []string           `json:"requires"`
	Values          map[string]*string `json:"values"`
	ExactOff        bool               `json:"exact_off"`
	InjectWith      *string            `json:"inject_with"`
	SuppressedBy    []string           `json:"suppressed_by"`
	Purpose         string             `json:"purpose"`
	DefaultText     string             `json:"default_text"`
	HardwareScope   string             `json:"hardware_scope"`
	Note            string             `json:"note"`
	BuiltIn         *string            `json:"built_in"`
	ValueRule       string             `json:"value_rule"`
	Aliases         []string           `json:"aliases"`
	TakesValue      bool               `json:"takes_value"`
	Echo            bool               `json:"echo"`
	Secret          bool               `json:"secret,omitempty"` // absent on pre-B3 registries = false
	Group           string             `json:"group"`
	Audience        string             `json:"audience"`
	Experimental    bool               `json:"experimental"`
	RestartRequired bool               `json:"restart_required"`
	GUIControl      bool               `json:"gui_control"`
	Allowed         json.RawMessage    `json:"allowed"`
}
type Switches struct {
	SchemaVersion int      `json:"schema_version"`
	Switches      []Switch `json:"switches"`
}

// ByAudience returns the registry rows of one audience (user, advanced or
// debug) in registry order.
func (s *Switches) ByAudience(audience string) []Switch {
	var out []Switch
	if s != nil {
		for _, row := range s.Switches {
			if row.Audience == audience {
				out = append(out, row)
			}
		}
	}
	return out
}

// Event is the common envelope. InstanceID, PID, CreationTime and ExePath are
// the serve process identity (top level); control commands omit them and carry
// the identity only in the payload.
type Event struct {
	SchemaVersion int             `json:"schema_version"`
	Type          string          `json:"type"`
	OperationID   string          `json:"operation_id"`
	Seq           uint64          `json:"seq"`
	Time          string          `json:"time"`
	Product       Product         `json:"product"`
	Payload       json.RawMessage `json:"payload"`
	InstanceID    string          `json:"instance_id,omitempty"`
	PID           uint32          `json:"pid,omitempty"`
	CreationTime  string          `json:"process_creation_time,omitempty"`
	ExePath       string          `json:"exe_path,omitempty"`
	Value         any             `json:"-"`
}

// Identity of the lifecycle payload, when the value has one.
func (e Event) PayloadIdentity() (Identity, bool) {
	if l, ok := e.Value.(*Lifecycle); ok {
		return l.Identity, true
	}
	return Identity{}, false
}

func (i Identity) String() string {
	return fmt.Sprintf("%s pid %d", strings.ToLower(i.ID), i.PID)
}

func (i Instance) asIdentity() Identity {
	return Identity{ID: i.ID, PID: i.PID, CreationTime: i.CreationTime, ExePath: i.ExePath}
}

// MemoryUse types the status memory object. ok is false when memory is
// unavailable (null) or does not read as the contract object.
func (s *Status) MemoryUse() (MemoryUse, bool) {
	if s == nil || s.Memory == nil {
		return MemoryUse{}, false
	}
	var m MemoryUse
	// Decode the CLI-owned fields directly; polling needs no encode/decode copy.
	if raw, ok := s.Memory["rss_bytes"]; ok && json.Unmarshal(raw, &m.RSSBytes) != nil {
		return MemoryUse{}, false
	}
	if raw, ok := s.Memory["host_available_bytes"]; ok && json.Unmarshal(raw, &m.HostAvailableBytes) != nil {
		return MemoryUse{}, false
	}
	if raw, ok := s.Memory["evidence"]; ok && json.Unmarshal(raw, &m.Evidence) != nil {
		return MemoryUse{}, false
	}
	if raw, ok := s.Memory["unit"]; !ok || json.Unmarshal(raw, &m.Unit) != nil || m.Unit == "" {
		return MemoryUse{}, false
	}
	return m, true
}

// ApprovalLine is the prepare input record for approve or decline. It binds the
// answer to the operation and the plan digest the CLI sent.
func ApprovalLine(action, operationID, digest string) []byte {
	b, _ := json.Marshal(map[string]any{"action": action, "operation_id": operationID, "plan_digest": digest})
	return b
}

// CancelLine is the prepare input record that cancels the operation.
func CancelLine(operationID string) []byte {
	b, _ := json.Marshal(map[string]any{"action": "cancel", "operation_id": operationID})
	return b
}
