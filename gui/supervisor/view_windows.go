//go:build windows

package supervisor

import (
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:generate go run ./internal/viewgen

// Value preserves the CLI's open JSON values without untyped binding fields.
// Text is the exact number spelling for kind=number, or the string for
// kind=string. Null is explicit; properties and items preserve additive keys.
type Value struct {
	Kind       string           `json:"kind"`
	Text       string           `json:"text,omitempty"`
	Boolean    *bool            `json:"boolean,omitempty"`
	Items      []Value          `json:"items,omitempty"`
	Properties map[string]Value `json:"properties,omitempty"`
}

// Wails v2.16 mistakes a map's *string element for a struct named "string".
// Use the same typed value as allowed/requested/effective to retain nulls.
func valueFromNullableString(s *string) Value {
	if s == nil {
		return Value{Kind: "null"}
	}
	return Value{Kind: "string", Text: *s}
}

func valueFromJSON(raw json.RawMessage) Value {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	v := Value{Kind: "null"}
	if len(raw) == 0 || string(raw) == "null" {
		return v
	}
	switch raw[0] {
	case '"':
		v.Kind = "string"
		_ = json.Unmarshal(raw, &v.Text)
	case 't', 'f':
		v.Kind = "boolean"
		_ = json.Unmarshal(raw, &v.Boolean)
	case '[':
		v.Kind = "array"
		var rows []json.RawMessage
		_ = json.Unmarshal(raw, &rows)
		v.Items = make([]Value, len(rows))
		for i, r := range rows {
			v.Items[i] = valueFromJSON(r)
		}
	case '{':
		v.Kind = "object"
		var rows map[string]json.RawMessage
		_ = json.Unmarshal(raw, &rows)
		v.Properties = make(map[string]Value, len(rows))
		for k, r := range rows {
			v.Properties[k] = valueFromJSON(r)
		}
	default:
		v.Kind = "number"
		v.Text = string(raw)
	}
	return v
}

type ViewSettingRow struct {
	Name string `json:"name"`
	ViewSetting
}

// RuntimeView reports paths and readiness verbatim from status. No path is
// inferred from the executable or the GUI's runtime preference.
type RuntimeView struct {
	Paths     map[string]Value `json:"paths"`
	Kind      string           `json:"runtime_kind"`
	Readiness Readiness        `json:"readiness"`
}

type KeyView struct {
	LANAccess            bool             `json:"lan_access"`
	LANActive            bool             `json:"lan_active"`
	Present              bool             `json:"present"`
	Mask                 string           `json:"mask"`
	RestartNeeded        bool             `json:"restart_needed"`
	Unavailable          string           `json:"unavailable,omitempty"`
	CopyWithKey          bool             `json:"copy_with_key"`
	Addresses            []NetworkAddress `json:"addresses"`
	AddressesUnavailable string           `json:"addresses_unavailable,omitempty"`
}

type NetworkAddress struct {
	Adapter string `json:"adapter"`
	IPv4    string `json:"ipv4"`
	URL     string `json:"url"`
}

type WindowView struct {
	TrayAction    string `json:"tray_action"`
	TrayAvailable bool   `json:"tray_available"`
	Closing       bool   `json:"closing"`
}

type ViewFailure struct {
	Code      string           `json:"code"`
	Source    string           `json:"source"`
	Time      string           `json:"time"`
	Remedy    string           `json:"remedy"`
	Detail    string           `json:"detail,omitempty"`
	Memory    *MemoryShortfall `json:"memory,omitempty"`
	Again     bool             `json:"again,omitempty"`
	Forced    bool             `json:"forced"`
	Component string           `json:"component,omitempty"`
	Stage     string           `json:"stage,omitempty"`
	Switch    string           `json:"switch,omitempty"`
}
type ViewStall struct {
	Seconds       int    `json:"seconds"`
	ThresholdSecs int    `json:"threshold_seconds"`
	Since         string `json:"since"`
	Message       string `json:"message"`
}
type ViewGPU struct {
	SampleTime       string   `json:"sample_time,omitempty"`
	AgeSeconds       *float64 `json:"age_seconds,omitempty"`
	WindowSeconds    *float64 `json:"window_seconds,omitempty"`
	UsePct           *float64 `json:"use_pct,omitempty"`
	MemoryUsedBytes  *uint64  `json:"memory_used_bytes,omitempty"`
	MemoryTotalBytes *uint64  `json:"memory_total_bytes,omitempty"`
	SharedUsedBytes  *uint64  `json:"shared_used_bytes,omitempty"`
	TemperatureC     *float64 `json:"temperature_c,omitempty"`
	ChipPowerW       *float64 `json:"chip_power_w,omitempty"`
}

type UpdateState string

const (
	UpdateNone            UpdateState = "none"
	UpdateCurrent         UpdateState = "current"
	UpdateAvailable       UpdateState = "available"
	UpdateInProgress      UpdateState = "progress"
	UpdateReady           UpdateState = "ready"
	UpdateFailed          UpdateState = "failed"
	UpdateRollbackPending UpdateState = "rollback_pending"
)

type UpdateProgress struct {
	Phase      string  `json:"phase"`
	BytesDone  uint64  `json:"bytes_done"`
	BytesTotal *uint64 `json:"bytes_total"`
}

// Update is the frozen page contract. U4 supplies state and last-result
// wording; unavailable functionality carries a typed unsupported error.
type Update struct {
	State                   UpdateState `json:"state"`
	Ready                   bool        `json:"ready"`
	RuntimeFollowupRequired bool        `json:"runtime_followup_required"`
	LogPath                 string      `json:"log_path"`
	LogLines                []string    `json:"log_lines"`
	CurrentVersion          string      `json:"current_version"`
	AvailableVersion        string      `json:"available_version,omitempty"`
	Notes                   string      `json:"notes,omitempty"`
	SizeBytes               *uint64     `json:"size_bytes"`
	// Present only when a checked release or complete CLI plan supplies the
	// fact. Zero means that component needs no network transfer.
	RuntimeDownloadBytes *uint64         `json:"runtime_download_bytes,omitempty"`
	ModelsDownloadBytes  *uint64         `json:"models_download_bytes,omitempty"`
	Progress             *UpdateProgress `json:"progress"`
	Error                *CLIError       `json:"error"`
	LastResultNotice     string          `json:"last_result_notice,omitempty"`
	// CheckFailed is true when State is failed because the updater could not
	// start or a check failed. The user started no update then.
	CheckFailed bool `json:"check_failed"`
	// PreviousVersion is the version kept in the .previous folder while State
	// is rollback_pending. It is empty when Bubo cannot read it.
	PreviousVersion string `json:"previous_version,omitempty"`
}

// SavedSetupEstimate is the same cached CLI verdict used by lifecycle actions.
// Draft form estimates use the independent Estimate action.
type SavedSetupEstimate struct {
	MemoryEstimate *ViewMemoryEstimate `json:"memory_estimate"`
	Reason         *string             `json:"reason"`
	Error          *CLIError           `json:"error"`
}

type ViewSlot struct {
	ID              uint64   `json:"id"`
	Processing      bool     `json:"processing"`
	State           string   `json:"state"`
	StateLabel      string   `json:"state_label"`
	TokensUsed      *uint64  `json:"tokens_used"`
	Speculative     bool     `json:"speculative"`
	RequestID       *uint64  `json:"request_id"`
	ContextTokens   *uint64  `json:"context_tokens"`
	PromptTokens    *uint64  `json:"prompt_tokens"`
	CachedPrompt    *uint64  `json:"cached_prompt_tokens"`
	PrefilledTokens *uint64  `json:"prefilled_tokens"`
	GeneratedTokens *uint64  `json:"generated_tokens"`
	RemainingTokens *uint64  `json:"remaining_tokens"`
	PrefillTPS      *float64 `json:"prefill_tps"`
	DecodeTPS       *float64 `json:"decode_tps"`
	// ElapsedMS measures time since this supervisor first observed the request.
	ElapsedMS *float64 `json:"elapsed_ms"`
}

func projectSlot(s Slot) ViewSlot {
	label := slotLabels[s.State]
	if label == "" {
		label = "Status unavailable"
	}
	return ViewSlot{ID: s.ID, Processing: s.Processing, State: s.State, StateLabel: label, TokensUsed: s.TokensUsed, Speculative: s.Speculative, RequestID: s.RequestID, ContextTokens: s.ContextTokens, PromptTokens: s.PromptTokens, CachedPrompt: s.CachedPrompt, PrefilledTokens: s.PrefilledTokens, GeneratedTokens: s.GeneratedTokens, RemainingTokens: s.RemainingTokens}
}

// UserOverrides contains only user requests. Nil engine fields mean use the
// CLI default; effective values are exclusively in View.Settings.
type UserOverrides struct {
	Port               *int              `json:"port,omitempty"`
	Profile            *string           `json:"profile,omitempty"`
	Context            *uint64           `json:"context,omitempty"`
	Sessions           *uint32           `json:"sessions,omitempty"`
	RuntimePath        string            `json:"runtime_path,omitempty"`
	ModelPath          string            `json:"model_path,omitempty"`
	DataFolder         string            `json:"data_folder,omitempty"`
	ManifestPath       string            `json:"manifest_path,omitempty"`
	NPUPrefill         string            `json:"npu_prefill,omitempty"`
	Effort             string            `json:"effort,omitempty"`
	ModelFolders       []string          `json:"model_folders"`
	StartGUIAtLogin    bool              `json:"start_gui_at_login"`
	StartServerWithGUI bool              `json:"start_server_with_gui"`
	CloseToTray        bool              `json:"close_to_tray"`
	InferenceExpanded  bool              `json:"inference_expanded"`
	AllowTightSetup    bool              `json:"allow_tight_setup"`
	LANAccess          bool              `json:"lan_access"`
	StallSeconds       int               `json:"stall_seconds,omitempty"`
	CheckForUpdates    *bool             `json:"check_for_updates,omitempty"`
	SwitchOverrides    map[string]string `json:"switch_overrides,omitempty"`
}

// View is the sole input of the new frontend. Mutable collections are owned by
// the returned value; callers cannot mutate the supervisor or another reader.
type View struct {
	PreparationFile     *FilePosition         `json:"preparation_file,omitempty"`
	Actions             LifecycleActions      `json:"actions"`
	StatusLabel         string                `json:"status_label"`
	Serving             bool                  `json:"serving"`
	AutoRestarted       bool                  `json:"auto_restarted"`
	ServerWarnings      []ServerWarning       `json:"server_warnings"`
	OwnedServer         bool                  `json:"owned_server"`
	CurrentModel        *CurrentModel         `json:"current_model"`
	Fit                 FitVerdict            `json:"fit"`
	Source              string                `json:"_source,omitempty"`
	State               State                 `json:"state"`
	Reason              string                `json:"reason"`
	Remedy              string                `json:"remedy"`
	Initialized         bool                  `json:"initialized"`
	StartupPhase        StartupPhase          `json:"startup_phase"`
	Prepared            bool                  `json:"prepared"`
	Stale               bool                  `json:"stale"`
	Generation          uint64                `json:"generation"`
	Managed             bool                  `json:"managed"`
	ManagementSupported bool                  `json:"management_supported"`
	ManagementMessage   string                `json:"management_message"`
	Phase               ViewPhase             `json:"phase"`
	Instance            *ViewInstance         `json:"instance"`
	Discovered          []ViewInstance        `json:"discovered"`
	Failure             *ViewFailure          `json:"failure"`
	Settings            []ViewSettingRow      `json:"settings"`
	Switches            []ViewSwitch          `json:"switches"`
	SwitchOverrides     []SwitchOverrideView  `json:"switch_overrides"`
	EffectiveSwitches   []ViewEffectiveSwitch `json:"effective_switches"`
	Profiles            []Profile             `json:"profiles"`
	Product             *Product              `json:"product"`
	Capabilities        *Capabilities         `json:"capabilities"`
	MemoryEstimate      *ViewMemoryEstimate   `json:"memory_estimate"`
	SavedEstimate       SavedSetupEstimate    `json:"saved_estimate"`
	EstimateReason      *string               `json:"memory_estimate_reason"`
	ProcessMemory       *MemoryUse            `json:"process_memory"`
	Connection          *Connection           `json:"connection"`
	LogPath             *string               `json:"log_path,omitempty"`
	ActiveModel         *ActiveModel          `json:"active_model"`
	ClientSnippets      []ClientSnippet       `json:"client_snippets"`
	NPUPrefillRuntime   *NPUPrefillRuntime    `json:"npu_prefill_runtime"`
	Endpoints           *Endpoints            `json:"endpoints"`
	APIKeyEnv           string                `json:"api_key_env"`
	PiSnippet           string                `json:"pi_snippet"`
	Rates               *Rates                `json:"rates"`
	Sessions            *uint64               `json:"sessions"`
	Queue               *uint64               `json:"queue"`
	Slots               []ViewSlot            `json:"slots"`
	LiveSamples         []LiveSample          `json:"live_samples"`
	LiveSampleTime      string                `json:"live_sample_time"`
	Requests            []RequestRow          `json:"requests"`
	Counters            *Counters             `json:"counters"`
	LastCompleted       *LastCompletedRequest `json:"last_completed_request"`
	GPU                 *ViewGPU              `json:"gpu"`
	Models              *ViewModelsState      `json:"models"`
	PreparedModels      []PreparedModel       `json:"prepared_models"`
	PreparedDefault     *string               `json:"prepared_default"`
	Runtime             RuntimeView           `json:"runtime"`
	Saved               UserOverrides         `json:"saved"`
	DownloadFolder      DownloadFolderView    `json:"download_folder"`
	Key                 KeyView               `json:"key"`
	Window              WindowView            `json:"window"`
	Stall               *ViewStall            `json:"stall"`
	Progress            *Progress             `json:"progress"`
	Approval            *Approval             `json:"approval"`
	Preparation         []ModelPlan           `json:"preparation"`
	PreparationPending  bool                  `json:"preparation_pending"`
	PlanError           *CLIError             `json:"plan_error"`
	Pending             []PendingChange       `json:"restart_needed"`
	Inference           *InferenceDefaults    `json:"inference_defaults"`
	ThinkBias           *ViewThinkMarkerBias  `json:"think_marker_bias"`
	LoopBreaker         *ViewLoopBreaker      `json:"loop_breaker"`
	EntropyGate         *EntropyGate          `json:"think_marker_entropy_gate"`
	Logs                []string              `json:"logs"`
	Notice              *Notice               `json:"notice"`
	Warnings            []string              `json:"warnings"`
	Diagnostic          string                `json:"diagnostic"`
	BundlePreview       string                `json:"bundle_preview"`
	Update              Update                `json:"update"`
	CheckForUpdates     bool                  `json:"check_for_updates"`
	StatusFeed          StatusFeed            `json:"status_feed"`
}

type Notice struct {
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

type ViewInput struct {
	// App-owned request identity, independent of an occupied-memory sample.
	RunningSetupUnchanged     bool
	SavedEstimate             *MemoryEstimate
	SavedEstimateError        *CLIError
	Snapshot                  Snapshot
	Registry                  *Switches
	Models                    *ModelsState
	Saved                     Settings
	Key                       KeyView
	Window                    WindowView
	Logs                      []string
	Notice                    *Notice
	Diagnostic, BundlePreview string
	Update                    Update
	Source                    string
}

// ViewBuilder serializes the entire projection under one lock. Inputs must be
// owned snapshots (Supervisor.Snapshot and ModelService.State provide them).
// The generated explicit copy makes the result independent of every input.
type ViewBuilder struct{ mu sync.Mutex }

func (b *ViewBuilder) Build(in ViewInput) View {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := in.Snapshot
	in.Saved.AllowTightSetup = tightConsentActive(in.Saved, s.Status, in.Registry)
	actions := lifecycleActions(in)
	v := View{Source: in.Source, State: s.State, Reason: s.Label, Initialized: s.Initialized, Prepared: s.Prepared, Stale: s.Stale, Generation: s.Generation, Rates: s.Rates, Progress: s.Progress, Approval: s.Approval, Key: in.Key, Logs: in.Logs, Notice: in.Notice, Diagnostic: in.Diagnostic, BundlePreview: in.BundlePreview, Update: in.Update}
	v.Actions = actions
	v.LiveSamples = visibleLiveSamples(s.LiveSamples, time.Now())
	v.LiveSampleTime = s.LiveSampleTime
	v.DownloadFolder = downloadFolderView(in)
	v.Phase = nativePhase(s, in.Models, in.Saved)
	v.StatusLabel = phaseLabels[v.Phase]
	if v.Phase == "checking" && s.StartupWaiting {
		v.StatusLabel = FirstStartLabel
	}
	v.Serving = s.State == Ready || s.State == AttachedExternal
	v.AutoRestarted = s.AutoRestarted
	v.ServerWarnings = s.ServerWarnings
	v.OwnedServer = s.State == Ready
	v.CurrentModel = currentModel(s, in.Models, in.Saved)
	f := in.SavedEstimate
	if f == nil && statusEstimateMatches(in.Saved, s.Status) {
		f = s.Status.MemoryEstimate
	}
	if v.OwnedServer && (in.RunningSetupUnchanged || statusEstimateMatches(in.Saved, s.Status)) {
		v.Fit = FitVerdict{Class: "in_use", Sentence: "In use"}
	} else {
		v.Fit = buildFit(f)
		if f == nil && in.SavedEstimateError != nil && in.SavedEstimateError.Code == "estimate_no_verdict" {
			// The CLI answered and has no verdict for this setup: not "in flight".
			v.Fit = noVerdictFit()
		}
	}
	v.StartupPhase = s.StartupPhase
	if in.SavedEstimate != nil {
		estimate := projectMemoryEstimate(*in.SavedEstimate)
		v.SavedEstimate.MemoryEstimate = &estimate
	} else if in.SavedEstimateError == nil && statusEstimateMatches(in.Saved, s.Status) {
		estimate := projectMemoryEstimate(*s.Status.MemoryEstimate)
		v.SavedEstimate.MemoryEstimate = &estimate
	}
	if in.SavedEstimateError != nil {
		reason := in.SavedEstimateError.Remedy
		v.SavedEstimate.Reason, v.SavedEstimate.Error = &reason, in.SavedEstimateError
	}
	if v.Update.State == "" {
		v.Update.State = UpdateNone
	}
	v.CheckForUpdates = in.Saved.UpdatesEnabled()
	v.StatusFeed = statusFreshness(s, time.Now())
	if v.StatusFeed.State == StatusNotUpdating || v.StatusFeed.State == StatusPaused {
		v.Stale = true
	}
	// Startup has no established running feed to become stale. Preserve the
	// observation diagnostics in Snapshot; the page should ask the user to wait.
	if s.State == Starting {
		v.StatusFeed.State = StatusWaiting
	}
	v.Stale = v.Serving && v.Stale
	v.Managed = s.Managed
	v.Window = in.Window
	if s.SettingsWarning != "" {
		v.Warnings = append(v.Warnings, s.SettingsWarning)
	}
	if s.LogWarning != "" {
		v.Warnings = append(v.Warnings, s.LogWarning)
	}
	v.ManagementSupported = s.ManagementSupported
	v.ManagementMessage = s.ManagementMessage
	components := map[string]bool{}
	for _, p := range s.PlanParts {
		components[p.Component] = true
	}
	var ordered []string
	for c := range components {
		ordered = append(ordered, c)
	}
	sort.Strings(ordered)
	for _, c := range ordered {
		p, err := assemblePlan(s.PlanParts, c)
		if err != nil {
			var total uint64
			var count uint64
			for _, part := range s.PlanParts {
				if part.Component == c {
					total = part.Parts
					count++
				}
			}
			if s.Approval == nil && count < total {
				v.PreparationPending = true
				continue
			}
			v.PlanError = &CLIError{Code: "protocol_mismatch", Component: c, Stage: "plan", Remedy: err.Error()}
			break
		}
		p.Kind = c
		v.Preparation = append(v.Preparation, p)
		// A refresh can need approval while reusing every payload file. Read
		// the complete CLI file states, never infer reuse from Ready alone.
		version := v.Update.AvailableVersion
		followup := v.Update.State == UpdateCurrent && v.Update.RuntimeFollowupRequired
		if followup {
			version = v.Update.CurrentVersion
		}
		planForRelease := version == "" || s.Status != nil && s.Status.Versions.Version == version
		if planForRelease && (v.Update.State == UpdateAvailable || v.Update.State == UpdateInProgress || v.Update.State == UpdateReady || followup) {
			var download *uint64
			if len(p.Files) > 0 {
				reused := true
				for _, file := range p.Files {
					reused = reused && file.State == "reuse"
				}
				if reused {
					zero := uint64(0)
					download = &zero
				}
			}
			switch c {
			case "runtime":
				v.Update.RuntimeDownloadBytes = download
			case "model":
				v.Update.ModelsDownloadBytes = download
			}
		}
	}
	if s.LastError != nil {
		v.Remedy = s.LastError.Remedy
		f := s.LastError
		v.Failure = &ViewFailure{Code: f.Code, Source: f.Source, Time: f.Time.UTC().Format(time.RFC3339Nano), Remedy: f.Remedy, Detail: f.Detail, Memory: f.Memory, Again: f.Again, Forced: f.Forced, Component: f.Component, Stage: f.Stage, Switch: f.Switch}
	}
	if s.Stall != nil {
		w := s.Stall
		v.Stall = &ViewStall{Seconds: w.Seconds, ThresholdSecs: w.ThresholdSecs, Since: w.Since.UTC().Format(time.RFC3339Nano), Message: w.Message}
	}
	if s.GPU != nil {
		g := s.GPU
		v.GPU = &ViewGPU{UsePct: g.UsePct, MemoryUsedBytes: g.MemoryUsedBytes, MemoryTotalBytes: g.MemoryTotalBytes, SharedUsedBytes: g.SharedUsedBytes, TemperatureC: g.TemperatureC, ChipPowerW: g.ChipPowerW}
		v.GPU.SampleTime, v.GPU.WindowSeconds = g.SampleTime, g.WindowSeconds
		if sampled, err := time.Parse(time.RFC3339Nano, g.SampleTime); err == nil {
			age := max(0, time.Since(sampled).Seconds())
			v.GPU.AgeSeconds = &age
		}
	}
	if s.Instance != nil {
		i := projectInstance(*s.Instance)
		i.Effective = projectSettings(RedactSettings(s.Instance.Effective, in.Registry))
		v.Instance = &i
	}
	for _, i := range s.Discovered {
		i.Effective = RedactSettings(i.Effective, in.Registry)
		v.Discovered = append(v.Discovered, projectInstance(i))
	}
	if in.Registry != nil {
		for _, sw := range in.Registry.Switches {
			v.Switches = append(v.Switches, projectSwitch(sw))
		}
	}
	if in.Models != nil {
		m := projectModelsState(*in.Models)
		v.Models = &m
		if v.Models.Download != nil && in.Models.Download != nil && in.Models.Download.Plan != nil {
			v.Models.Download.FilePosition = catalogueFilePosition(in.Models.Download.FileID, in.Models.Download.Plan, in.Models.Catalogue)
		}
	}
	x := in.Saved
	v.Saved = UserOverrides{RuntimePath: x.RuntimePath, ModelPath: x.ModelPath, DataFolder: x.DataFolder, ManifestPath: x.ManifestPath, NPUPrefill: x.NPUPrefill, Effort: x.Effort, ModelFolders: x.ModelFolders, StartGUIAtLogin: x.StartGUIAtLogin, StartServerWithGUI: x.StartServerWithGUI, CloseToTray: x.CloseToTray, AllowTightSetup: x.AllowTightSetup, LANAccess: x.LanAccess, StallSeconds: x.StallSeconds}
	v.Saved.CheckForUpdates = x.CheckForUpdates
	v.Saved.InferenceExpanded = x.InferenceExpanded
	v.SwitchOverrides = switchOverrideViews(in)
	for _, row := range v.SwitchOverrides {
		if row.Saved != nil && row.WriteVia == "override" {
			if v.Saved.SwitchOverrides == nil {
				v.Saved.SwitchOverrides = map[string]string{}
			}
			v.Saved.SwitchOverrides[row.Name] = *row.Saved
		}
	}
	if x.Port != 0 {
		v.Saved.Port = &x.Port
	}
	if x.Profile != "" {
		v.Saved.Profile = &x.Profile
	}
	if x.Context != 0 {
		v.Saved.Context = &x.Context
	}
	if x.Sessions != 0 {
		v.Saved.Sessions = &x.Sessions
	}
	if st := s.Status; st != nil {
		if s.LastError == nil && st.LastError != nil {
			v.Reason = st.LastError.Code
			v.Remedy = st.LastError.Remedy
		}
		v.Profiles = st.Capabilities.Profiles
		v.Product = &st.Versions
		if v.Update.CurrentVersion == "" {
			v.Update.CurrentVersion = st.Versions.Version
		}
		v.Capabilities = &st.Capabilities
		v.Runtime = RuntimeView{Paths: projectSettings(st.Paths), Readiness: st.Readiness}
		if raw, ok := st.Paths["runtime_kind"]; ok {
			_ = json.Unmarshal(raw, &v.Runtime.Kind)
		}
		v.Settings = settingRows(st.Effective, in.Registry)
		for _, sw := range st.EffectiveSwitches {
			if hiddenRegistryValue(in.Registry, sw.Name) {
				sw.Effective = hiddenValue(sw.Effective)
			}
			v.EffectiveSwitches = append(v.EffectiveSwitches, projectEffectiveSwitch(sw))
		}
		v.Pending = pendingChanges(in)
		if st.MemoryEstimate != nil {
			m := projectMemoryEstimate(*st.MemoryEstimate)
			v.MemoryEstimate = &m
		}
		v.EstimateReason = st.MemoryEstimateReason
		if m, ok := st.MemoryUse(); ok {
			v.ProcessMemory = &m
		}
		v.Connection = st.Connection
		v.LogPath = st.LogPath
		v.ActiveModel = st.ActiveModel
		v.ClientSnippets = st.ClientSnippets
		v.NPUPrefillRuntime = st.NPUPrefillRuntime
		v.Endpoints = st.Endpoints
		v.APIKeyEnv = st.APIKeyEnv
		v.PiSnippet = PiSnippetText(st.PiSnippet)
		v.Sessions = st.Sessions
		v.Queue = st.Queue
		for _, slot := range st.Slots {
			projected := projectSlot(slot)
			if live, ok := s.LiveSlots[slot.ID]; ok && slot.RequestID != nil && *slot.RequestID == live.RequestID {
				projected.PrefillTPS, projected.DecodeTPS, projected.ElapsedMS = live.PrefillTPS, live.DecodeTPS, live.ElapsedMS
			}
			v.Slots = append(v.Slots, projected)
		}
		v.Requests = st.Requests
		v.Counters = st.Counters
		v.LastCompleted = st.LastCompleted
		v.PreparedModels = st.PreparedModels
		v.PreparedDefault = st.PreparedDefault
		v.Inference = st.InferenceDefaults
		v.EntropyGate = st.EntropyGate
		if st.ThinkMarkerBias != nil {
			a := projectThinkMarkerBias(*st.ThinkMarkerBias)
			v.ThinkBias = &a
		}
		if st.LoopBreaker != nil {
			a := projectLoopBreaker(*st.LoopBreaker)
			v.LoopBreaker = &a
		}
	}
	if s.Progress != nil {
		var catalogue []CatalogueEntry
		if in.Models != nil {
			catalogue = in.Models.Catalogue
		}
		for _, plan := range v.Preparation {
			if plan.Kind == s.Progress.Component {
				v.PreparationFile = catalogueFilePosition(s.Progress.FileID, &plan, catalogue)
				if v.PreparationFile == nil && plan.Kind == "runtime" {
					for i, file := range plan.Files {
						if file.FileID == s.Progress.FileID {
							v.PreparationFile = &FilePosition{Index: i + 1, Total: len(plan.Files)}
						}
					}
				}
			}
		}
	}
	// A retained status is not proof of a serving instance after preparation,
	// failure or stop. Only running lifecycle states advertise addresses.
	if s.State != Ready && s.State != AttachedExternal {
		v.Endpoints = nil
		v.Connection = nil
	}
	out := copyView(v)
	fillNilLists(reflect.ValueOf(&out).Elem())
	return out
}

func projectSettings(in map[string]json.RawMessage) map[string]Value {
	if in == nil {
		return nil
	}
	out := make(map[string]Value, len(in))
	for k, v := range in {
		out[k] = valueFromJSON(v)
	}
	return out
}

func settingRows(in map[string]json.RawMessage, registry *Switches) []ViewSettingRow {
	redacted := RedactSettings(in, registry)
	var keys []string
	for k := range redacted {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []ViewSettingRow
	for _, k := range keys {
		if row, ok := SettingRow(redacted, k); ok {
			out = append(out, ViewSettingRow{Name: k, ViewSetting: projectSetting(row)})
		}
	}
	return out
}
