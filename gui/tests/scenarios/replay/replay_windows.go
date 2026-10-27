//go:build windows

// Package replay shares scenario recordings between the fake CLI and the View
// generator. Only fixture scenarios apply synthetic running-server state.
package replay

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"bubo/gui/supervisor"
)

type Manifest struct {
	Source             string               `json:"_source"`
	State              supervisor.State     `json:"state"`
	Records            map[string]string    `json:"records"`
	Busy               bool                 `json:"busy,omitempty"`
	Stall              bool                 `json:"stall,omitempty"`
	LAN                bool                 `json:"lan,omitempty"`
	Download           bool                 `json:"download,omitempty"`
	Update             *supervisor.Update   `json:"update,omitempty"`
	// Package facts for an arranged upgrade from the published 0.1.0 build.
	InstalledProduct   *supervisor.Product  `json:"installed_product,omitempty"`
	SavedOverrides     map[string]string    `json:"saved_overrides,omitempty"`
	OverrideFailure    *supervisor.CLIError `json:"override_failure,omitempty"`
	SavedEstimateError *supervisor.CLIError `json:"saved_estimate_error,omitempty"`
	LogRecord          string               `json:"log_record,omitempty"`
	FirstStart         bool                 `json:"first_start,omitempty"`
	CLIMissing         bool                 `json:"cli_missing,omitempty"`
	// AutoRestarted: the supervisor restarted the server after exit status 75.
	AutoRestarted bool `json:"auto_restarted,omitempty"`
	// Failure: the supervisor's own failure of the last run (no CLI record).
	Failure *supervisor.CLIError `json:"failure,omitempty"`
	// SavedModelID: the id of a prepared package of the models record that the
	// user selected (by its model file). The row is the recorded one.
	SavedModelID string `json:"saved_model_id,omitempty"`

	// NoRowFits (hand-written, a finding): no recording has a model that fits at
	// no context size. This overlay clears the fits flag of every context row of
	// the saved estimate, which the CLI does for a model larger than the machine.
	NoRowFits bool `json:"no_row_fits,omitempty"`
	// FailureAgain: the failure (device_lost) came after the app's own restart.
	FailureAgain bool `json:"failure_again,omitempty"`
}

func Read(folder string) (Manifest, map[string][]supervisor.Event, error) {
	var m Manifest
	b, err := os.ReadFile(filepath.Join(folder, "scenario.json"))
	if err != nil {
		return m, nil, err
	}
	if err = json.Unmarshal(b, &m); err != nil {
		return m, nil, err
	}
	if m.Source != "recorded" && m.Source != "fixture" {
		return m, nil, fmt.Errorf("invalid scenario provenance %q", m.Source)
	}
	if m.Source == "recorded" && (m.Busy || m.Stall || m.LAN || m.Download || m.Update != nil || m.AutoRestarted || m.Failure != nil || m.FailureAgain || len(m.SavedOverrides) != 0 || m.OverrideFailure != nil || m.SavedEstimateError != nil || m.FirstStart || m.CLIMissing || m.SavedModelID != "" || m.NoRowFits || m.Records["estimate"] != "" || m.Records["saved_estimate"] != "" || m.Records["npu_runtime"] != "") {
		return m, nil, fmt.Errorf("recorded scenario cannot contain fixture overlays")
	}
	all := map[string][]supervisor.Event{}
	for kind, path := range m.Records {
		if filepath.Ext(path) == ".json" {
			b, err := os.ReadFile(filepath.Join(folder, path))
			if err != nil {
				return m, nil, err
			}
			e, err := supervisor.ParseEvent(b)
			if err != nil {
				return m, nil, fmt.Errorf("%s: %w", path, err)
			}
			all[kind] = []supervisor.Event{e}
			continue
		}
		f, err := os.Open(filepath.Join(folder, path))
		if err != nil {
			return m, nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 4096), 1<<20)
		for sc.Scan() {
			if len(sc.Bytes()) == 0 {
				continue
			}
			e, err := supervisor.ParseEvent(sc.Bytes())
			if err != nil {
				f.Close()
				return m, nil, fmt.Errorf("%s: %w", path, err)
			}
			all[kind] = append(all[kind], e)
		}
		err = sc.Err()
		f.Close()
		if err != nil {
			return m, nil, err
		}
	}
	return m, all, nil
}

func Inputs(folder string) (supervisor.ViewInput, error) {
	m, records, err := Read(folder)
	if err != nil {
		return supervisor.ViewInput{}, err
	}
	// A recording has no updater attached. Only an explicit app-state overlay
	// may represent a completed update check or another updater transition.
	in := supervisor.ViewInput{Source: m.Source, Snapshot: supervisor.Snapshot{State: m.State, Initialized: !m.FirstStart}, Saved: supervisor.Settings{Version: 1}, Update: supervisor.Update{State: supervisor.UpdateNone, Ready: true}}
	if m.FirstStart {
		if len(m.Records) != 0 || m.State != supervisor.NotPrepared {
			return in, fmt.Errorf("first-start must have no CLI records and a not-prepared state")
		}
		in.Snapshot.StartupPhase = supervisor.StartingCLI
		return in, nil
	}
	if m.CLIMissing {
		// The supervisor's own failure when bin\bubo.exe is absent: no CLI
		// record can exist, as in the real exe's S2.
		if len(m.Records) != 0 || m.State != supervisor.Failed {
			return in, fmt.Errorf("cli-missing must have no CLI records and a failed state")
		}
		in.Snapshot.LastError = &supervisor.Failure{Code: "cli_missing", Source: "supervisor"}
		return in, nil
	}
	in.Saved.SwitchOverrides = m.SavedOverrides
	for kind, events := range records {
		for _, e := range events {
			switch v := e.Value.(type) {
			case *supervisor.Status:
				if kind == "status" {
					in.Snapshot.Status = v
					in.Snapshot.Prepared = v.Prepared
				}
			case *supervisor.Switches:
				in.Registry = v
			case *supervisor.Models:
				in.Models = &supervisor.ModelsState{Status: supervisor.ModelsOK, Listing: &v.Listing, Catalogue: v.Catalogue, PreparedModels: v.PreparedModels, Scanned: v.Scanned, ScanWarnings: v.ScanWarnings}
			case *supervisor.Progress:
				in.Snapshot.Progress = v
			case *supervisor.Plan:
				in.Snapshot.PlanParts = append(in.Snapshot.PlanParts, *v)
			case *supervisor.CLIError:
				in.Snapshot.LastError = &supervisor.Failure{Code: v.Code, Source: "cli", Time: parseTime(e.Time), Remedy: v.Remedy, Component: v.Component, Stage: v.Stage, Switch: v.Switch}
			}
		}
	}
	st := in.Snapshot.Status
	if st == nil {
		return in, fmt.Errorf("scenario has no status recording")
	}
	if m.InstalledProduct != nil {
		st.Versions = *m.InstalledProduct
	}
	in.Snapshot.Discovered = st.Instances
	// The running flow adds the whole-job number to each progress event from the
	// plan parts (E8, M3); a preparing fixture carries it the same way.
	if p := in.Snapshot.Progress; p != nil && p.Overall == nil && m.Source == "fixture" && m.State == supervisor.Preparing {
		p.Overall = supervisor.OverallProgress(in.Snapshot.PlanParts, p)
	}
	// While an update waits for its rollback, the running build is the new one
	// (updater helper afterFailedHealth): "This build" is the update's version.
	if m.Update != nil && m.Update.State == supervisor.UpdateRollbackPending && m.Update.AvailableVersion != "" {
		st.Versions.Version = m.Update.AvailableVersion
	}
	if (m.State == supervisor.Ready || m.State == supervisor.AttachedExternal) && len(st.Instances) > 0 {
		in.Snapshot.Instance = &st.Instances[0]
		in.Snapshot.Managed = m.State == supervisor.Ready
	}
	if m.Update != nil {
		in.Update = *m.Update
	}
	in.SavedEstimateError = m.SavedEstimateError
	if id := m.SavedModelID; id != "" {
		var row *supervisor.PreparedModel
		for _, e := range records["models"] {
			if list, ok := e.Value.(*supervisor.Models); ok {
				for i := range list.PreparedModels {
					if list.PreparedModels[i].ID == id {
						row = &list.PreparedModels[i]
					}
				}
			}
		}
		if row == nil || row.ModelPath == nil {
			return in, fmt.Errorf("saved model %s is not a prepared package of the models record", id)
		}
		st.PreparedModels = append(st.PreparedModels, *row)
		in.Saved.ModelPath = *row.ModelPath
	}

	if m.LogRecord != "" {
		b, err := os.ReadFile(filepath.Join(folder, m.LogRecord))
		if err != nil {
			return in, err
		}
		in.Logs = strings.Split(strings.TrimSpace(string(b)), "\n")
		for i := range in.Logs {
			in.Logs[i] = strings.TrimSpace(in.Logs[i])
		}
	}
	in.Snapshot.AutoRestarted = m.AutoRestarted
	if f := m.Failure; f != nil {
		in.Snapshot.LastError = &supervisor.Failure{Code: f.Code, Source: "supervisor", Remedy: f.Remedy, Again: m.FailureAgain, Time: parseTime(records["status"][0].Time)}
	}
	if f := m.OverrideFailure; f != nil {
		// A rejected start retains the exact override request it attempted.
		in.Snapshot.RunningSwitchOverrides = m.SavedOverrides
		in.Snapshot.LastError = &supervisor.Failure{Code: f.Code, Source: "cli", Component: f.Component, Stage: f.Stage, Remedy: f.Remedy, Switch: f.Switch, Time: parseTime(records["status"][0].Time)}
	}
	if events := records["npu_runtime"]; len(events) != 0 {
		if len(events) != 1 {
			return in, fmt.Errorf("NPU fixture requires one status record")
		}
		npu, ok := events[0].Value.(*supervisor.Status)
		if !ok || npu.NPUPrefillRuntime == nil {
			return in, fmt.Errorf("NPU fixture has no typed runtime observation")
		}
		st.NPUPrefillRuntime = npu.NPUPrefillRuntime
	}
	if events := records["estimate"]; len(events) == 1 {
		estimate, ok := events[0].Value.(*supervisor.ModelEstimate)
		if !ok || estimate.MemoryEstimate == nil {
			return in, fmt.Errorf("fixture has no typed model estimate")
		}
		st.MemoryEstimate = estimate.MemoryEstimate
		st.MemoryEstimateReason = estimate.Reason
	}
	if events := records["saved_estimate"]; len(events) != 0 {
		if len(events) != 1 {
			return in, fmt.Errorf("saved-estimate fixture requires one model estimate")
		}
		estimate, ok := events[0].Value.(*supervisor.ModelEstimate)
		if !ok || estimate.MemoryEstimate == nil || st.PreparedDefault == nil {
			return in, fmt.Errorf("saved-estimate fixture lacks a typed prepared-model estimate")
		}
		matching := false
		for _, prepared := range st.PreparedModels {
			if prepared.ManifestPath == nil || !supervisor.SameFolder(*prepared.ManifestPath, *st.PreparedDefault) {
				continue
			}
			// The estimate names the package by its catalogue id or by its folder.
			if (estimate.SubjectKind == "catalogue" && prepared.ID == estimate.Subject) || (estimate.SubjectKind == "path" && supervisor.SameFolder(estimate.Subject, filepath.Dir(*prepared.ManifestPath))) {
				matching = true
			}
		}
		if !matching {
			return in, fmt.Errorf("saved-estimate fixture does not match the CLI prepared default")
		}
		// This is the recorded request's context/session pair. Keep its
		// verdict separate from the authentic no-server status allocation.
		in.Saved.Context = uint64(estimate.MemoryEstimate.ContextTokens)
		in.Saved.Sessions = uint32(estimate.MemoryEstimate.Sessions)
		in.SavedEstimate = estimate.MemoryEstimate
		if m.NoRowFits && estimate.MemoryEstimate.RecommendedContext != nil {
			for i := range estimate.MemoryEstimate.RecommendedContext.BySessions {
				row := &estimate.MemoryEstimate.RecommendedContext.BySessions[i]
				row.Fits = false
				row.Recommended = ptr(false)
			}
		}
	}
	if m.Source == "fixture" {
		in.Snapshot.Prepared = m.State != supervisor.NotPrepared && m.State != supervisor.Preparing
		st.Prepared = in.Snapshot.Prepared
		switch m.State {
		case supervisor.Ready, supervisor.AttachedExternal:
			if len(st.Instances) == 0 {
				return in, fmt.Errorf("running fixture has no instance")
			}
			in.Snapshot.Instance = &st.Instances[0]
		case supervisor.Stopped:
			st.Phase = "stopped"
			st.Instances = []supervisor.Instance{}
			st.Identity = nil
			st.Readiness.Server = nil
		case supervisor.Starting:
			st.Phase = "starting"
			st.Readiness.Server = ptr(false)
		case supervisor.Preparing:
			st.Phase = "not_prepared"
			// Fixture-only active transfer: the base ready status predates this
			// preparation. Do not claim the component being transferred is ready.
			if p := in.Snapshot.Progress; p != nil {
				switch p.Component {
				case "model":
					st.Readiness.Runtime, st.Readiness.Model = true, false
				case "runtime":
					st.Readiness.Runtime = false
				}
			}
		case supervisor.Failed:
			st.Phase = "error"
			st.Instances = []supervisor.Instance{}
			st.Identity = nil
		}
	}
	if m.Busy {
		if len(st.Slots) == 0 {
			return in, fmt.Errorf("busy fixture has no source slot")
		}
		a := st.Slots[0]
		a.State = "generating"
		a.Processing = true
		b := a
		b.ID = a.ID + 1
		st.Slots = []supervisor.Slot{a, b}
		st.Sessions = ptr(uint64(2))
	}
	if st.LiveThroughput != nil {
		r := st.LiveThroughput
		in.Snapshot.Rates = &supervisor.Rates{PrefillTPS: r.PrefillTPS, DecodeTPS: r.DecodeTPS, IntervalMS: r.IntervalMS, Source: "cli", Label: r.Label, Unit: r.Unit}
	}
	if m.Stall {
		in.Snapshot.Stall = &supervisor.StallWarning{Seconds: int(supervisor.DefaultStallTimeout / time.Second), ThresholdSecs: int(supervisor.DefaultStallTimeout / time.Second), Since: parseTime(records["status"][0].Time), Message: "No token progress in fixture history"}
	}
	if m.LAN {
		in.Key = supervisor.KeyView{LANAccess: true, Present: true, Mask: supervisor.KeyMask}
		in.Saved.LanAccess = true
	}
	if m.Download {
		if in.Models == nil || in.Snapshot.Progress == nil {
			return in, fmt.Errorf("download fixture lacks model/progress records")
		}
		p := in.Snapshot.Progress
		in.Models.Download = &supervisor.ModelJob{Kind: "download", State: "running", Phase: p.Phase, FileID: p.FileID, BytesDone: p.Done, BytesTotal: p.Total}
		if len(in.Models.Catalogue) > 0 {
			in.Models.Download.ID = in.Models.Catalogue[0].ID
		}
	}
	return in, nil
}

func Build(folder string) (supervisor.View, error) {
	in, err := Inputs(folder)
	if err != nil {
		return supervisor.View{}, err
	}
	var builder supervisor.ViewBuilder
	return builder.Build(in), nil
}

// Command returns the same CLI records used by the builder, including the
// explicitly marked fixture modifications. It never starts a server.
func Command(folder, kind string) ([]supervisor.Event, error) {
	_, records, err := Read(folder)
	if err != nil {
		return nil, err
	}
	events := records[kind]
	if len(events) == 0 {
		return nil, fmt.Errorf("scenario has no %s recording", kind)
	}
	if kind == "status" {
		in, err := Inputs(folder)
		if err != nil {
			return nil, err
		}
		for i := range events {
			if events[i].Type == "status" {
				events[i].Payload, err = json.Marshal(in.Snapshot.Status)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	return events, nil
}
func ptr[T bool | uint64](v T) *T  { return &v }
func parseTime(s string) time.Time { t, _ := time.Parse(time.RFC3339, s); return t }
