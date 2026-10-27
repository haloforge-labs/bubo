//go:build windows

package main

import (
	"errors"
	"strings"
	"testing"

	"bubo/gui/supervisor"
)

type bindingBackend struct {
	snapshot supervisor.Snapshot
	saved    supervisor.Settings
	actions  []string
}

func (b *bindingBackend) Snapshot() supervisor.Snapshot    { return b.snapshot }
func (*bindingBackend) Logs() []string                     { return nil }
func (*bindingBackend) Notice() string                     { return "" }
func (*bindingBackend) Diagnostic() string                 { return "" }
func (*bindingBackend) BundlePreview() string              { return "" }
func (*bindingBackend) SetVisible(bool)                    {}
func (b *bindingBackend) Save(s supervisor.Settings) error { b.saved = s; return nil }
func (b *bindingBackend) Action(action string, _ supervisor.Settings, _ string) <-chan supervisor.ResultReply {
	b.actions = append(b.actions, action)
	result := make(chan supervisor.ResultReply, 1)
	result <- supervisor.ResultReply{}
	return result
}

func TestBindingsDelegateToSupervisorService(t *testing.T) {
	fit := true
	b := &bindingBackend{snapshot: supervisor.Snapshot{Initialized: true, Prepared: true, State: supervisor.Stopped,
		Status: &supervisor.Status{Readiness: supervisor.Readiness{Runtime: true, Model: true}, MemoryEstimate: &supervisor.MemoryEstimate{Fits: &fit}}}}
	a := &App{service: supervisor.NewAppService(b, supervisor.DefaultSettings(), nil, "", supervisor.AppHooks{})}
	seconds := 13
	if result := a.PatchPreferences(supervisor.PreferencesPatch{Set: supervisor.PreferenceValues{StallSeconds: &seconds}}); result.Error != nil || b.saved.StallSeconds != seconds {
		t.Fatal("preference delegation", result)
	}
	if v := a.GetView(); !v.Actions.Start.Allowed || v.Saved.StallSeconds != seconds {
		t.Fatal("View delegation", v.Actions)
	}
	if result := a.Start(); result.Error != nil || len(b.actions) != 1 || b.actions[0] != "start" {
		t.Fatal("Start delegation", result, b.actions)
	}
	if result := a.Stop(); result.Error == nil || len(b.actions) != 1 {
		t.Fatal("binding bypassed service state rule", result)
	}
}

type bindingKeyStore struct {
	key string
	err error
}

func (s *bindingKeyStore) Read(string) (string, error) {
	if s.err != nil {
		return "", s.err
	}
	if s.key == "" {
		return "", supervisor.ErrCredentialNotFound
	}
	return s.key, nil
}
func (s *bindingKeyStore) Write(_, key string) error {
	if s.err != nil {
		return s.err
	}
	s.key = key
	return nil
}
func (s *bindingKeyStore) Delete(string) error { s.key = ""; return s.err }

func TestNativeActionBindingsReturnStructuredResults(t *testing.T) {
	a := &App{service: supervisor.NewAppService(&bindingBackend{}, supervisor.DefaultSettings(), nil, "", supervisor.AppHooks{})}
	for _, test := range []struct {
		name string
		call func() *supervisor.CLIError
	}{
		{"RefreshModels", func() *supervisor.CLIError { return a.RefreshModels().Error }},
		{"CheckModel", func() *supervisor.CLIError { return a.CheckModel("model", false).Error }},
		{"CheckModelRepair", func() *supervisor.CLIError { return a.CheckModel("model", true).Error }},
		{"RepairPackage", func() *supervisor.CLIError { return a.RepairPackage("model", `C:ubo`).Error }},
		{"Download", func() *supervisor.CLIError { return a.Download("model").Error }},
		{"ConfirmDownload", func() *supervisor.CLIError { return a.ConfirmDownload("model").Error }},
		{"CancelDownload", func() *supervisor.CLIError { return a.CancelDownload().Error }},
		{"Estimate", func() *supervisor.CLIError { return a.Estimate("model", 0, 0).Error }},
		{"AddModelFolder", func() *supervisor.CLIError { return a.AddModelFolder(`C:\models`).Error }},
		{"RemoveModelFolder", func() *supervisor.CLIError { return a.RemoveModelFolder(`C:\models`).Error }},
		{"BrowseDownloadFolder", func() *supervisor.CLIError { return a.BrowseDownloadFolder().Error }},
		{"BrowseModelFolder", func() *supervisor.CLIError { return a.BrowseModelFolder().Error }},
		{"BrowseModelFile", func() *supervisor.CLIError { return a.BrowseModelFile().Error }},
		{"DismissModelJob", func() *supervisor.CLIError { return a.DismissModelJob().Error }},
		{"SetVisible", func() *supervisor.CLIError { return a.SetVisible(true).Error }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil || err.Code != "internal_error" || err.Remedy == "" {
				t.Fatal("binding lost service's structured error result", err)
			}
		})
	}
	if result := a.OpenModelFolder("relative"); result.Error == nil || result.Error.Code != "model_folder_invalid" {
		t.Fatal("folder policy failure was not typed", result)
	}
	if result := a.TakeTrayAction(); result.Error != nil || result.Value != "" {
		t.Fatal("empty tray action did not use the structured value result", result)
	}
	t.Log("native/result binding delegates checked: 15")
}

func TestKeyBindingsUseOneStructuredValuePath(t *testing.T) {
	store := &bindingKeyStore{key: strings.Repeat("ab", 32)}
	a := &App{service: supervisor.NewAppService(&bindingBackend{}, supervisor.DefaultSettings(), nil, "", supervisor.AppHooks{})}
	a.service.UseKeys(supervisor.NewKeyManager(store))
	if result := a.RevealKey(); result.Error != nil || result.Value != store.key {
		t.Fatal("RevealKey changed explicit value result", result.Error)
	}
	old := store.key
	if result := a.ResetKey(); result.Error != nil || store.key == old || len(store.key) != 64 {
		t.Fatal("ResetKey did not delegate to credential service", result.Error)
	}
	store.err = errors.New("test credential failure")
	if result := a.RevealKey(); result.Error == nil || result.Error.Remedy == "" || result.Value != "" {
		t.Fatal("RevealKey leaked a value on failure", result.Error)
	}
	if result := a.ResetKey(); result.Error == nil || result.Error.Remedy == "" {
		t.Fatal("ResetKey failure lost typed error", result)
	}
}
