//go:build windows

package supervisor

import (
	"path/filepath"
	"testing"
	"time"
)

func TestStartWaitsForNewSavedMemoryCheck(t *testing.T) {
	m := newModelsApp(t, "")
	m.b.view = actionTestInput(Stopped).Snapshot
	m.b.view.Status.MemoryEstimate = nil
	m.a.settings = Settings{ManifestPath: filepath.Join(e21Folder(t, e21Flash), "bubo-package.json"), Context: 65536, Sessions: 1}
	if result := m.a.Start(); result.Error != nil {
		t.Fatal("Start did not wait for the saved setup check", result.Error)
	}
	m.b.mu.Lock()
	defer m.b.mu.Unlock()
	if len(m.b.actions) != 1 || m.b.actions[0] != "start" || m.b.settings.Context != 65536 {
		t.Fatal("Start did not use the saved draft", m.b.actions, m.b.settings)
	}
	if len(recordedEstimates(t, m.dir)) == 0 {
		t.Fatal("Start skipped the memory check")
	}
}

func TestSuccessfulStatusClearsPriorProtocolFailure(t *testing.T) {
	s := streamSupervisor(t, "normal")
	waitFeed(t, s, StatusUpdating)
	s.emit(update{s.Snapshot().Generation, "failure", Failure{Code: "protocol_mismatch", Source: "supervisor", Time: time.Now().Add(-time.Second)}})
	waitFor(t, s, "failure recorded", time.Second, func(v Snapshot) bool { return v.LastError != nil })
	waitFor(t, s, "later successful status clears prior error", 3*time.Second, func(v Snapshot) bool { return v.LastError == nil && v.State == Stopped })
}
