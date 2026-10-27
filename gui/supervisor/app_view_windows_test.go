//go:build windows

package supervisor

import (
	"context"
	"strings"
	"testing"
)

func TestTypedStreamBridgeUsesInjectedHeadlessTransport(t *testing.T) {
	a := newTestApp(&fakeBackend{}, Settings{}, nil, "")
	var seen []StreamKind
	a.events = func(_ context.Context, e StreamEvent) { seen = append(seen, e.Kind) }
	a.emitStream(StreamEvent{Kind: LogLine, Line: "before startup"})
	if len(seen) != 0 {
		t.Fatal("stream emitted before startup")
	}
	a.ctx = context.Background()
	a.emitStream(StreamEvent{Kind: PrepareProgress})
	if len(seen) != 1 || seen[0] != PrepareProgress {
		t.Fatal(seen)
	}
}

func TestTypedActionsDelegateToSupervisor(t *testing.T) {
	b := &fakeBackend{view: actionTestInput(Stopped).Snapshot}
	a := newTestApp(b, DefaultSettings(), nil, "")
	for i, f := range []func() ActionResult{a.Start, a.Stop, a.Restart, a.Prepare, a.Cancel, a.Approve, a.Decline, a.DismissError, a.Manage, a.Diagnose, a.PreviewBundle} {
		if i == 1 {
			b.view.State = Ready
		}
		if result := f(); result.Error != nil {
			t.Fatal(result.Error)
		}
	}
	if result := a.SelectInstance("test-instance"); result.Error != nil {
		t.Fatal(result.Error)
	}
	if result := a.SaveBundle("test-path"); result.Error != nil {
		t.Fatal(result.Error)
	}
	want := "start,stop,restart,prepare,cancel,approve,decline,dismiss,manage,diagnose,preview,select,bundle"
	if strings.Join(b.actions, ",") != want {
		t.Fatal(b.actions)
	}
	v := a.GetView()
	if v.Update.State != UpdateNone || v.Update.Error != nil || v.Logs[0] != "sample CLI log" {
		t.Fatal("typed view or unchecked update state missing")
	}
	for _, action := range []func() ActionResult{a.CheckUpdate, a.DownloadUpdate, a.CancelUpdate, a.InstallUpdate, a.DismissUpdate} {
		if result := action(); result.Error == nil || result.Error.Code != "internal_error" {
			t.Fatal("unwired update reported success", result)
		}
	}
	b.view.State = Stopped
	b.actionErr = &CommandError{Event: Event{Value: &CLIError{Code: "future_cli_error", Remedy: "CLI remedy", Retryable: true}}}
	if result := a.Start(); result.Error == nil || result.Error.Code != "future_cli_error" || result.Error.Remedy != "CLI remedy" || !result.Error.Retryable {
		t.Fatal("CLI error was not preserved", result)
	}
	if result := a.Download("missing"); result.Error == nil || result.Error.Code != "internal_error" {
		t.Fatal("missing model service reported success", result)
	}
}
