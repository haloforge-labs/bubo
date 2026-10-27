//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type runtimeChild struct {
	Args    []string
	Runtime string
	Present bool
	Exe     string
}

func runtimeChildren(t *testing.T, dir string) []runtimeChild {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "runtime-children.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var out []runtimeChild
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var row runtimeChild
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatal(err)
		}
		out = append(out, row)
	}
	return out
}

func TestPackagedCLIPathIndependentOfRuntime(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := DefaultCLIPath(), filepath.Join(filepath.Dir(exe), "bin", "bubo.exe"); got != want {
		t.Fatalf("%s != %s", got, want)
	}
}

func TestRuntimeEnvironmentReachesStatusServePrepare(t *testing.T) {
	for _, custom := range []bool{false, true} {
		t.Run(map[bool]string{false: "shared", true: "custom"}[custom], func(t *testing.T) {
			c := configFor(t, "ui")
			t.Setenv("BUBO_FAKE_RECORD_RUNTIME", "1")
			t.Setenv("BUBO_RUNTIME_DIR", `C:\inherited-must-not-leak`)
			s := DefaultSettings()
			if custom {
				s.RuntimePath = filepath.Join(t.TempDir(), "Runtime 日本語")
			}
			// No port override: readiness accepts the port chosen by the fake CLI.
			c.Port = 0
			b, err := NewBackend(c, s, c.Exe)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { reply(t, b.Action("exit", s, "")) })
			waitBackend(t, b, func(v Snapshot) bool { return v.Initialized })
			reply(t, b.Action("prepare", s, ""))
			first := waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
			if len(first.Approval.Components) != 1 || first.Approval.Components[0] != "runtime" {
				t.Fatal("first consent must be runtime-only", first.Approval)
			}
			reply(t, b.Action("approve", s, ""))
			waitBackend(t, b, func(v Snapshot) bool {
				return v.State == Stopped && v.Status != nil && v.Status.Readiness.Runtime && !v.Status.Readiness.Model
			})
			reply(t, b.s.Prepare("model"))
			second := waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
			if len(second.Approval.Components) != 1 || second.Approval.Components[0] != "model" {
				t.Fatal("second consent must be model-only", second.Approval)
			}
			reply(t, b.Action("approve", s, ""))
			waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped && v.Status != nil && v.Status.Readiness.Model })
			reply(t, b.Action("start", s, ""))
			waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
			reply(t, b.Action("stop", s, ""))
			waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped })
			seen := map[string]bool{}
			for _, row := range runtimeChildren(t, filepath.Dir(c.LogDir)) {
				if row.Runtime != s.RuntimePath || row.Present != custom || !strings.EqualFold(row.Exe, c.Exe) {
					t.Fatalf("wrong child runtime/exe: %+v", row)
				}
				args := strings.Join(row.Args, " ")
				if strings.Contains(args, "--runtime-dir") || strings.Contains(args, "--port") {
					t.Fatal("invented override", args)
				}
				seen[row.Args[0]] = true
			}
			for _, command := range []string{"status", "serve", "prepare"} {
				if !seen[command] {
					t.Fatal("child not observed", command)
				}
			}
		})
	}
}

func TestModelConfirmationKeepsPlannedRuntime(t *testing.T) {
	r := newModelRig(t, "models")
	t.Setenv("BUBO_FAKE_RECORD_RUNTIME", "1")
	first := filepath.Join(t.TempDir(), "first runtime")
	r.setInputs(ModelInputs{RuntimeDir: first})
	plan, err := r.svc.StartDownload(e21Flash, "download")
	if err != nil {
		t.Fatal(err)
	}
	r.setInputs(ModelInputs{RuntimeDir: filepath.Join(t.TempDir(), "next runtime")})
	if err := r.svc.ConfirmDownload(plan.ID); err != nil {
		t.Fatal(err)
	}
	if done := r.waitJob(t, func(j ModelJob) bool { return j.State == "done" || j.State == "failed" }); done.State != "done" {
		t.Fatalf("confirmation failed: %+v", done)
	}
	rows := runtimeChildren(t, r.dir)
	if len(rows) != 2 {
		t.Fatalf("expected plan + confirmation: %+v", rows)
	}
	for _, row := range rows {
		if row.Runtime != first || !row.Present {
			t.Fatalf("runtime changed after approval: %+v", row)
		}
	}
}

func TestViewReportsCLIRuntimePathKindReadiness(t *testing.T) {
	st := viewFixture(t, "status").Value.(*Status)
	st.Paths = map[string]json.RawMessage{"runtime": json.RawMessage(`"D:/chosen/runtime"`), "runtime_kind": json.RawMessage(`"custom"`)}
	st.Readiness.Runtime = true
	var builder ViewBuilder
	v := builder.Build(ViewInput{Snapshot: Snapshot{Status: st}, Saved: Settings{RuntimePath: "different preference"}})
	if v.Runtime.Paths["runtime"].Text != "D:/chosen/runtime" || v.Runtime.Kind != "custom" || !v.Runtime.Readiness.Runtime {
		t.Fatalf("runtime was inferred instead of projected: %+v", v.Runtime)
	}
}
