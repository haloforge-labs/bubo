//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func streamSupervisor(t *testing.T, mode string) *Supervisor {
	t.Helper()
	c := configFor(t, mode)
	c.StatusStaleTimeout = 1500 * time.Millisecond
	c.HiddenStatusTimeout = 160 * time.Millisecond
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	return s
}

func waitFeed(t *testing.T, s *Supervisor, state string) Snapshot {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		v := s.Snapshot()
		if v.StatusFeed.State == state {
			return v
		}
		time.Sleep(10 * time.Millisecond)
	}
	v := s.Snapshot()
	t.Fatalf("wanted feed %s, got %+v, error=%+v", state, v.StatusFeed, v.StatusFeed.Error)
	return v
}

func streamFlag(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(os.Getenv("BUBO_FAKE_DIR"), name)
	if err := os.WriteFile(path, []byte("test"), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func statusChildren(t *testing.T, name string) []uint32 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), name))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var pids []uint32
	for _, line := range strings.Fields(string(b)) {
		n, err := strconv.ParseUint(line, 10, 32)
		if err != nil {
			t.Fatal(err)
		}
		pids = append(pids, uint32(n))
	}
	return pids
}

func assertStatusChildrenExited(t *testing.T, pids []uint32) {
	t.Helper()
	if b, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "watch-overlap")); err == nil {
		t.Fatalf("overlapping status children: %s", b)
	}
	for _, pid := range pids {
		h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, pid)
		if err != nil {
			continue
		}
		state, err := windows.WaitForSingleObject(h, 1000)
		windows.CloseHandle(h)
		if err != nil || state != windows.WAIT_OBJECT_0 {
			t.Fatalf("owned status PID %d still alive: %v", pid, err)
		}
	}
}

func TestStatusStreamFeedsViewWithoutPolling(t *testing.T) {
	s := streamSupervisor(t, "normal")
	v := waitFeed(t, s, StatusUpdating)
	first := v.Status.SampleTime
	oneShots := len(statusChildren(t, "status-oneshots.txt"))
	waitFor(t, s, "next streamed View", 3*time.Second, func(v Snapshot) bool { return v.Status.SampleTime != first })
	var builder ViewBuilder
	view := builder.Build(ViewInput{Snapshot: s.Snapshot()})
	if view.StatusFeed.State != StatusUpdating || view.StatusFeed.SampleTime == "" || view.StatusFeed.AgeSeconds == nil || view.Stale {
		t.Fatalf("View freshness %+v", view.StatusFeed)
	}
	if n := len(statusChildren(t, "status-oneshots.txt")); n != oneShots {
		t.Fatalf("poll spawned helpers: %d -> %d", oneShots, n)
	}
	if pids := statusChildren(t, "watches.jsonl"); len(pids) != 1 {
		t.Fatalf("watch children: %v", pids)
	}
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
}

func TestStatusStreamStallsAndRecovers(t *testing.T) {
	s := streamSupervisor(t, "normal")
	// The stale badge belongs to an established running server. Exercise
	// that lifecycle state while retaining the feed age and recovery checks.
	reply(t, s.Start())
	waitState(t, s, Ready)
	waitFeed(t, s, StatusUpdating)
	flag := streamFlag(t, "watch-stall")
	v := waitFeed(t, s, StatusNotUpdating)
	var builder ViewBuilder
	view := builder.Build(ViewInput{Snapshot: v})
	if !view.Stale || view.StatusFeed.AgeSeconds == nil || *view.StatusFeed.AgeSeconds < .3 || view.Stall != nil {
		t.Fatalf("status silence confused with token stall: %+v", view.StatusFeed)
	}
	if err := os.Remove(flag); err != nil {
		t.Fatal(err)
	}
	waitFeed(t, s, StatusUpdating)
}

func TestStatusStreamSilenceBeforeFirstRecord(t *testing.T) {
	c := configFor(t, "normal")
	c.StatusStaleTimeout = 1500 * time.Millisecond
	streamFlag(t, "watch-stall")
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	v := waitFeed(t, s, StatusNotUpdating)
	if !v.Stale || len(statusChildren(t, "watches.jsonl")) != 1 {
		t.Fatal("silent stream was presented as current or restarted")
	}
}

func TestStatusStreamOldSamplesDoNotClaimFreshness(t *testing.T) {
	t.Setenv("BUBO_FAKE_WATCH_SAMPLE_TIME", time.Now().Add(-time.Minute).UTC().Format(time.RFC3339))
	s := streamSupervisor(t, "normal")
	v := waitFeed(t, s, StatusNotUpdating)
	if v.StatusFeed.AgeSeconds == nil || *v.StatusFeed.AgeSeconds < 60 || !v.Stale {
		t.Fatalf("old sample age %+v", v.StatusFeed)
	}
	first := v.StatusRecordAt
	waitFor(t, s, "records still arrive", 3*time.Second, func(v Snapshot) bool { return v.Status != nil && v.StatusRecordAt.After(first) })
	if len(statusChildren(t, "watches.jsonl")) != 1 {
		t.Fatal("old sample killed an otherwise live stream")
	}
}

func TestStatusStreamRestartsAfterExitWithBackoff(t *testing.T) {
	s := streamSupervisor(t, "normal")
	waitFeed(t, s, StatusUpdating)
	flag := streamFlag(t, "watch-exit")
	waitFeed(t, s, StatusReconnecting)
	time.Sleep(850 * time.Millisecond)
	pids := statusChildren(t, "watches.jsonl")
	if len(pids) < 2 || len(pids) > 4 {
		t.Fatalf("restart did not back off: %v", pids)
	}
	assertStatusChildrenExited(t, pids)
	if err := os.Remove(flag); err != nil {
		t.Fatal(err)
	}
	waitFeed(t, s, StatusUpdating)
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
}

func TestStatusStreamVisibilitySuspendsAndResumes(t *testing.T) {
	s := streamSupervisor(t, "normal")
	waitFeed(t, s, StatusUpdating)
	pids := statusChildren(t, "watches.jsonl")
	s.SetVisible(false)
	waitFeed(t, s, StatusPaused)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "watch-"+strconv.Itoa(int(pids[0]))+"-closed")); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "watch-"+strconv.Itoa(int(pids[0]))+"-closed")); err != nil {
		t.Fatal("hidden child did not close through stdin", err)
	}
	assertStatusChildrenExited(t, pids)
	s.SetVisible(true)
	waitFeed(t, s, StatusUpdating)
	if pids := statusChildren(t, "watches.jsonl"); len(pids) != 2 {
		t.Fatalf("show did not resume exactly once: %v", pids)
	}
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
}

func TestStatusStreamRejectsProtocolErrors(t *testing.T) {
	for _, mode := range []string{"watch-bad-json", "watch-bad-sequence", "watch-wrong-instance"} {
		t.Run(mode, func(t *testing.T) {
			s := streamSupervisor(t, mode)
			if mode == "watch-wrong-instance" {
				reply(t, s.Start())
				waitState(t, s, Ready)
			}
			v := waitFeed(t, s, StatusReconnecting)
			if v.StatusFeed.Error == nil || v.StatusFeed.Error.Code != "protocol_mismatch" || !v.Stale {
				t.Fatalf("protocol error lost: %+v", v.StatusFeed)
			}
		})
	}
}

func TestStatusStreamMissingCapabilityIsUnsupported(t *testing.T) {
	s := streamSupervisor(t, "no-status-watch")
	v := waitFeed(t, s, StatusUnsupported)
	if v.StatusFeed.Error == nil || v.StatusFeed.Error.Code != "protocol_mismatch" || len(statusChildren(t, "watches.jsonl")) != 0 {
		t.Fatalf("unsupported watch: %+v", v.StatusFeed)
	}
}

func TestStatusStreamRetainsFinalCLIErrorAndRawExit(t *testing.T) {
	s := streamSupervisor(t, "watch-last-error")
	v := waitFeed(t, s, StatusReconnecting)
	if v.StatusFeed.Error == nil || v.StatusFeed.Error.Code != "storage_failed" || v.StatusFeed.Error.Remedy != "Final status error fixture." {
		t.Fatalf("final error lost: %+v", v.StatusFeed)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(strings.Join(s.Logs(), "\n"), "status stream exit: 7") {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("raw exit code was not retained")
}

func TestStatusStreamManagedRestartKeepsOneChild(t *testing.T) {
	s := streamSupervisor(t, "normal")
	reply(t, s.Start())
	v := waitState(t, s, Ready)
	first := v.Instance.ID
	waitFeed(t, s, StatusUpdating)
	watches := statusChildren(t, "watches.jsonl")
	shots := len(statusChildren(t, "status-oneshots.txt"))
	time.Sleep(150 * time.Millisecond)
	if len(statusChildren(t, "status-oneshots.txt")) != shots {
		t.Fatal("ready supervision spawned one-shot status polls")
	}
	reply(t, s.Restart())
	waitFor(t, s, "new owned identity", 5*time.Second, func(v Snapshot) bool { return v.State == Ready && v.Instance != nil && v.Instance.ID != first })
	waitFeed(t, s, StatusUpdating)
	assertStatusChildrenExited(t, watches)
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
}

func TestStatusStreamExitBoundsUnresponsiveChild(t *testing.T) {
	c := configFor(t, "watch-ignore-stdin")
	c.QueryTimeout = time.Second
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	waitFeed(t, s, StatusUpdating)
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
}

func TestStatusStreamExternalUsesOneChildAndLeavesServer(t *testing.T) {
	c := configFor(t, "normal")
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	_, instance := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	waitState(t, s, AttachedExternal)
	waitFeed(t, s, StatusUpdating)
	shots := len(statusChildren(t, "status-oneshots.txt"))
	time.Sleep(150 * time.Millisecond)
	if len(statusChildren(t, "status-oneshots.txt")) != shots {
		t.Fatal("external supervision spawned status polls")
	}
	reply(t, s.Exit())
	assertStatusChildrenExited(t, statusChildren(t, "watches.jsonl"))
	if err := VerifyIdentity(instance, c.Exe); err != nil {
		t.Fatal("status shutdown touched the external server", err)
	}
}
