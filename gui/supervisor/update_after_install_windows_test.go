//go:build windows

package supervisor

import (
	"context"
	"sync"
	"testing"

	"bubo/gui/updater"
)

// Walk WU F1. The new GUI that the helper starts after an update reads the
// update result while the helper finishes, and the startup check ran at the same
// time. The real client lets one operation run at a time (a TryLock), so the
// check answered busy and About said "Bubo can't check for updates: another Bubo
// action is still running." beside "Updated to 0.1.2." (both WU attempts).
//
// Ways this could fail, written before the code:
//
//	P1 the startup check runs while the result is still pending, and its busy
//	   answer is shown as a failed check.
//	P2 the "installed" result is lost: no "Updated to 0.1.2." notice.
//	P3 the startup check never runs after a pending result, so a newer release
//	   is never offered.
//	P4 a startup check that Settings turned off runs anyway.
//	P5 with no pending result, the startup check changes (it runs once, at once).
func TestStartupCheckWaitsForThePendingUpdateResult(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		client := &pendingResultClient{stubUpdateClient: &stubUpdateClient{
			check:  updater.CheckResult{Status: updater.UpToDate},
			result: &updater.InstallResult{Status: "installed", Version: "0.1.2"},
		}, pendingReads: 3}
		u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot, Enabled: func() bool { return enabled }})
		u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
		u.Start(readyWindow())
		u.wg.Wait()
		v := u.State()
		u.Close()
		if client.busyChecks != 0 || v.CheckFailed || v.Error != nil { // P1
			t.Fatalf("enabled=%v: a check ran beside the pending result: busy=%d state=%+v", enabled, client.busyChecks, v)
		}
		if v.LastResultNotice != "Updated to 0.1.2." { // P2
			t.Fatalf("enabled=%v: notice=%q", enabled, v.LastResultNotice)
		}
		want := 0
		if enabled {
			want = 1
		}
		if client.checks != want { // P3 P4
			t.Fatalf("enabled=%v: %d startup checks, want %d", enabled, client.checks, want)
		}
		if enabled && v.State != UpdateCurrent {
			t.Fatalf("state=%q after the check", v.State)
		}
	}
	// P5: nothing pending; one check at start, as before.
	plain := &stubUpdateClient{check: updater.CheckResult{Status: updater.UpToDate}}
	u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot})
	u.factory = func(updater.Config) (updateClient, *updater.Failure) { return plain, nil }
	defer u.Close()
	u.Start(readyWindow())
	u.wg.Wait()
	if plain.checks != 1 || u.State().State != UpdateCurrent {
		t.Fatalf("plain start: checks=%d state=%q", plain.checks, u.State().State)
	}
}

// pendingResultClient answers LastResult with nothing for its first reads, as
// the real client does while the helper finishes, and answers a Check made
// before the result is read with busy, as the real client's lock does.
type pendingResultClient struct {
	*stubUpdateClient
	mu           sync.Mutex
	pendingReads int
	delivered    bool
	busyChecks   int
}

func (c *pendingResultClient) PendingResult() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.delivered
}

func (c *pendingResultClient) LastResult() (*updater.InstallResult, *updater.Failure) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pendingReads > 0 {
		c.pendingReads--
		return nil, nil
	}
	c.delivered = true
	return c.result, nil
}

func (c *pendingResultClient) Check(ctx context.Context) updater.CheckResult {
	c.mu.Lock()
	delivered := c.delivered
	if !delivered {
		c.busyChecks++
	}
	c.mu.Unlock()
	if !delivered {
		return updater.CheckResult{Status: updater.Failed, Failure: &updater.Failure{Code: updater.CodeBusy, Message: "another updater operation is running"}}
	}
	return c.stubUpdateClient.Check(ctx)
}
