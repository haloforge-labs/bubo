//go:build windows

package supervisor

import (
	"strings"
	"testing"

	"bubo/gui/updater"
)

// E21w W1, GUI side. The updater reports a handoff whose helper ended before it
// changed anything as a failed result with code update_interrupted.
//
// Ways this could fail, written before the code:
//
//	S1 the first launch is silent (no notice, no state).
//	S2 the notice says "Update failed." and then repeats the sentence the error
//	   box shows, or it shows the updater's own words ("helper", "handoff").
//	S3 the error code is lost, so the front end cannot word it.
//	S4 the state is not "failed", so the card offers no Try again.
//	S5 the update is marked a failed check, so no header pill and no Try again box.
//	S6 the user cannot dismiss it.
func TestUpdateInterruptedResultIsShownOnceInPlainWords(t *testing.T) {
	client := &clearingStubClient{stubUpdateClient: &stubUpdateClient{result: &updater.InstallResult{Status: "failed", Version: "0.1.2",
		Failure: &updater.Failure{Code: updater.CodeUpdateStopped, Message: "The update stopped before it changed anything. Your installed version is unchanged."}}}}
	u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot, Enabled: func() bool { return false }})
	u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
	defer u.Close()
	u.Start(readyWindow())
	u.wg.Wait()
	v := u.State()
	if v.State != UpdateFailed || v.CheckFailed { // S4 S5
		t.Fatalf("state=%q check_failed=%v", v.State, v.CheckFailed)
	}
	if v.Error == nil || v.Error.Code != "update_interrupted" || !v.Error.Retryable { // S3
		t.Fatalf("error=%+v", v.Error)
	}
	if v.LastResultNotice != client.result.Failure.Message { // S1 S2
		t.Fatalf("notice=%q", v.LastResultNotice)
	}
	for _, word := range []string{"Update failed", "helper", "handoff", "lock", "journal"} {
		if strings.Contains(strings.ToLower(v.LastResultNotice), strings.ToLower(word)) {
			t.Fatalf("notice has %q: %q", word, v.LastResultNotice)
		}
	}
	if r := u.DismissUpdate(); r.Error != nil || u.State().State != UpdateNone || client.cleared != 1 { // S6: the saved result goes too, so it is shown once
		t.Fatalf("dismiss: %+v %+v cleared=%d", r, u.State(), client.cleared)
	}
}

// clearingStubClient adds the saved-result removal the real updater client has.
type clearingStubClient struct {
	*stubUpdateClient
	cleared int
}

func (c *clearingStubClient) ClearLastResult() *updater.Failure { c.cleared++; return nil }
