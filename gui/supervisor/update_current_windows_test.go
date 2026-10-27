//go:build windows

package supervisor

import (
	"testing"

	"bubo/gui/updater"
)

func TestUpdateNoneAndCheckedCurrentAreDistinct(t *testing.T) {
	v := new(ViewBuilder).Build(ViewInput{})
	if v.Update.State != UpdateNone || v.Update.Error != nil {
		t.Fatal("an unattached updater was reported as a failed check", v.Update)
	}
	client := &stubUpdateClient{check: updater.CheckResult{Status: updater.UpToDate}}
	u := NewUpdateService(UpdateServiceConfig{Snapshot: saneUpdateSnapshot, Enabled: func() bool { return false }})
	u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
	defer u.Close()
	if u.State().State != UpdateNone {
		t.Fatal("new updater claimed a completed check")
	}
	u.Start(readyWindow())
	u.wg.Wait()
	if u.State().State != UpdateNone || client.checks != 0 {
		t.Fatal("disabled automatic checks claimed current", u.State())
	}
	if r := u.CheckUpdate(); r.Error != nil || u.State().State != UpdateCurrent || client.checks != 1 || u.State().Error != nil {
		t.Fatal("successful current check was not represented", r, u.State())
	}
	client.check = updater.CheckResult{Status: updater.Failed, Failure: &updater.Failure{Code: updater.CodeIO, Message: "fixture failure"}}
	if r := u.CheckUpdate(); r.Error == nil || u.State().State != UpdateFailed {
		t.Fatal("failed later check retained a current claim", r, u.State())
	}
}
