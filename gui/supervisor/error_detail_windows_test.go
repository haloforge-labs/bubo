//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// E14 item 5: an optional `detail` on CLI errors reaches the View, for Details.
func TestServeFailureCarriesTheCLIDetail(t *testing.T) {
	s := supervisorFor(t, "ui-fail")
	t.Setenv("BUBO_FAKE_ERROR_DETAIL", "HIP error 700 at kernel decode_attn")
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600)
	reply(t, s.Start())
	waitState(t, s, Ready)
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "fail"), []byte("fail"), 0600)
	v := waitFor(t, s, "failure", 8*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	if v.LastError == nil || v.LastError.Detail != "HIP error 700 at kernel decode_attn" {
		t.Fatalf("detail lost: %+v", v.LastError)
	}
	view := (&ViewBuilder{}).Build(ViewInput{Snapshot: v})
	if view.Failure == nil || view.Failure.Detail != "HIP error 700 at kernel decode_attn" {
		t.Fatalf("the View lost the detail: %+v", view.Failure)
	}
}

func TestModelJobFailureCarriesTheCLIDetail(t *testing.T) {
	t.Setenv("BUBO_FAKE_ERROR_DETAIL", "write failed at shard 2")
	r := newModelRig(t, "model-error")
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	if j.State != "failed" || j.Error == nil || j.Error.Detail == "" {
		t.Fatalf("the model job lost the detail: %+v", j.Error)
	}
}
