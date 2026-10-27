//go:build windows

package supervisor

import (
	"testing"
	"time"
)

func TestViewGPUReportsSampleAgeAndMeasuredWindow(t *testing.T) {
	sampled := time.Now().Add(-2 * time.Second).UTC().Format(time.RFC3339Nano)
	window, watts := 1.25, 12.5
	in := ViewInput{Snapshot: Snapshot{GPU: &GPUStats{SampleTime: sampled, WindowSeconds: &window, ChipPowerW: &watts}}}
	v := (&ViewBuilder{}).Build(in)
	if v.GPU.SampleTime != sampled || v.GPU.AgeSeconds == nil || *v.GPU.AgeSeconds < 2 || *v.GPU.AgeSeconds > 3 || *v.GPU.WindowSeconds != window {
		t.Fatalf("GPU timing was lost: %+v", v.GPU)
	}
	// The returned View owns its data, including new timing pointers.
	*v.GPU.WindowSeconds = 9
	if window != 1.25 {
		t.Fatal("View aliases the source window")
	}
	in.Snapshot.GPU.SampleTime, in.Snapshot.GPU.WindowSeconds = "", nil
	v = (&ViewBuilder{}).Build(in)
	if v.GPU.AgeSeconds != nil || v.GPU.WindowSeconds != nil {
		t.Fatalf("missing timing was fabricated: %+v", v.GPU)
	}
}

func TestViewForwardsStatusLogPathWithoutAnInstance(t *testing.T) {
	path := `D:\fixture\server.log`
	in := ViewInput{Snapshot: Snapshot{Status: &Status{LogPath: &path}}}
	v := (&ViewBuilder{}).Build(in)
	if v.LogPath == nil || *v.LogPath != path {
		t.Fatalf("CLI log path was lost: %+v", v.LogPath)
	}
	*v.LogPath = "changed"
	if path != `D:\fixture\server.log` {
		t.Fatal("View aliases the CLI log path")
	}
}
