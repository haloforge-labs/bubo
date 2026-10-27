//go:build windows

package supervisor

import (
	"sync/atomic"
	"testing"
	"time"
)

// TestLiveGPUSource reads the real counters. It only checks bounds and that
// nothing panics; it skips when the machine returns nothing (no GPU, a
// session without the counters).
func TestLiveGPUSource(t *testing.T) {
	src := NewGPUSource()
	defer src.Close()
	time.Sleep(1100 * time.Millisecond) // rate counters need two collects
	g := src.Sample()
	if g.Empty() {
		t.Skip("no GPU stats available on this machine")
	}
	t.Logf("live: use=%v used=%v total=%v shared=%v temp=%v power=%v", fp(g.UsePct), up(g.MemoryUsedBytes), up(g.MemoryTotalBytes), up(g.SharedUsedBytes), fp(g.TemperatureC), fp(g.ChipPowerW))
	if g.UsePct != nil && (*g.UsePct < 0 || *g.UsePct > 100) {
		t.Errorf("use_pct %v", *g.UsePct)
	}
	if g.TemperatureC != nil && (*g.TemperatureC <= 0 || *g.TemperatureC > 150) {
		t.Errorf("temperature %v", *g.TemperatureC)
	}
	if g.MemoryUsedBytes != nil && g.MemoryTotalBytes != nil && *g.MemoryUsedBytes > *g.MemoryTotalBytes {
		t.Errorf("used %d above total %d", *g.MemoryUsedBytes, *g.MemoryTotalBytes)
	}
	src.Close() // second Close and a Sample after Close must be safe
	if src.Sample() != nil {
		t.Error("Sample after Close must be nil")
	}
}

func fp(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}
func up(p *uint64) any {
	if p == nil {
		return nil
	}
	return *p
}

type fakeGPU struct {
	calls  atomic.Int32
	closed atomic.Bool
	stats  *GPUStats
}

func (f *fakeGPU) Sample() *GPUStats {
	f.calls.Add(1)
	if f.stats == nil {
		panic("a broken source must not take the supervisor down")
	}
	return f.stats
}
func (f *fakeGPU) Close() { f.closed.Store(true) }

func TestSnapshotCarriesGPUAndClosesSource(t *testing.T) {
	cfg := configFor(t, "ready")
	fake := &fakeGPU{stats: &GPUStats{UsePct: f64(12.5), MemoryTotalBytes: u64v(100)}}
	cfg.GPU = fake
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	var got *GPUStats
	for time.Now().Before(deadline) {
		if got = s.Snapshot().GPU; got != nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got == nil || got.UsePct == nil || *got.UsePct != 12.5 || got.TemperatureC != nil {
		t.Fatalf("snapshot gpu %+v", got)
	}
	// Polled at most once a second even though PollInterval is 40 ms.
	time.Sleep(500 * time.Millisecond)
	if n := fake.calls.Load(); n != 1 {
		t.Fatalf("sampled %d times in 0.5 s", n)
	}
	select {
	case <-s.Exit():
	case <-time.After(5 * time.Second):
		t.Fatal("Exit did not finish")
	}
	deadline = time.Now().Add(2 * time.Second)
	for !fake.closed.Load() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !fake.closed.Load() {
		t.Fatal("source not closed on exit")
	}
}

func TestSnapshotGPUNullWhenSourceBrokenOrAbsent(t *testing.T) {
	cfg := configFor(t, "ready")
	bad := &fakeGPU{} // Sample panics
	cfg.GPU = bad
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(300 * time.Millisecond)
	if s.Snapshot().GPU != nil || bad.calls.Load() == 0 {
		t.Fatalf("panicking source: gpu %+v calls %d", s.Snapshot().GPU, bad.calls.Load())
	}
	<-s.Exit()
	cfg2 := configFor(t, "ready")
	s2, err := New(cfg2)
	if err != nil {
		t.Fatal(err)
	}
	if s2.Snapshot().GPU != nil {
		t.Fatal("no source must give null")
	}
	<-s2.Exit()
}
