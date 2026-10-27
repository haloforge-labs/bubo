package supervisor

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)



const (
	radeonLUID = "luid_0x00000000_0x000125fd_"
	softLUID   = "luid_0x00000000_0x0001480a_"
	npuLUID    = "luid_0x00000000_0x000148d4_"
)

func f64(v float64) *float64 { return &v }
func u64v(v uint64) *uint64  { return &v }

func TestLUIDPrefixAndInstanceFilter(t *testing.T) {
	if got := luidPrefix(0, 0x125fd); got != radeonLUID {
		t.Fatalf("prefix %q", got)
	}
	if got := luidPrefix(0xABCD, 0x1480A); got != "luid_0x0000abcd_0x0001480a_" {
		t.Fatalf("prefix %q", got)
	}
	cases := []struct {
		name, prefix string
		want         bool
	}{
		{"pid_4312_luid_0x00000000_0x000125fd_phys_0_eng_0_engtype_3D", radeonLUID, true},
		{"luid_0x00000000_0x000125fd_phys_0", radeonLUID, true},
		{"LUID_0X00000000_0X000125FD_PHYS_0", radeonLUID, true},
		{"pid_9_luid_0x00000000_0x0001480a_phys_0_eng_0_engtype_3D", radeonLUID, false},
		{"luid_0x00000000_0x000125fd0_phys_0", radeonLUID, false}, // longer low part is another adapter
		{"anything", "", false},
		{"", radeonLUID, false},
	}
	for _, c := range cases {
		if got := instanceHasLUID(c.name, c.prefix); got != c.want {
			t.Errorf("instanceHasLUID(%q,%q)=%v want %v", c.name, c.prefix, got, c.want)
		}
	}
}

func TestEngineType(t *testing.T) {
	for in, want := range map[string]string{
		"pid_1_luid_0x00000000_0x000125fd_phys_0_eng_0_engtype_3D":        "3D",
		"pid_1_luid_0x00000000_0x000125fd_phys_0_eng_5_engtype_Compute_0": "Compute_0",
		"pid_1_luid_0x00000000_0x000125fd_phys_0_eng_5":                   "",
	} {
		if got := engineType(in); got != want {
			t.Errorf("engineType(%q)=%q want %q", in, got, want)
		}
	}
}

func eng(luid string, pid string, idx string, typ string, v float64) pdhItem {
	return pdhItem{Name: "pid_" + pid + "_" + luid + "phys_0_eng_" + idx + "_engtype_" + typ, Value: v}
}

func TestBusiestEngineSumsPerTypeThenTakesMax(t *testing.T) {
	items := []pdhItem{
		eng(radeonLUID, "10", "0", "3D", 2.73),
		eng(radeonLUID, "11", "0", "3D", 40),
		eng(radeonLUID, "10", "1", "Compute_0", 30),
		eng(radeonLUID, "12", "1", "Compute_0", 25),
		eng(softLUID, "13", "0", "3D", 99), // another adapter: ignored
		eng(npuLUID, "14", "0", "Compute_0", 99),
	}
	got, ok := busiestEngine(items, radeonLUID)
	if !ok || math.Abs(got-55) > 1e-9 {
		t.Fatalf("got %v %v want 55", got, ok)
	}

	idle, ok := busiestEngine([]pdhItem{eng(radeonLUID, "1", "0", "3D", 2.73)}, radeonLUID)
	if !ok || idle != 2.73 {
		t.Fatalf("idle %v %v", idle, ok)
	}
	// Sum above 100 is capped.
	capped, _ := busiestEngine([]pdhItem{eng(radeonLUID, "1", "0", "3D", 70), eng(radeonLUID, "2", "0", "3D", 70)}, radeonLUID)
	if capped != 100 {
		t.Fatalf("cap %v", capped)
	}
	// Absent: no instance for this adapter, bad values only, no prefix.
	if _, ok := busiestEngine(items, "luid_0x00000000_0x00000001_"); ok {
		t.Fatal("unknown adapter must be absent")
	}
	if _, ok := busiestEngine([]pdhItem{eng(radeonLUID, "1", "0", "3D", math.NaN()), eng(radeonLUID, "1", "0", "3D", -1)}, radeonLUID); ok {
		t.Fatal("NaN and negative values must be absent")
	}
	if _, ok := busiestEngine(nil, radeonLUID); ok {
		t.Fatal("empty list must be absent")
	}
}

func TestUnitConversions(t *testing.T) {
	if v := deciCelsius(470); v == nil || *v != 47.0 {
		t.Fatalf("470 deci -> %v", v)
	}
	if v := deciCelsius(615); v == nil || *v != 61.5 {
		t.Fatalf("615 deci -> %v", v)
	}
	if deciCelsius(0) != nil || deciCelsius(1501) != nil {
		t.Fatal("0 and above 150 C must be absent")
	}

	for in, want := range map[float64]float64{25347: 25.3, 17008: 17.0, 17952: 18.0} {
		if v := milliwattsToWatts(in); v == nil || *v != want {
			t.Errorf("%v mW -> %v want %v", in, v, want)
		}
	}
	if milliwattsToWatts(0) != nil || milliwattsToWatts(-5) != nil || milliwattsToWatts(math.Inf(1)) != nil {
		t.Fatal("0, negative and infinite power must be absent")
	}
	if bytesPtr(-1) != nil || bytesPtr(math.NaN()) != nil || bytesPtr(1e30) != nil {
		t.Fatal("bad byte counts must be absent")
	}
	if v := bytesPtr(2522112000); v == nil || *v != 2522112000 {
		t.Fatal("bytes")
	}
}

func TestBuildGPUStatsWithProbeValues(t *testing.T) {
	raw := gpuRaw{
		LUIDPrefix: radeonLUID,
		TotalBytes: 102892924928, // GETSEGMENTSIZE dedicated, run1/run2
		TempDeci:   470,          // ADAPTERPERFDATA Temperature, run1/run2
		Engines: []pdhItem{
			eng(radeonLUID, "1", "0", "3D", 2.73),
			eng(softLUID, "2", "0", "3D", 50),
		},
		Dedicated: []pdhItem{
			{Name: "luid_0x00000000_0x000125fd_phys_0", Value: 2522112000},
			{Name: "luid_0x00000000_0x0001480a_phys_0", Value: 7},
		},
		Shared: []pdhItem{
			{Name: "luid_0x00000000_0x000125fd_phys_0", Value: 900657152},
			{Name: "luid_0x00000000_0x0001480a_phys_0", Value: 9},
		},
		PowerMW: 25347, HavePowerRaw: true,
	}
	g := buildGPUStats(raw)
	if g == nil {
		t.Fatal("nil")
	}
	want := GPUStats{UsePct: f64(2.7), MemoryUsedBytes: u64v(2522112000), MemoryTotalBytes: u64v(102892924928),
		SharedUsedBytes: u64v(900657152), TemperatureC: f64(47), ChipPowerW: f64(25.3)}
	gb, _ := json.Marshal(g)
	wb, _ := json.Marshal(want)
	if string(gb) != string(wb) {
		t.Fatalf("\n got %s\nwant %s", gb, wb)
	}
}

func TestBuildGPUStatsAbsentFields(t *testing.T) {
	if g := buildGPUStats(gpuRaw{}); g != nil {
		t.Fatalf("nothing returned must give nil, got %+v", g)
	}
	// D3DKMT failed (no adapter chosen): PDH values cannot be attributed, so
	// only the power reading (not tied to an adapter) can remain.
	g := buildGPUStats(gpuRaw{Engines: []pdhItem{eng(radeonLUID, "1", "0", "3D", 5)}, PowerMW: 17008, HavePowerRaw: true})
	if g == nil || g.UsePct != nil || g.MemoryUsedBytes != nil || g.MemoryTotalBytes != nil || g.TemperatureC != nil {
		t.Fatalf("no adapter: %+v", g)
	}
	if g.ChipPowerW == nil || *g.ChipPowerW != 17.0 {
		t.Fatalf("power %+v", g.ChipPowerW)
	}
	// PDH failed entirely: totals and temperature remain.
	g = buildGPUStats(gpuRaw{LUIDPrefix: radeonLUID, TotalBytes: 100, TempDeci: 0})
	if g == nil || g.MemoryTotalBytes == nil || g.UsePct != nil || g.TemperatureC != nil || g.ChipPowerW != nil || g.MemoryUsedBytes != nil || g.SharedUsedBytes != nil {
		t.Fatalf("pdh failed: %+v", g)
	}
	// Absent fields marshal as null, never as zero.
	b, _ := json.Marshal(g)
	for _, k := range []string{`"use_pct":null`, `"memory_used_bytes":null`, `"shared_used_bytes":null`, `"temperature_c":null`, `"chip_power_w":null`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("%s missing in %s", k, b)
		}
	}
	// A power counter that never produced data is not 0 W.
	if g = buildGPUStats(gpuRaw{LUIDPrefix: radeonLUID, TotalBytes: 1, PowerMW: 0, HavePowerRaw: false}); g.ChipPowerW != nil {
		t.Fatal("power without data must be absent")
	}
}

func TestGPUStatsEmpty(t *testing.T) {
	var nilStats *GPUStats
	if !nilStats.Empty() || !(&GPUStats{}).Empty() || (&GPUStats{UsePct: f64(0)}).Empty() {
		t.Fatal("Empty")
	}
}
