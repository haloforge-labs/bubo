package supervisor

import (
	"fmt"
	"math"
	"strings"
)

// GPUStats is the GUI's machine GPU reading (snapshot.gpu). Every field is
// optional: nil (JSON null) means the source returned nothing or its call
// failed. Nothing here is a CLI key; the CLI schema does not carry it.
//
//	use_pct             busiest engine type of the chosen adapter, 0..100
//	memory_used_bytes   dedicated GPU memory in use
//	memory_total_bytes  dedicated GPU memory as the OS reports it (the carve)
//	shared_used_bytes   shared (system) memory used by the adapter
//	temperature_c       GPU temperature, degrees Celsius
//	chip_power_w        whole-chip power: CPU, GPU and NPU together
type GPUStats struct {
	SampleTime       string   `json:"sample_time,omitempty"`
	WindowSeconds    *float64 `json:"window_seconds,omitempty"`
	UsePct           *float64 `json:"use_pct"`
	MemoryUsedBytes  *uint64  `json:"memory_used_bytes"`
	MemoryTotalBytes *uint64  `json:"memory_total_bytes"`
	SharedUsedBytes  *uint64  `json:"shared_used_bytes"`
	TemperatureC     *float64 `json:"temperature_c"`
	ChipPowerW       *float64 `json:"chip_power_w"`
}

// Empty reports whether no field carries a value.
func (g *GPUStats) Empty() bool {
	return g == nil || g.UsePct == nil && g.MemoryUsedBytes == nil && g.MemoryTotalBytes == nil &&
		g.SharedUsedBytes == nil && g.TemperatureC == nil && g.ChipPowerW == nil
}

// GPUSource reads the GPU stats. Sample returns nil when nothing is available
// and never panics. Close releases the OS handles.
type GPUSource interface {
	Sample() *GPUStats
	Close()
}

// pdhItem is one instance of a wildcard PDH counter with a valid value.
type pdhItem struct {
	Name  string
	Value float64
}

// gpuRaw is what the OS calls returned in one poll, before conversion.
type gpuRaw struct {
	TotalBytes   uint64 // GETSEGMENTSIZE dedicated; 0 = unknown
	TempDeci     uint32 // ADAPTERPERFDATA Temperature, deci-Celsius; 0 = unknown
	Engines      []pdhItem
	Dedicated    []pdhItem
	Shared       []pdhItem
	PowerMW      float64
	HavePowerRaw bool
	LUIDPrefix   string // "" = no adapter was chosen
}

// luidPrefix is the lowercase PDH spelling of an adapter LUID, ended by "_"
// so that one LUID never matches the start of another.
func luidPrefix(high uint32, low uint32) string {
	return fmt.Sprintf("luid_0x%08x_0x%08x_", high, low)
}

// instanceHasLUID reports whether a PDH instance name belongs to the adapter.
// Engine instances read "pid_<n>_luid_0x..._0x..._phys_0_eng_0_engtype_3D";
// adapter memory instances read "luid_0x..._0x..._phys_0".
func instanceHasLUID(name, prefix string) bool {
	if prefix == "" {
		return false
	}
	return strings.Contains(strings.ToLower(name), prefix)
}

// engineType returns the part after "_engtype_" ("3D", "Compute_0"), or "".
func engineType(name string) string {
	const key = "_engtype_"
	i := strings.LastIndex(strings.ToLower(name), key)
	if i < 0 {
		return ""
	}
	return name[i+len(key):]
}

// busiestEngine sums the utilisation of the adapter's instances per engine
// type and returns the largest sum, capped at 100. ok is false when no
// instance of the adapter carried a value.
func busiestEngine(items []pdhItem, prefix string) (float64, bool) {
	sums := map[string]float64{}
	for _, it := range items {
		if !instanceHasLUID(it.Name, prefix) || !finiteNonNeg(it.Value) {
			continue
		}
		sums[strings.ToLower(engineType(it.Name))] += it.Value
	}
	if len(sums) == 0 {
		return 0, false
	}
	best := 0.0
	for _, v := range sums {
		best = math.Max(best, v)
	}
	return math.Min(best, 100), true
}

// sumForLUID adds the values of the adapter's instances.
func sumForLUID(items []pdhItem, prefix string) (float64, bool) {
	total, n := 0.0, 0
	for _, it := range items {
		if instanceHasLUID(it.Name, prefix) && finiteNonNeg(it.Value) {
			total += it.Value
			n++
		}
	}
	return total, n > 0
}

func finiteNonNeg(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) && f >= 0 }

func bytesPtr(f float64) *uint64 {
	if !finiteNonNeg(f) || f >= math.MaxUint64/2 {
		return nil
	}
	v := uint64(f + 0.5)
	return &v
}

func roundTenth(f float64) *float64 {
	v := math.Round(f*10) / 10
	return &v
}

// deciCelsius converts D3DKMT deci-degrees to degrees. Zero is the value of a
// failed or empty call, and a reading above 150 C is not a GPU temperature.
func deciCelsius(d uint32) *float64 {
	if d == 0 || d > 1500 {
		return nil
	}
	return roundTenth(float64(d) / 10)
}

// milliwattsToWatts converts the Energy Meter counter (milliwatts).
func milliwattsToWatts(mw float64) *float64 {
	if !finiteNonNeg(mw) || mw == 0 {
		return nil
	}
	return roundTenth(mw / 1000)
}

// buildGPUStats converts one poll. It returns nil when nothing is available.
func buildGPUStats(r gpuRaw) *GPUStats {
	g := &GPUStats{}
	if v, ok := busiestEngine(r.Engines, r.LUIDPrefix); ok {
		g.UsePct = roundTenth(v)
	}
	if v, ok := sumForLUID(r.Dedicated, r.LUIDPrefix); ok {
		g.MemoryUsedBytes = bytesPtr(v)
	}
	if v, ok := sumForLUID(r.Shared, r.LUIDPrefix); ok {
		g.SharedUsedBytes = bytesPtr(v)
	}
	if r.TotalBytes > 0 {
		t := r.TotalBytes
		g.MemoryTotalBytes = &t
	}
	g.TemperatureC = deciCelsius(r.TempDeci)
	if r.HavePowerRaw {
		g.ChipPowerW = milliwattsToWatts(r.PowerMW)
	}
	if g.Empty() {
		return nil
	}
	return g
}
