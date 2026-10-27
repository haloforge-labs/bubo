//go:build windows

package supervisor

import (
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Windows GPU source: gdi32 D3DKMT (adapter choice, memory totals, temperature)
// and PDH (use, memory in use, chip power). No cgo, no vendor DLL, no admin.
// D3DKMT is documented in the WDK but is not a stable SDK contract: any
// non-zero NTSTATUS makes that reading unavailable.

var (
	gdi32 = windows.NewLazySystemDLL("gdi32.dll")
	pdh   = windows.NewLazySystemDLL("pdh.dll")

	pEnumAdapters2 = gdi32.NewProc("D3DKMTEnumAdapters2")
	pOpenLuid      = gdi32.NewProc("D3DKMTOpenAdapterFromLuid")
	pQueryInfo     = gdi32.NewProc("D3DKMTQueryAdapterInfo")
	pCloseAdapter  = gdi32.NewProc("D3DKMTCloseAdapter")

	pPdhOpenQuery   = pdh.NewProc("PdhOpenQueryW")
	pPdhAddCounter  = pdh.NewProc("PdhAddEnglishCounterW")
	pPdhCollect     = pdh.NewProc("PdhCollectQueryData")
	pPdhFormatArray = pdh.NewProc("PdhGetFormattedCounterArrayW")
	pPdhFormatValue = pdh.NewProc("PdhGetFormattedCounterValue")
	pPdhCloseQuery  = pdh.NewProc("PdhCloseQuery")
)

const (
	kmtTypeSegmentSize = 3
	kmtTypePerfData    = 62
	pdhFmtDouble       = 0x200
	pdhMoreData        = 0x800007D2
)

type kmtAdapterInfo struct {
	Handle        uint32
	LUID          windows.LUID
	NumOfSources  uint32
	PresentMoveOK uint32
}
type kmtEnumAdapters2 struct {
	Count    uint32
	Adapters *kmtAdapterInfo
}
type kmtOpenLuid struct {
	LUID   windows.LUID
	Handle uint32
}
type kmtQueryInfo struct {
	Handle uint32
	Type   uint32
	Data   unsafe.Pointer
	Size   uint32
}
type kmtSegmentSize struct{ Dedicated, DedicatedSystem, Shared uint64 }
type kmtPerfData struct {
	PhysicalAdapterIndex uint32
	MemoryFrequency      uint64
	MaxMemoryFrequency   uint64
	MaxMemoryFrequencyOC uint64
	MemoryBandwidth      uint64
	PCIEBandwidth        uint64
	FanRPM               uint32
	Power                uint32
	Temperature          uint32
	PowerStateOverride   uint8
}
type kmtClose struct{ Handle uint32 }

// pdhValue is PDH_FMT_COUNTERVALUE with a double.
type pdhValue struct {
	Status uint32
	_      uint32
	Double float64
}

// pdhItemW is PDH_FMT_COUNTERVALUE_ITEM_W with a double.
type pdhItemW struct {
	Name   *uint16
	Status uint32
	_      uint32
	Double float64
}

type adapterPick struct {
	LUID     windows.LUID
	Total    uint64
	TempDeci uint32
	OK       bool
}

// pickAdapter chooses the adapter whose dedicated memory is above zero. It
// never goes by index: LUIDs change after a reboot or a driver reset.
func pickAdapter() (pick adapterPick) {
	defer func() { _ = recover() }()
	if pEnumAdapters2.Find() != nil || pOpenLuid.Find() != nil || pQueryInfo.Find() != nil || pCloseAdapter.Find() != nil {
		return
	}
	var e kmtEnumAdapters2
	if st, _, _ := pEnumAdapters2.Call(uintptr(unsafe.Pointer(&e))); st != 0 || e.Count == 0 {
		return
	}
	buf := make([]kmtAdapterInfo, e.Count+1)
	e.Adapters = &buf[0]
	if st, _, _ := pEnumAdapters2.Call(uintptr(unsafe.Pointer(&e))); st != 0 {
		return
	}
	n := int(e.Count)
	if n > len(buf) {
		n = len(buf)
	}
	for i := 0; i < n; i++ {
		o := kmtOpenLuid{LUID: buf[i].LUID}
		if st, _, _ := pOpenLuid.Call(uintptr(unsafe.Pointer(&o))); st != 0 {
			continue
		}
		var seg kmtSegmentSize
		q := kmtQueryInfo{Handle: o.Handle, Type: kmtTypeSegmentSize, Data: unsafe.Pointer(&seg), Size: uint32(unsafe.Sizeof(seg))}
		st, _, _ := pQueryInfo.Call(uintptr(unsafe.Pointer(&q)))
		if st == 0 && seg.Dedicated > 0 {
			pick = adapterPick{LUID: buf[i].LUID, Total: seg.Dedicated, OK: true}
			var pd kmtPerfData
			q = kmtQueryInfo{Handle: o.Handle, Type: kmtTypePerfData, Data: unsafe.Pointer(&pd), Size: uint32(unsafe.Sizeof(pd))}
			if st, _, _ := pQueryInfo.Call(uintptr(unsafe.Pointer(&q))); st == 0 {
				pick.TempDeci = pd.Temperature
			}
		}
		c := kmtClose{Handle: o.Handle}
		pCloseAdapter.Call(uintptr(unsafe.Pointer(&c)))
		if pick.OK {
			return
		}
	}
	return
}

type winGPU struct {
	collectedAt time.Time
	mu          sync.Mutex
	query       uintptr
	engine      uintptr
	ded         uintptr
	shared      uintptr
	power       uintptr
	closed      bool
}

// NewGPUSource opens the PDH query once; it stays open between polls. One
// collect at creation primes the rate counters.
func NewGPUSource() GPUSource {
	g := &winGPU{}
	defer func() { _ = recover() }()
	if pPdhOpenQuery.Find() != nil {
		return g
	}
	if r, _, _ := pPdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&g.query))); r != 0 {
		g.query = 0
		return g
	}
	g.addCounters()
	if g.query != 0 {
		if r, _, _ := pPdhCollect.Call(g.query); r == 0 {
			g.collectedAt = time.Now()
		}
	}
	return g
}

func (g *winGPU) add(path string) uintptr {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0
	}
	var h uintptr
	if r, _, _ := pPdhAddCounter.Call(g.query, uintptr(unsafe.Pointer(p)), 0, uintptr(unsafe.Pointer(&h))); r != 0 {
		return 0
	}
	return h
}

// addCounters adds the counters that are not added yet (a failed add is tried
// again at the next poll).
func (g *winGPU) addCounters() {
	if g.query == 0 {
		return
	}
	if g.engine == 0 {
		g.engine = g.add(`\GPU Engine(*)\Utilization Percentage`)
	}
	if g.ded == 0 {
		g.ded = g.add(`\GPU Adapter Memory(*)\Dedicated Usage`)
	}
	if g.shared == 0 {
		g.shared = g.add(`\GPU Adapter Memory(*)\Shared Usage`)
	}
	if g.power == 0 {
		g.power = g.add(`\Energy Meter(rapl_package0_pkg)\Power`)
	}
}

func readArray(h uintptr) []pdhItem {
	if h == 0 {
		return nil
	}
	var size, count uint32
	r, _, _ := pPdhFormatArray.Call(h, pdhFmtDouble, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), 0)
	if uint32(r) != pdhMoreData || size == 0 || size > 64<<20 {
		return nil
	}
	buf := make([]uint64, (size+7)/8+1)
	r, _, _ = pPdhFormatArray.Call(h, pdhFmtDouble, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0])))
	if r != 0 {
		return nil
	}
	items := unsafe.Slice((*pdhItemW)(unsafe.Pointer(&buf[0])), int(count))
	out := make([]pdhItem, 0, len(items))
	for _, it := range items {
		if it.Status != 0 && it.Status != 1 { // valid data, new data
			continue
		}
		out = append(out, pdhItem{Name: windows.UTF16PtrToString(it.Name), Value: it.Double})
	}
	return out
}

func readValue(h uintptr) (float64, bool) {
	if h == 0 {
		return 0, false
	}
	var v pdhValue
	if r, _, _ := pPdhFormatValue.Call(h, pdhFmtDouble, 0, uintptr(unsafe.Pointer(&v))); r != 0 {
		return 0, false
	}
	if v.Status != 0 && v.Status != 1 {
		return 0, false
	}
	return v.Double, true
}

// Sample reads every source once. It returns nil when nothing came back.
func (g *winGPU) Sample() (out *GPUStats) {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer func() {
		if recover() != nil {
			out = nil
		}
	}()
	if g.closed {
		return nil
	}
	pick := pickAdapter()
	var raw gpuRaw
	var window *float64
	if pick.OK {
		raw.TotalBytes, raw.TempDeci = pick.Total, pick.TempDeci
		raw.LUIDPrefix = luidPrefix(uint32(pick.LUID.HighPart), pick.LUID.LowPart)
	}
	if g.query != 0 {
		g.addCounters()
		if r, _, _ := pPdhCollect.Call(g.query); r == 0 {
			now := time.Now()
			if !g.collectedAt.IsZero() {
				seconds := now.Sub(g.collectedAt).Seconds()
				window = &seconds
			}
			g.collectedAt = now
			raw.Engines, raw.Dedicated, raw.Shared = readArray(g.engine), readArray(g.ded), readArray(g.shared)
			raw.PowerMW, raw.HavePowerRaw = readValue(g.power)
		}
	}
	out = buildGPUStats(raw)
	if out != nil {
		out.SampleTime = time.Now().UTC().Format(time.RFC3339Nano)
		out.WindowSeconds = window
	}
	return out
}

func (g *winGPU) Close() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.closed && g.query != 0 {
		pPdhCloseQuery.Call(g.query)
	}
	g.closed = true
}
