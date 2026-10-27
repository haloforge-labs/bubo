#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "gufo" / "h3_profile.py"
SPEC = importlib.util.spec_from_file_location("h3_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


schedule = MODULE.balanced_schedule(["exact"], 1)
check(schedule == ["exact"], "retention benchmark schedules one exact video")
check(
    MODULE.DEFAULT_BENCHMARK_PRESETS == ("exact",),
    "default benchmark preset is exact only",
)

parsed = MODULE.parse_key_value_lines(
    "VmRSS:\t123 kB\nVmSwap:\t4 kB\nread_bytes: 99\nignored\n"
)
check(parsed == {"VmRSS": 123, "VmSwap": 4, "read_bytes": 99}, "proc parser")

descriptor = MODULE.public_descriptor(
    "fast", 2, "warm", MODULE.hashlib.sha256(b"secret prompt").hexdigest(), 42
)
serialized = json.dumps(descriptor)
check("secret prompt" not in serialized, "public descriptor excludes prompt")
check("/private/model" not in serialized, "public descriptor excludes model path")

samples = [
    MODULE.MonitorSample(
        elapsed_ms=0,
        rss_kib=10,
        rss_anon_kib=6,
        rss_file_kib=4,
        rss_shmem_kib=0,
        hwm_kib=12,
        swap_kib=0,
        read_bytes=100,
        write_bytes=20,
        gpu_temp_millic=45000,
        gpu_power_microwatts=1000000,
        gpu_clock_mhz=900,
    ),
    MODULE.MonitorSample(
        elapsed_ms=1000,
        rss_kib=20,
        rss_anon_kib=11,
        rss_file_kib=8,
        rss_shmem_kib=1,
        hwm_kib=22,
        swap_kib=1,
        read_bytes=200,
        write_bytes=40,
        gpu_temp_millic=50000,
        gpu_power_microwatts=2000000,
        gpu_clock_mhz=1200,
    ),
]
summary = MODULE.monitor_summary(samples, 1.0)
check(summary["peak_rss_kib"] == 20, "peak RSS")
check(summary["peak_rss_anon_kib"] == 11, "peak anonymous RSS")
check(summary["peak_rss_file_kib"] == 8, "peak file-backed RSS")
check(summary["peak_rss_shmem_kib"] == 1, "peak shared-memory RSS")
check(summary["maximum_gpu_temp_millic"] == 50000, "peak temperature")
check(summary["read_bytes"] == 100, "process I/O delta")
encoded_samples = [MODULE.asdict(sample) for sample in samples]
check(encoded_samples[1]["gpu_clock_mhz"] == 1200, "raw sample retention")

delta = MODULE.rusage_delta(
    {"user_seconds": 1.0, "maximum_rss_kib": 20, "minor_page_faults": 3},
    {"user_seconds": 2.5, "maximum_rss_kib": 30, "minor_page_faults": 8},
)
check(delta["user_seconds"] == 1.5, "child CPU delta")
check(delta["maximum_rss_kib"] == 30, "child peak RSS is absolute")
check(delta["minor_page_faults"] == 5, "child fault delta")

with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / "report.json"
    MODULE.atomic_json(target, descriptor)
    check(json.loads(target.read_text(encoding="utf-8")) == descriptor, "atomic JSON")

completed = MODULE.subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
completed.wait()
MODULE.stop_process_group(completed)

aggregate = MODULE.aggregate(
    [
        {
            "preset": "fast",
            "cache_label": "first-observation",
            "return_code": 0,
            "wall_ms": 120.0,
            "telemetry": {
                "total_ms": 100.0,
                "inventory_ms": 5.0,
                "denoiser_ms": 70.0,
                "video_vae_ms": 20.0,
            },
        },
        {
            "preset": "fast",
            "cache_label": "warm-observation",
            "return_code": 0,
            "wall_ms": 100.0,
            "telemetry": {
                "total_ms": 90.0,
                "inventory_ms": 2.0,
                "denoiser_ms": 60.0,
                "video_vae_ms": 20.0,
            },
        },
    ]
)
fast = aggregate["presets"]["fast"]
check(fast["generation_ms_median"] == 95.0, "generation median")
check(fast["bottlenecks"][0]["phase"] == "denoiser", "bottleneck ranking")
check(len(fast["cache_states"]) == 2, "cache-state split")

print("MiniMax H3 profiling harness tests passed.")
