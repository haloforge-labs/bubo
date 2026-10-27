import subprocess
import sys

result = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=120)
print(result.stdout, end="")
print(result.stderr, end="")
assert result.returncode == 0, result.returncode
mode = sys.argv[2]
failure = mode.startswith("startup_") or mode in ("error", "timeout", "stream_timeout", "invalid")
# hal_event: HAL fault line + "HAL fault event during an NPU chunk" line (NPU stays on).
# invalid_hal: HAL fault line + the one process failure line.
expected = 2 if mode in ("hal_event", "invalid_hal") else int(failure)
assert result.stderr.count("[WARN] [npu]") == expected, result.stderr
if mode in ("invalid", "invalid_hal"):
    line = [l for l in result.stderr.splitlines() if "invalid activations or packed output" in l]
    assert len(line) == 1, line
    for part in ("cause=0x4(C-exp255)", "chunk=", "tokens=4096", "start_utc=", "end_utc=",
                 "start_local=", "end_local=", "idle_gap_ms=", "avail_mib_start=", "commit_mib_end=",
                 "linear_begins=", "finish_calls=", "hostfunc_lat_ms=", "hal_faults_total=", "hal_in_chunk=",
                 "NPU disabled for process; GPU path"):
        assert part in line[0], (part, line[0])
    if mode == "invalid_hal":
        assert "hal_in_chunk=1" in line[0], line[0]
if mode in ("hal_event", "invalid_hal"):
    assert "HAL-fault event: 2026-10-13T22:01:41.123Z id=15 reason=0x6 info=0x2E05B000 device=0xC501" in result.stderr, result.stderr
if mode == "hal_event":
    assert "HAL fault event during an NPU chunk with a valid result" in result.stderr, result.stderr
if mode.startswith("startup_"):
    lines = [line for line in result.stderr.splitlines() if "[npu]" in line]
    assert len(lines) == 1 and "NPU disabled for process; GPU path" in lines[0], lines
if mode in ("denied", "power_unavailable"):
    assert "[npu]" not in result.stderr, result.stderr
if mode == "power_turbo":
    assert result.stderr.count("[INFO] [npu]") == 1 and "Turbo" in result.stderr, result.stderr
