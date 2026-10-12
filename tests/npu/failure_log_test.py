import subprocess
import sys

result = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=120)
print(result.stdout, end="")
print(result.stderr, end="")
assert result.returncode == 0, result.returncode
mode = sys.argv[2]
failure = mode.startswith("startup_") or mode in ("error", "timeout", "stream_timeout", "invalid")
assert result.stderr.count("[WARN] [npu]") == int(failure), result.stderr
if mode.startswith("startup_"):
    lines = [line for line in result.stderr.splitlines() if "[npu]" in line]
    assert len(lines) == 1 and "NPU disabled for process; GPU path" in lines[0], lines
if mode in ("denied", "power_unavailable"):
    assert "[npu]" not in result.stderr, result.stderr
if mode == "power_turbo":
    assert result.stderr.count("[INFO] [npu]") == 1 and "Turbo" in result.stderr, result.stderr
