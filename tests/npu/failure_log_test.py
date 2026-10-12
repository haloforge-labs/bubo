import subprocess
import sys

result = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=120)
print(result.stdout, end="")
print(result.stderr, end="")
assert result.returncode == 0, result.returncode
mode = sys.argv[2]
assert result.stderr.count("[ERROR] [npu]") == (mode in ("error", "timeout", "stream_timeout", "invalid"))
