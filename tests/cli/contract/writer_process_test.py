import json
import os
from pathlib import Path
import subprocess
import sys

executable=Path(sys.argv[1]).resolve()
flags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0
native=subprocess.run([str(executable),'--native'],capture_output=True,timeout=10,creationflags=flags)
assert native.returncode==0
lines=native.stdout.splitlines()
assert len(lines)==1 and json.loads(lines[0])['type']=='capabilities'
assert b'noise' not in native.stdout and b'stream noise' in native.stderr and b'library noise' in native.stderr
assert b'\r' not in native.stdout and not native.stdout.startswith(b'\xef\xbb\xbf')
passed=1
if os.name=='nt':
    child=subprocess.Popen([str(executable),'--blocked'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=flags)
    try:
        assert child.wait(timeout=6)==0
        passed+=1
    finally:
        if child.poll() is None:
            child.kill()  # Only the exact child this CPU test started.
            child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()
print(f'Native stdout isolation and blocked consumer: {passed} passed, 0 failed')
