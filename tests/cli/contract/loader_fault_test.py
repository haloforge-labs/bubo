"""Run copied bootstrap bytes with missing CRT imports. Do not alter system DLLs."""
import ctypes
import json
import os
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import uuid

assert os.name == 'nt'
exe = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()/uuid.uuid4().hex
assert not root.exists(), 'use a fresh test-owned root'
root.mkdir(parents=True)
# Keep the normal bootstrap dependency closure present. The only injected
# failure is the renamed CRT import, not a missing curl/crypto/z dependency.
for dependency in exe.parent.glob('*.dll'):
    shutil.copy2(dependency, root/dependency.name)
data = bytearray(exe.read_bytes())
pe = struct.unpack_from('<I', data, 0x3c)[0]
assert data[pe:pe+4] == b'PE\0\0'
sections, optional_size = struct.unpack_from('<H12xH', data, pe+6)
optional = pe+24
assert struct.unpack_from('<H', data, optional)[0] == 0x20b, 'expected PE32+'
import_rva = struct.unpack_from('<I', data, optional+120)[0]
section_table = optional+optional_size
def raw(rva):
    for i in range(sections):
        size, start, raw_size, offset = struct.unpack_from('<IIII', data, section_table+40*i+8)
        if start <= rva < start+max(size, raw_size):
            return offset+rva-start
    raise AssertionError(f'RVA {rva} has no section')
entry = raw(import_rva)
changed = []
while any(data[entry:entry+20]):
    name_rva = struct.unpack_from('<I', data, entry+12)[0]
    offset = raw(name_rva)
    end = data.index(0, offset)
    name = bytes(data[offset:end])
    if name.lower().startswith((b'vcruntime', b'msvcp')):
        replacement = b'z' * (len(name)-4) + b'.dll'
        data[offset:end] = replacement
        changed.append(name.decode('ascii'))
    entry += 20
assert changed, 'candidate has no dynamic VC++ import; missing-CRT gate needs another candidate'
candidate = root/'missing-crt.exe'
candidate.write_bytes(data)
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.SetErrorMode.argtypes = [ctypes.c_uint32]
kernel.SetErrorMode.restype = ctypes.c_uint32
old = kernel.SetErrorMode(0x0001 | 0x0002 | 0x8000)
try:
    result = subprocess.run([str(candidate), 'status', '--json'], cwd=root,
                            capture_output=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
finally:
    kernel.SetErrorMode(old)
record = dict(imports=changed, raw_exit=f'0x{result.returncode & 0xffffffff:08x}',
              stdout_hex=result.stdout.hex(), stderr_hex=result.stderr.hex())
(root/'loader-result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
assert result.returncode & 0xffffffff == 0xc0000135, record
assert result.stdout == b'', record
print(json.dumps(record))
print('missing_crt_before_main_raw_loader_exit: 1 passed, 0 failed')
