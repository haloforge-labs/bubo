"""Exercise real OS pipes. Kill only exact processes started by this test."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

exe = str(Path(sys.argv[1]).resolve())
flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
passed = 0
for name, count in [('stdout_consumer_closes', 0), ('stdout_truncated_mid_record', 4096)]:
    child = subprocess.Popen([exe, '--closed'], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        if count:
            assert len(child.stdout.read(count)) == count
        child.stdout.close()
        assert child.wait(timeout=6) == 0
        passed += 1
        print(f'{name}: 1 passed, 0 failed')
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
if os.name == 'nt':
    with tempfile.TemporaryFile() as errors:
        child = subprocess.Popen([exe, '--child'], stdout=subprocess.PIPE,
                                 stderr=errors, creationflags=flags)
        holder = None
        try:
            assert child.wait(timeout=3) == 0
            errors.seek(0)
            holder = int(errors.read().split(b'holder_pid=')[1].splitlines()[0])
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x00100000, False, holder)
            assert handle and kernel.WaitForSingleObject(handle, 0) == 258, 'holder must still be alive'
            with concurrent.futures.ThreadPoolExecutor() as pool:
                read = pool.submit(child.stdout.read)
                # The child is alive, but cannot keep the wire pipe open.
                try:
                    data = read.result(timeout=1)
                except concurrent.futures.TimeoutError:
                    kernel.WaitForSingleObject(handle, 6000)
                    raise AssertionError('child inherited machine stdout')
            assert json.loads(data)['type'] == 'capabilities'
            kernel.WaitForSingleObject(handle, 6000)
            kernel.CloseHandle(handle)
            passed += 1
            print('child_holds_inherited_stdout: 1 passed, 0 failed')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()
print(f'Writer fault injection: {passed} passed, 0 failed')
if len(sys.argv) > 2:
    prepare = str(Path(sys.argv[2]).resolve())
    for name, count in [('prepare_stdout_closes_before_output',0),
                        ('prepare_stdout_closes_while_waiting_approval',-1)]:
        child = subprocess.Popen([prepare, 'verify'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 creationflags=flags)
        try:
            if count == -1:
                while json.loads(child.stdout.readline())['type'] != 'approval_required':
                    pass
            elif count:
                assert len(child.stdout.read(count)) == count
            child.stdout.close()
            # Keep stdin open. Stdout closure alone must end preparation.
            assert child.wait(timeout=6) == 3
            print(f'{name}: 1 passed, 0 failed')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdin.close()
    print('Prepare native stdout closure: 2 passed, 0 failed')
