"""Fault injection through real native stdin pipes, with bounded reads."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

exe = str(Path(sys.argv[1]).resolve())
flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
passed = 0
cases = ['partial_json_eof', 'partial_json_split', 'garbage', 'wrong_operation',
         'stale_digest', 'double_approve', 'cancel_before_approval', 'eof_approval',
         'eof_download', 'eof_verify', 'eof_promote', 'cancel_verify', 'garbage_verify',
         'oversized_input', 'unknown_input_field', 'cancel_preflight_verify',
         'eof_preflight_verify', 'eof_copy', 'eof_extract', 'eof_before_preflight',
         'cancel_invalid_digest', 'noop_eof', 'input_64k_boundary']
for name in cases:
    target = name.removeprefix('eof_') if name.startswith('eof_') else 'verify'
    if name in ('cancel_preflight_verify','eof_preflight_verify'):
        target = 'preflight'
    elif name == 'noop_eof':
        target = 'noop'
    child = subprocess.Popen([exe, target], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, creationflags=flags)
    records = queue.Queue()
    data = bytearray()
    errors = bytearray()
    def drain():
        try:
            while line := child.stdout.readline():
                data.extend(line)
                records.put(json.loads(line.decode('utf-8')))
        except Exception as exc:
            records.put(exc)
        finally:
            records.put(None)
    reader = threading.Thread(target=drain)
    err_reader = threading.Thread(target=lambda: errors.extend(child.stderr.read()))
    reader.start()
    err_reader.start()
    rows = []
    def until(kind, phase=None):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = records.get(timeout=max(0.01, deadline-time.monotonic()))
            assert isinstance(row, dict), (name, row, bytes(errors))
            rows.append(row)
            if row['type'] == kind and (phase is None or row['payload']['phase'] == phase):
                return row
        raise AssertionError(f'{name}: missing {kind}/{phase}')
    def send(raw):
        child.stdin.write(raw)
        child.stdin.flush()
    try:
        if name == 'noop_eof':
            child.stdin.close()
            until('result')
            assert child.wait(timeout=6) == 0
            passed += 1
            print(f'native_stdin_{name}: 1 passed, 0 failed')
            continue
        if name in ('cancel_preflight_verify','eof_preflight_verify'):
            event = until('progress', 'verify')
            if name.startswith('eof'):
                child.stdin.close()
            else:
                send((json.dumps(dict(action='cancel', operation_id=event['operation_id']))+'\n').encode())
            assert child.wait(timeout=6) == 3
            reader.join(timeout=2)
            err_reader.join(timeout=2)
            rows = [json.loads(line) for line in data.decode('utf-8').splitlines()]
            assert rows[-1]['type'] == 'error' and rows[-1]['payload']['code'] == 'cancelled'
            assert not any(row['type']=='approval_required' for row in rows)
            passed += 1
            print(f'native_stdin_{name}: 1 passed, 0 failed')
            continue
        if name == 'eof_before_preflight':
            child.stdin.close()
        request = until('approval_required')
        approval = dict(action='approve', operation_id=request['operation_id'],
                        plan_digest=request['payload']['plan_digest'])
        expected = 3
        if name in ('eof_approval','eof_before_preflight'):
            if not child.stdin.closed:
                child.stdin.close()
        elif name == 'partial_json_eof':
            send(b'{"action":')
            child.stdin.close()
            expected = 2
        elif name == 'garbage':
            send(b'garbage\n')
            expected = 2
        elif name == 'oversized_input':
            raw = json.dumps(approval).encode('utf-8')
            # Valid bound approval, padded with legal JSON whitespace. This
            # must fail because of its length, not because of JSON syntax.
            send(raw + b' ' * (65537-len(raw)) + b'\n')
            expected = 2
        else:
            if name == 'wrong_operation':
                approval['operation_id'] = '0'*32
                expected = 2
            elif name == 'stale_digest':
                approval['plan_digest'] = '0'*64
                expected = 2
            elif name == 'unknown_input_field':
                approval['unexpected'] = True
                expected = 2
            elif name == 'cancel_before_approval':
                approval = dict(action='cancel', operation_id=request['operation_id'])
            elif name == 'cancel_invalid_digest':
                approval = dict(action='cancel', operation_id=request['operation_id'],plan_digest='invalid')
                expected = 2
            raw = (json.dumps(approval) + '\n').encode('utf-8')
            if name == 'partial_json_split':
                send(raw[:17])
                time.sleep(0.1)
                assert child.poll() is None, 'partial input must wait for LF'
                send(raw[17:])
                until('progress', 'verify')
                child.stdin.close()
            elif name == 'double_approve':
                send(raw + raw)
                expected = 2
            elif name == 'input_64k_boundary':
                send(raw[:-1] + b' ' * (65536-(len(raw)-1)) + b'\n')
                until('progress', 'verify')
                child.stdin.close()
            else:
                send(raw)
                if name in ('cancel_verify', 'garbage_verify'):
                    until('progress', 'verify')
                    if name == 'cancel_verify':
                        send((json.dumps(dict(action='cancel', operation_id=request['operation_id']))+'\n').encode())
                    else:
                        send(b'{broken\n')
                        expected = 2
                elif name in ('eof_download', 'eof_copy', 'eof_verify', 'eof_extract', 'eof_promote'):
                    until('progress', target)
                    child.stdin.close()
        assert child.wait(timeout=6) == expected, (name, child.returncode, bytes(errors))
        reader.join(timeout=2)
        err_reader.join(timeout=2)
        assert not reader.is_alive() and not err_reader.is_alive()
        rows = [json.loads(line) for line in data.decode('utf-8').splitlines()]
        assert rows[-1]['type'] == 'error'
        assert rows[-1]['payload']['code'] == ('invalid_arguments' if expected == 2 else 'cancelled')
        assert b'\r' not in data and not data.startswith(b'\xef\xbb\xbf')
        assert [row['seq'] for row in rows] == list(range(1, len(rows)+1))
        passed += 1
        print(f'native_stdin_{name}: 1 passed, 0 failed')
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if not child.stdin.closed:
            child.stdin.close()
        reader.join(timeout=2)
        err_reader.join(timeout=2)
        child.stdout.close()
        child.stderr.close()
print(f'Native stdin faults: {passed} passed, 0 failed')
