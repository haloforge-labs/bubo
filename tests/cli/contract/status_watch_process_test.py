"""Real bubo status --watch, replayed /slots, pipe and Windows ConPTY proof.

No model, GPU, download or production HTTP port. The bridge uses production
process identity checks and private control pipes. Every child is owned here.
Artifacts retain raw terminal bytes, plain lines, replay facts and verdicts.
"""
from __future__ import annotations

import argparse
import ctypes as c
from ctypes import wintypes as w
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid


def terminal_screen(data, width=120, height=30):
    """Read ConPTY's screen updates, including its optimized cursor moves.

    ConPTY may re-encode a program's erase/up instructions as cursor-position
    instructions. Judge the visible rows, not one chosen ANSI spelling.
    """
    rows = [[' '] * width for _ in range(height)]
    row = column = 0
    decoded = data.decode('utf-8', errors='replace')
    decoded = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', decoded)
    for token in re.findall(r'\x1b\[[0-?]*[ -/]*[@-~]|[^\x1b]', decoded):
        if token.startswith('\x1b['):
            final, fields = token[-1], token[2:-1]
            if fields.startswith('?'):
                continue
            parts = [int(value or 0) for value in fields.split(';')]
            n = parts[0] or 1
            if final in 'Hf':
                row = min(height - 1, max(0, n - 1))
                column = min(width - 1, max(0, (parts[1] or 1) - 1)) if len(parts) > 1 else 0
            elif final == 'A':
                row = max(0, row - n)
            elif final == 'B':
                row = min(height - 1, row + n)
            elif final == 'C':
                column = min(width - 1, column + n)
            elif final == 'D':
                column = max(0, column - n)
            elif final == 'G':
                column = min(width - 1, n - 1)
            elif final == 'K':
                start, end = (0, width) if parts[0] == 2 else (0, column + 1) if parts[0] == 1 else (column, width)
                rows[row][start:end] = [' '] * (end - start)
            elif final == 'J':
                if parts[0] in (2, 3):
                    rows = [[' '] * width for _ in range(height)]
                else:
                    rows[row][column:] = [' '] * (width - column)
                    for index in range(row + 1, height):
                        rows[index] = [' '] * width
            continue
        if token == '\r':
            column = 0
        elif token == '\n':
            row += 1
        elif token == '\b':
            column = max(0, column - 1)
        elif token >= ' ':
            if column == width:
                column, row = 0, row + 1
            if row == height:
                rows.pop(0)
                rows.append([' '] * width)
                row -= 1
            rows[row][column] = token
            column += 1
        if row == height:
            rows.pop(0)
            rows.append([' '] * width)
            row -= 1
    return '\n'.join(''.join(line).rstrip() for line in rows).rstrip()


class Console:
    """Small Windows pseudoconsole wrapper; no dependency installation."""
    def __init__(self, args, width=120):
        k = self.k = c.WinDLL('kernel32', use_last_error=True)
        handle = w.HANDLE
        k.CreatePipe.argtypes = [c.POINTER(handle), c.POINTER(handle), c.c_void_p, w.DWORD]
        k.CloseHandle.argtypes = [handle]
        k.CreatePseudoConsole.argtypes = [w.DWORD, handle, handle, w.DWORD, c.POINTER(handle)]
        k.CreatePseudoConsole.restype = c.c_long
        k.ClosePseudoConsole.argtypes = [handle]
        k.InitializeProcThreadAttributeList.argtypes = [c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)]
        k.UpdateProcThreadAttribute.argtypes = [c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p]
        k.DeleteProcThreadAttributeList.argtypes = [c.c_void_p]
        k.ReadFile.argtypes = [handle, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p]
        k.WriteFile.argtypes = k.ReadFile.argtypes
        k.WaitForSingleObject.argtypes = [handle, w.DWORD]
        k.TerminateProcess.argtypes = [handle, w.UINT]
        k.GetExitCodeProcess.argtypes = [handle, c.POINTER(w.DWORD)]
        class SI(c.Structure):
            _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR),
                        ('title', w.LPWSTR), ('x', w.DWORD), ('y', w.DWORD),
                        ('xs', w.DWORD), ('ys', w.DWORD), ('xc', w.DWORD),
                        ('yc', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                        ('show', w.WORD), ('reserved2size', w.WORD),
                        ('reserved2', c.c_void_p), ('input', handle),
                        ('output', handle), ('error', handle)]
        class SIX(c.Structure):
            _fields_ = [('si', SI), ('attrs', c.c_void_p)]
        class PI(c.Structure):
            _fields_ = [('process', handle), ('thread', handle), ('pid', w.DWORD), ('tid', w.DWORD)]
        k.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                                     w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR,
                                     c.POINTER(SIX), c.POINTER(PI)]
        inp, self.input, self.output, out = handle(), handle(), handle(), handle()
        assert k.CreatePipe(c.byref(inp), c.byref(self.input), None, 0)
        assert k.CreatePipe(c.byref(self.output), c.byref(out), None, 0)
        self.console = handle()
        assert k.CreatePseudoConsole(width | (30 << 16), inp, out, 0, c.byref(self.console)) == 0
        k.CloseHandle(inp)
        k.CloseHandle(out)
        size = c.c_size_t()
        k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        self.attrs = c.create_string_buffer(size.value)
        assert k.InitializeProcThreadAttributeList(self.attrs, 1, 0, c.byref(size))
        assert k.UpdateProcThreadAttribute(self.attrs, 0, 0x00020016, self.console,
                                          c.sizeof(handle), None, None)
        si, pi = SIX(), PI()
        si.si.cb, si.attrs = c.sizeof(SIX), c.addressof(self.attrs)
        command = c.create_unicode_buffer(subprocess.list2cmdline(args))
        assert k.CreateProcessW(None, command, None, None, False, 0x00080000,
                                None, None, c.byref(si), c.byref(pi)), c.get_last_error()
        self.process = pi.process
        k.CloseHandle(pi.thread)
        self.data = bytearray()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        buffer, count = c.create_string_buffer(4096), w.DWORD()
        while self.k.ReadFile(self.output, buffer, len(buffer), c.byref(count), None) and count.value:
            self.data.extend(buffer.raw[:count.value])

    def interrupt(self):
        count = w.DWORD()
        assert self.k.WriteFile(self.input, c.c_char_p(b'\x03'), 1, c.byref(count), None)

    def close(self):
        if self.k.WaitForSingleObject(self.process, 0) == 258:
            self.interrupt()
        clean = self.k.WaitForSingleObject(self.process, 5000) == 0
        if not clean:
            self.k.TerminateProcess(self.process, 1)  # Only this wrapper's child.
            self.k.WaitForSingleObject(self.process, 5000)
        self.k.ClosePseudoConsole(self.console)
        self.reader.join(5)
        for handle in (self.input, self.output, self.process):
            self.k.CloseHandle(handle)
        self.k.DeleteProcThreadAttributeList(self.attrs)
        assert clean, 'Ctrl+C did not stop the terminal watch'
        return bytes(self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--transport-only', action='store_true')
    args = parser.parse_args()
    root = args.artifacts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    facts = json.loads((repo / 'gui/wails/frontend/src/features/requests/progress-cli-facts.json').read_text(encoding='utf-8'))
    frames = [frame for frame in facts['replay'] for _ in range(frame['polls'])]
    (root / 'replay.json').write_text(json.dumps(frames, indent=2) + '\n', encoding='utf-8')
    lock, position, polls = threading.Lock(), [0], []
    stall_next = [False]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == '/slots'
            with lock:
                frame = dict(frames[position[0]])
                if stall_next[0]:
                    frame['stall_ms'] = 1800
                    stall_next[0] = False
                body = json.dumps(frame).encode()
                polls.append(time.monotonic())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    assert server.server_port not in (8081, 8082, 8083, 9931)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ, LOCALAPPDATA=str(root / 'local'), NO_COLOR='1',
               TERM='xterm-256color')
    env.pop('BUBO_MANIFEST', None)
    Path(env['LOCALAPPDATA']).mkdir(parents=True, exist_ok=True)
    bridge = None
    verdicts = []
    try:
        # Leave a genuine stopped discovery record before the live bridge.
        # Automatic watch must ignore it, while retaining live timeout errors.
        stopped_id = uuid.uuid4().hex
        stopped_bridge = subprocess.Popen(
            [str(args.bridge.resolve()), env['LOCALAPPDATA'], str(server.server_port), stopped_id],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        stopped_identity = json.loads(stopped_bridge.stdout.readline())
        stopped_out, stopped_error = stopped_bridge.communicate(timeout=5)
        assert stopped_bridge.returncode == 0, (stopped_out, stopped_error)
        (root / 'stopped-identity.json').write_text(json.dumps(stopped_identity, indent=2), encoding='utf-8')
        instance = uuid.uuid4().hex
        bridge = subprocess.Popen([str(args.bridge.resolve()), env['LOCALAPPDATA'],
                                   str(server.server_port), instance], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        identity = json.loads(bridge.stdout.readline())
        (root / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
        command = [str(args.cli.resolve()), 'status', '--watch', '--instance-id', instance]
        for name, width in (() if args.transport_only else (('plain', 120), ('tty', 120), ('tty80', 80))):
            tty = name != 'plain'
            screens = []
            with lock:
                position[0] = 0
                polls.clear()
            if tty:
                previous = dict(os.environ)
                os.environ.update(env)
                try:
                    process = Console([str(args.bridge.resolve()), '--console-run', *command], width)
                finally:
                    os.environ.clear()
                    os.environ.update(previous)
            else:
                process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                for index in range(len(frames)):
                    with lock:
                        position[0] = index
                        before = len(polls)
                    deadline = time.monotonic() + 5
                    while True:
                        with lock:
                            observed = len(polls) > before
                        if observed:
                            break
                        assert time.monotonic() < deadline, ('watch did not poll', tty, index)
                        time.sleep(0.025)
                    # Wait past rendering; frame selection follows actual polls.
                    time.sleep(0.1)
                    if tty:
                        screens.append(terminal_screen(bytes(process.data), width))
            finally:
                if tty:
                    data = process.close()
                else:
                    process.stdin.close()
                    process.wait(timeout=5)
                    data = process.stdout.read()
                    errors = process.stderr.read()
                    assert process.returncode == 0, (process.returncode, errors)
                (root / f'{name}.stdout.txt').write_bytes(data)
                if tty:
                    (root / f'{name}.screens.json').write_text(json.dumps(screens, indent=2) + '\n', encoding='utf-8')
            raw_text = data.decode('utf-8', errors='strict')
            text = '\n'.join(screens) if tty else raw_text
            reading = [line for line in text.splitlines() if 'prefill' in line or ' reading ' in line]
            assert all('| 0.0 tok/s' not in line for line in reading), reading
            partial = [line for line in reading if '4096/100000' in line]
            rates = [float(match[1]) for line in partial
                     if (match := re.search(r'([0-9]+\.[0-9]) tok/s', line))]
            expected = facts['provenance']['prefill_tokens_per_second']
            assert rates and all(abs(rate - expected) / expected < 0.05 for rate in rates), rates
            estimates = [int(match[1]) for line in partial
                         if (match := re.search(r'~ETA ([0-9]+)s', line))]
            assert estimates and all(70 <= seconds <= 76 for seconds in estimates), estimates
            assert '| - tok/s | ~ETA -' in partial[-1], partial[-1]
            assert 'Session 1 reading' in text and '1345/1373' in text
            assert '0/493' in text and 'Session 1 writing' in text and '14 generated' in text
            assert 'progress unavailable' in text and 'idle' in text
            assert 'Server loading. Session progress unavailable.' in text
            assert '8192/100000' in text and '8%' in text
            first = next(line for line in text.splitlines() if '0/100000' in line)
            assert '| - tok/s | ~ETA -' in first, first
            cached_line = next(line for line in text.splitlines() if '1345/1373' in line)
            assert '| - tok/s | ~ETA -' in cached_line, cached_line
            if width == 80:
                assert len(cached_line) <= 79 and '~ETA -' in cached_line, cached_line
                assert cached_line.endswith('(1345 cached)') or cached_line.endswith('...'), cached_line
            else:
                assert cached_line.split(' | Session')[0].endswith('(1345 cached)'), cached_line
            if width == 80:
                finished = next(i for i, frame in enumerate(frames) if frame['name'] == 'cached-finish')
                line = next(line for line in screens[finished].splitlines() if '1373/1373' in line)
                assert '~ETA 0s' in line and 'tok/s' in line and line.endswith('...'), line
                assert '1345 cached' not in line, line
            if tty:
                reset = next(i for i, frame in enumerate(frames) if frame['name'] == 'backwards')
                assert '| - tok/s | ~ETA -' in screens[reset], screens[reset]
            # ConPTY itself emits a reset at startup; it must not show any
            # selected foreground/background colors or bold decoration.
            assert not re.search(r'\x1b\[(?:[1-9][0-9]*)(?:;[0-9]+)*m', raw_text), 'NO_COLOR emitted decoration'
            for line in text.splitlines():
                if ' writing ' in line:
                    assert '%' not in line and 'ETA' not in line, line
            if tty:
                assert '\x1b[' in raw_text, repr(raw_text)
                cached_index = next(i for i, frame in enumerate(frames) if frame['name'] == 'cached')
                writing_index = next(i for i, frame in enumerate(frames) if frame['name'] == 'writing')
                assert 'Session 2 idle' in screens[cached_index]
                assert 'Session 2' not in screens[-1]
                assert len([line for line in screens[cached_index].splitlines() if line.startswith('Session ')]) == 2
                assert len([line for line in screens[writing_index].splitlines() if line.startswith('Session ')]) == 1
            else:
                assert '\x1b' not in text and '\r' not in text.replace('\r\n', '\n')
                lines = text.splitlines()
                assert all(a != b for a, b in zip(lines, lines[1:])), lines
                assert lines.count('Session 1 idle') == 1, lines
                assert all(b - a >= 0.9 for a, b in zip(polls, polls[1:])), polls
            verdicts.append(f'{name}: PASS ({len(polls)} real status polls)')
        # A stalled status call also occupies the bridge during the next
        # handshake. Auto discovery must not mistake that timeout for stopped.
        with lock:
            position[0] = next(i for i, frame in enumerate(frames) if frame['name'] == 'idle')
            stall_next[0] = True
        stalled = subprocess.Popen([str(args.cli.resolve()), 'status', '--watch'],
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(8.5)
        stalled.stdin.close()
        stalled.wait(timeout=5)
        out, errors = stalled.stdout.read(), stalled.stderr.read()
        (root / 'stall.stdout.txt').write_bytes(out)
        assert stalled.returncode == 0 and not errors, (stalled.returncode, errors)
        assert b'Server not answering' in out and b'Session 1 idle' in out, out
        assert b'Start the server' not in out and b'Not ready' not in out, out
        verdicts.append('stalled snapshot/handshake recovers without false stopped state: PASS')
        # Real executable, attached console, deliberately unread stdout pipe:
        # the first Ctrl+C cannot unblock its native write; the second must.
        with lock:
            before = len(polls)
        previous = dict(os.environ)
        os.environ.update(env)
        try:
            blocked = Console([str(args.bridge.resolve()), '--console-block-output', *command])
        finally:
            os.environ.clear()
            os.environ.update(previous)
        try:
            deadline = time.monotonic() + 5
            while len(polls) == before:
                assert time.monotonic() < deadline, 'blocked writer did not poll'
                time.sleep(0.025)
            time.sleep(0.3)
            blocked.interrupt()
            assert blocked.k.WaitForSingleObject(blocked.process, 300) == 258, 'first Ctrl+C unexpectedly escaped blocked write'
            blocked.interrupt()
            assert blocked.k.WaitForSingleObject(blocked.process, 2000) == 0, 'second Ctrl+C did not terminate blocked writer'
            code = w.DWORD()
            assert blocked.k.GetExitCodeProcess(blocked.process, c.byref(code))
            assert code.value == 0xC000013A, hex(code.value)
            (root / 'second-ctrl-c.txt').write_text(f'first: still blocked; second: exit {code.value:#x}\n', encoding='utf-8')
        finally:
            (root / 'blocked.stdout.txt').write_bytes(blocked.close())
        verdicts.append('second Ctrl+C terminates a real blocked stdout write: PASS')
        result = subprocess.run([str(args.cli.resolve()), 'status', '--help'], env=env,
                                capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0 and b'--watch' in result.stdout
        verdicts.append('human help documents --watch: PASS')
    finally:
        if bridge is not None:
            bridge_out, bridge_error = bridge.communicate(timeout=5)
            (root / 'bridge.stderr.txt').write_bytes(bridge_error)
            assert bridge.returncode == 0, (bridge.returncode, bridge_out, bridge_error)
        server.shutdown()
        server.server_close()
    (root / 'verdicts.txt').write_text('\n'.join(verdicts) + '\n', encoding='utf-8')
    print('\n'.join(verdicts))


if __name__ == '__main__':
    main()
