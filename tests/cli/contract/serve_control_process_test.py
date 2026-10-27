"""CPU-only process proof for the production serve/control code.

Each subprocess is hidden and owned by this test. All HTTP requests use an
ephemeral loopback port. Cleanup can terminate only a Popen this test created.
No model, GPU runtime, process-name search or port-based process lookup exists.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from ctypes import wintypes
import http.client
import json
import os
from pathlib import Path
import queue
import random
import re
import socket
import subprocess
import threading
import time
import traceback
import uuid
from urllib.parse import urlsplit


def records(data: bytes, start_seq: int = 1) -> list[dict]:
    assert not data.startswith(b"\xef\xbb\xbf"), "Machine output has a BOM"
    assert b"\r" not in data, "Machine output has a CR"
    if data:
        assert data.endswith(b"\n"), "Machine output ends in a partial record"
    rows = []
    for line in data.splitlines():
        assert 0 < len(line) + 1 <= 1024 * 1024
        row = json.loads(line.decode("utf-8", errors="strict"))
        assert row["schema_version"] == 1
        assert re.fullmatch("[0-9a-f]{32}", row["operation_id"])
        assert isinstance(row["seq"], int) and row["seq"] == len(rows) + start_seq
        assert isinstance(row["product"], dict) and row["product"]["name"] == "bubo"
        assert isinstance(row["payload"], dict)
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", row["time"])
        rows.append(row)
    return rows


def identity(row: dict) -> dict:
    value = row.get("identity") or row.get("payload", row).get("identity")
    assert isinstance(value, dict), row
    assert re.fullmatch("[0-9a-f]{32}", value["instance_id"]), value
    assert isinstance(value["pid"], int) and value["pid"] > 0
    assert re.fullmatch("[1-9][0-9]{0,19}", value["process_creation_time"]), value
    assert isinstance(value["exe_path"], str) and Path(value["exe_path"]).is_absolute()
    return value


def ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert port not in (8081, 8082, 8083, 8084, 8085, 8086)
    return port


def read_record_line(process: subprocess.Popen, timeout: float = 5) -> dict:
    """Bound native stream reads so a failed proof still cleans owned children."""
    result: queue.Queue[object] = queue.Queue()

    def read():
        try:
            result.put(process.stdout.readline())
        except BaseException as error:
            result.put(error)

    threading.Thread(target=read, daemon=True).start()
    try:
        line = result.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError("The machine record deadline expired") from None
    if isinstance(line, BaseException):
        raise line
    assert line and line.endswith(b"\n") and b"\r" not in line
    return json.loads(line.decode("utf-8", errors="strict"))


def wait_busy_gate(gate: Path, server: "Server", responses: queue.Queue,
                   timeout: float = 30) -> None:
    """Wait for four arrivals and the blocked sampler, without timed sleeps."""
    # Python's stdlib has no filesystem notification API. Use the Windows
    # directory event through ctypes; this process proof is Windows-only.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FindFirstChangeNotificationW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL,
                                                   wintypes.DWORD]
    kernel.FindFirstChangeNotificationW.restype = wintypes.HANDLE
    kernel.FindNextChangeNotification.argtypes = [wintypes.HANDLE]
    kernel.FindCloseChangeNotification.argtypes = [wintypes.HANDLE]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    changed = kernel.FindFirstChangeNotificationW(str(gate.parent), False, 0x11)
    assert changed != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    signals = [Path(str(gate) + f".entered-{index}") for index in range(1, 5)]
    signals.append(Path(str(gate) + ".sampling"))
    deadline = time.monotonic() + timeout
    try:
        while True:
            assert server.process.poll() is None, ("busy server exited", server.events)
            assert responses.empty(), ("generation finished before gate release", responses.qsize())
            missing = [path.name for path in signals if not path.exists()]
            if not missing:
                print("BUSY GATE entered=4 sampler_blocked=true completed=0", flush=True)
                return
            remaining = deadline - time.monotonic()
            assert remaining > 0, ("busy gate readiness timed out", missing)
            result = kernel.WaitForSingleObject(changed, max(1, int(remaining * 1000)))
            assert result in (0, 258), ("directory wait failed", result, ctypes.get_last_error())
            if result == 0:
                assert kernel.FindNextChangeNotification(changed), ctypes.get_last_error()
    finally:
        assert kernel.FindCloseChangeNotification(changed), ctypes.get_last_error()


class Server:
    def __init__(self, proof: "Proof", folder: Path, *extra: str,
                 port: int | None = None, drain: bool = True):
        self.proof = proof
        self.folder = folder
        self.env = proof.environment(folder)
        self.port = port or ephemeral_port()
        assert self.port not in (8081, 8082, 8083, 8084, 8085, 8086)
        self.command = [str(proof.exe), "serve", "--jsonl", "--fake-backend",
                        "--host", "127.0.0.1", "--port", str(self.port),
                        "--log-level", "error", *extra]
        self.stderr = (folder / f"serve-{self.port}.stderr").open("wb")
        self.process = proof.popen(self.command, self.env, stderr=self.stderr)
        self.events: list[dict] = []
        self.raw = bytearray()
        self.lines: queue.Queue[object] = queue.Queue()
        self.instance: dict | None = None
        self.reader = None
        if drain:
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.raw.extend(line)
                parsed = json.loads(line.decode("utf-8", errors="strict"))
                self.events.append(parsed)
                self.lines.put(parsed)
        except BaseException as error:
            self.lines.put(error)
        finally:
            self.lines.put(None)

    def wait_event(self, kind: str, timeout: float = 12) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                row = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if isinstance(row, BaseException):
                raise row
            assert row is not None, (kind, self.process.poll(), self.events)
            if row["type"] == kind:
                self.instance = identity(row)
                assert self.instance["pid"] == self.process.pid
                return row
        raise AssertionError(("Missing event", kind, self.events, self.process.poll()))

    def wait_discovery(self, phase: str = "ready", timeout: float = 12) -> dict:
        deadline = time.monotonic() + timeout
        directory = self.folder / "Bubo" / "instances"
        while time.monotonic() < deadline:
            if directory.exists():
                for path in directory.glob("*.json"):
                    try:
                        value = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    candidate = value.get("identity", value)
                    if candidate.get("pid") == self.process.pid and value.get("phase") == phase:
                        self.instance = candidate
                        return value
            assert self.process.poll() is None, (self.process.returncode, self.events)
            time.sleep(0.02)
        raise AssertionError(("Missing discovery", phase, self.process.pid))

    @property
    def record_path(self) -> Path:
        assert self.instance
        return self.folder / "Bubo" / "instances" / f"{self.instance['instance_id']}.json"

    def request(self, prompt: str = "miss request", *, timeout: float = 15) -> dict:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request("POST", "/v1/completions",
                               json.dumps({"model": "bubo-test-only-fake",
                                           "prompt": prompt, "max_tokens": 2}),
                               {"Content-Type": "application/json"})
            result = connection.getresponse()
            body = result.read()
            assert result.status == 200, (result.status, body)
            value = json.loads(body)
            assert value["choices"][0]["text"] == "fake response", value
            assert value["usage"]["completion_tokens"] == 2, value
            return value
        finally:
            connection.close()

    def status(self) -> tuple[subprocess.CompletedProcess, list[dict]]:
        assert self.instance
        result, rows = self.proof.command(self.folder, "status", "--json", "--instance-id",
                                          self.instance["instance_id"])
        self.previous_status = getattr(self, "last_status", None)
        self.last_status = rows[0]["payload"]
        return result, rows

    def status_when(self, predicate):
        # Publication is asynchronous; require the exact expected observation
        # within one second, without relaxing counter/value assertions.
        deadline = time.monotonic() + 1
        while True:
            result, rows = self.status()
            assert result.returncode == 0, rows
            if predicate(rows[0]["payload"]):
                return result, rows
            assert time.monotonic() < deadline, ("publication exceeded 1 s", rows)
            time.sleep(0.01)

    def stop(self, wait_ms: int = 5000) -> tuple[subprocess.CompletedProcess, list[dict]]:
        assert self.instance
        return self.proof.command(self.folder, "stop", "--json", "--instance-id",
                                  self.instance["instance_id"], "--wait-ms", str(wait_ms),
                                  timeout=wait_ms / 1000 + 7)

    def exit(self, expected: int = 0, timeout: float = 8) -> None:
        assert self.process.wait(timeout=timeout) == expected, self.events
        if self.reader:
            self.reader.join(timeout=3)
            assert not self.reader.is_alive()
            for row in records(bytes(self.raw)):
                if row["type"] in ("starting", "loading", "ready", "warning",
                                   "stopping", "stopped", "error"):
                    value = identity(row)
                    assert value["pid"] == self.process.pid
                    for key in ("instance_id", "pid", "process_creation_time"):
                        assert row[key] == value[key], row
        self.stderr.close()

    def cleanup(self) -> None:
        if self.process.poll() is None:
            # The exact Popen handle belongs to this proof. No other process
            # is enumerated, selected by name or located through a port.
            self.process.kill()
            self.process.wait(timeout=5)
        if self.reader:
            self.reader.join(timeout=3)
        self.stderr.close()


class Proof:
    def __init__(self, executable: Path, root: Path, only_case: str | None = None):
        self.only_case = only_case
        self.exe = executable.resolve()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.owned: list[subprocess.Popen] = []
        self.servers: list[Server] = []
        self.passed: list[str] = []
        self.failed: list[str] = []

    def environment(self, folder: Path) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, LOCALAPPDATA=str(folder))
        for key in ("BUBO_MANIFEST", "HOST", "PORT", "BUBO_API_KEY"):
            env.pop(key, None)
        return env

    def popen(self, arguments: list[str], env: dict, **options) -> subprocess.Popen:
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                 if os.name == "nt" else 0)
        process = subprocess.Popen(arguments, env=env, cwd=self.root,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   creationflags=flags, **options)
        self.owned.append(process)
        return process

    def command(self, folder: Path, *arguments: str, timeout: float = 10):
        process = self.popen([str(self.exe), *arguments], self.environment(folder),
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        return result, records(stdout)

    def server(self, name: str, *extra: str, folder: Path | None = None, **options) -> Server:
        server = Server(self, folder or self.root / name, *extra, **options)
        self.servers.append(server)
        return server

    def run(self, name: str, function) -> None:
        if self.only_case is not None and self.only_case != name:
            return
        start = len(self.servers)
        process_start = len(self.owned)
        try:
            function()
            self.passed.append(name)
            print(f"PASS {name}", flush=True)
        except BaseException:
            self.failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
            for server in self.servers[start:]:
                if server.reader is None and server.process.poll() is not None:
                    print("FAILED CHILD MACHINE OUTPUT " + server.process.stdout.read().decode("utf-8"), flush=True)
        finally:
            for server in self.servers[start:]:
                server.cleanup()
            for process in self.owned[process_start:]:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


def assert_stop(rows: list[dict], exited: bool = True) -> None:
    assert rows and rows[0]["type"] == "stopping", rows
    assert rows[0]["payload"]["acknowledged"] is True, rows
    assert rows[0]["payload"]["process_exited"] is False, rows
    if exited:
        assert rows[-1]["type"] == "stopped", rows
        assert rows[-1]["payload"]["process_exited"] is True, rows


def error_exit(result: subprocess.CompletedProcess, rows: list[dict], code: int, name: str):
    assert result.returncode == code, (result.returncode, result.stderr, rows)
    assert rows[-1]["type"] == "error", rows
    assert rows[-1]["payload"]["code"] == name, rows


def pipe_acl(instance_id: str) -> dict:
    """Dump the actual pipe DACL, including each trustee, access and ACE flags."""
    assert os.name == "nt", "The local-control proof requires Windows"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    security.GetSecurityInfo.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.DWORD,
                                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_void_p)]
    security.GetSecurityInfo.restype = wintypes.DWORD
    security.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(wintypes.LPWSTR)]
    security.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(ctypes.c_void_p)]
    security.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p,
                                                      ctypes.POINTER(wintypes.WORD),
                                                      ctypes.POINTER(wintypes.DWORD)]
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.HANDLE)]
    security.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                             ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.POINTER(wintypes.DWORD)]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE

    def sid_string(pointer):
        string = wintypes.LPWSTR()
        assert security.ConvertSidToStringSidW(pointer, ctypes.byref(string))
        value = string.value
        kernel.LocalFree(string)
        return value

    token = wintypes.HANDLE()
    assert security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token))
    needed = wintypes.DWORD()
    security.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
    buffer = ctypes.create_string_buffer(needed.value)
    assert security.GetTokenInformation(token, 1, buffer, len(buffer), ctypes.byref(needed))
    current_sid = sid_string(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0])
    kernel.CloseHandle(token)

    name = rf"\\.\pipe\Bubo.Control.{instance_id}"
    handle = kernel.CreateFileW(name, 0x20000, 0, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, (name, ctypes.get_last_error())
    descriptor = ctypes.c_void_p()
    owner = ctypes.c_void_p()
    acl = ctypes.c_void_p()
    try:
        assert security.GetSecurityInfo(handle, 1, 5, ctypes.byref(owner), None,
                                        ctypes.byref(acl), None,
                                        ctypes.byref(descriptor)) == 0
        assert owner.value and acl.value
        owner_sid = sid_string(owner)
        ace_count = ctypes.cast(acl, ctypes.POINTER(wintypes.WORD))[2]
        entries = []
        for index in range(ace_count):
            ace = ctypes.c_void_p()
            assert security.GetAce(acl, index, ctypes.byref(ace))
            header = ctypes.string_at(ace, 8)
            entries.append({"type": header[0], "flags": header[1],
                            "mask": int.from_bytes(header[4:8], "little"),
                            "sid": sid_string(ace.value + 8)})
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        assert security.GetSecurityDescriptorControl(descriptor, ctypes.byref(control),
                                                      ctypes.byref(revision))
        dump = {"owner_sid": owner_sid, "current_sid": current_sid,
                "protected": bool(control.value & 0x1000), "aces": entries}
        assert owner_sid == current_sid and dump["protected"], dump
        assert len(entries) == 1 and entries[0]["type"] == 0, dump
        assert entries[0]["flags"] == 0 and entries[0]["sid"] == current_sid, dump
        assert entries[0]["sid"] not in ("S-1-1-0", "S-1-5-2"), dump
        return dump
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        kernel.CloseHandle(handle)


def remote_pipe_probe(instance_id: str) -> dict:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    # localhost traverses the remote named-pipe client path without contacting
    # any other host. An absent SMB transport is evidence of neither accept
    # nor reject and deliberately fails this mechanical proof.
    path = rf"\\localhost\pipe\Bubo.Control.{instance_id}"
    ctypes.set_last_error(0)
    handle = kernel.CreateFileW(path, 0xC0000000, 0, None, 3, 0, None)
    error = ctypes.get_last_error()
    if handle != ctypes.c_void_p(-1).value:
        kernel.CloseHandle(handle)
        raise AssertionError("A remote named-pipe client connected")
    assert error == 5, ("Remote rejection is not proven; transport result", error)
    return {"remote_path": path, "win32_error": error, "rejected": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", help="Run one exact case for failure diagnosis")
    options = parser.parse_args()
    proof = Proof(options.executable, options.root, options.case)

    def lifecycle():
        server = proof.server("lifecycle", "--fake-load-delay-ms", "150")
        ready = server.wait_event("ready")
        payload = ready["payload"]
        assert [row["type"] for row in server.events[:3]] == ["starting", "loading", "ready"]
        for row in server.events:
            assert identity(row) == server.instance
            for key in ("instance_id", "pid", "process_creation_time"):
                assert row[key] == server.instance[key], row
        assert payload["endpoint"] == f"http://127.0.0.1:{server.port}/v1"
        assert payload['endpoints'] == {
            'openai_base_url': f'http://127.0.0.1:{server.port}/v1',
            'anthropic_base_url': f'http://127.0.0.1:{server.port}'}
        assert payload['api_key_env'] == 'BUBO_API_KEY'
        estimate = payload['memory_estimate']
        assert estimate['gpu']['available']['bytes'] == 110192 * 1048576
        assert estimate['host']['available']['bytes'] == 11900000000
        assert estimate['fits'] is True and estimate['fit_verified'] is False
        assert estimate['recommended'] is True
        assert estimate['per_session_vision']['bytes'] == 41992256
        assert estimate['recommended_context']['by_sessions'][0]['context_per_session'] == 262144
        assert estimate['recommended_context']['by_sessions'][0]['fits'] is True
        assert estimate['recommended_context']['by_sessions'][0]['recommended'] is True
        assert payload["served_model"] == "bubo-test-only-fake"
        assert payload['connection'] == {'endpoint': payload['endpoint'], 'model_id': payload['served_model'],
                                          'context_tokens': 4096, 'authenticated': False}
        assert payload['slots'][0]['state'] == 'idle' and payload['slots'][0]['tokens_used'] == 0
        assert payload["log_path"] is None
        for setting in payload["effective_settings"].values():
            assert all(key in setting for key in ("requested", "effective", "source", "reason"))
        # A completion immediately after Ready proves the listener and backend
        # both accept requests. No log text or open port alone supplies state.
        server.request()
        status_result, status_rows = server.status_when(lambda sample: len(sample["requests"] or []) == 1)
        assert status_result.returncode == 0 and len(status_rows) == 1
        snapshot = status_rows[0]["payload"]
        assert snapshot["phase"] == "ready" and snapshot["prepared"] is True
        assert snapshot['endpoints'] == payload['endpoints']
        assert snapshot['api_key_env'] == payload['api_key_env']
        assert snapshot['memory_estimate'] == payload['memory_estimate']
        assert identity(status_rows[0]) == server.instance
        for key in ("versions", "effective_settings", "readiness", "sample_time",
                    "last_transition", "last_error", "counters", "memory"):
            assert key in snapshot, key
        stop_result, stop_rows = server.stop()
        assert stop_result.returncode == 0
        assert_stop(stop_rows)
        server.exit()
        assert [row["type"] for row in server.events] == ["starting", "loading", "ready", "request_completed", "stopping", "stopped"]
        completed = next(row['payload'] for row in server.events if row['type'] == 'request_completed')
        assert snapshot['requests'] == [{key: value for key, value in completed.items() if key != 'identity'}]
        assert completed['outcome'] == 'completed' and completed['prompt_tokens'] > 0
        assert not {'prompt', 'text', 'response', 'body', 'client_id'} & completed.keys()
        (proof.root / "captured-serve.jsonl").write_bytes(server.raw)
        (proof.root / "captured-status.jsonl").write_bytes(status_result.stdout)
        (proof.root / "captured-stop.jsonl").write_bytes(stop_result.stdout)
        transcript = ("$ " + subprocess.list2cmdline(server.command) + "\n" +
                      server.raw.decode("utf-8") + "$ " +
                      subprocess.list2cmdline(status_result.args) + "\n" +
                      status_result.stdout.decode("utf-8") + "$ " +
                      subprocess.list2cmdline(stop_result.args) + "\n" +
                      stop_result.stdout.decode("utf-8"))
        (proof.root / "captured-transcript.txt").write_text(transcript, encoding="utf-8", newline="\n")
        assert b"[INFO]" not in (server.folder / f"serve-{server.port}.stderr").read_bytes()

    proof.run("serve_lifecycle_quiet_order_fields_ready_after_bind", lifecycle)

    def sdk_endpoints():
        server = proof.server("sdk-endpoints")
        ready = server.wait_event("ready")["payload"]
        for key, suffix in (("openai_base_url", "/chat/completions"),
                            ("anthropic_base_url", "/v1/messages")):
            url = urlsplit(ready["endpoints"][key] + suffix)
            assert url.hostname == "127.0.0.1" and url.port == server.port
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=4)
            try:
                connection.request("POST", url.path,
                    json.dumps({"model": ready["served_model"], "max_tokens": 2,
                                "messages": [{"role": "user", "content": "endpoint proof"}]}),
                    {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                response = connection.getresponse()
                body = json.loads(response.read())
                assert response.status == 200 and "error" not in body, (key, response.status, body)
            finally:
                connection.close()
        assert server.status()[1][0]["payload"]["endpoints"] == ready["endpoints"]
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("serve_sdk_endpoints_match_bound_chat_and_messages_routes", sdk_endpoints)

    def page_facts():
        for imported, authenticated in ((False, False), (True, True)):
            extra = ["--fake-imported"] if imported else []
            if authenticated:
                extra += ["--fake-authenticated"]
            server = proof.server("page-facts-" + str(imported), *extra)
            ready = server.wait_event("ready")["payload"]
            result, rows = server.status()
            assert result.returncode == 0
            snapshot = rows[0]["payload"]
            active = snapshot["active_model"]
            assert active == ready["active_model"]
            assert active == snapshot["instances"][0]["active_model"]
            assert Path(active["model_path"]).is_absolute()
            assert (active["catalogue_id"] is None) == imported
            assert (active["manifest_path"] is None) == imported
            if not imported:
                assert active["catalogue_id"] == "huihui-qwen38-flash-next-ud-q4-k-xl-v1"
            snippets = snapshot["client_snippets"]
            assert snippets == ready["client_snippets"] and len(snippets) == 3
            assert {s["kind"] for s in snippets} == {
                "openai_python", "powershell_openai_chat", "anthropic_python"}
            for item in snippets:
                assert item["requires_key"] == authenticated
                assert item["default_api_key"] == (None if authenticated else "bubo-loopback")
                text = item["template"]
                assert "{api_key}" in text
                for name, url in snapshot["endpoints"].items():
                    # Also prove wildcard/LAN presentation needs no regeneration.
                    text = text.replace("{" + name + "}", url.replace("127.0.0.1", "192.0.2.1"))
                text = text.replace("{api_key}", "dummy")
                if item["kind"].endswith("python"):
                    ast.parse(text)
                assert "0.0.0.0" not in text
            assert snapshot["api_key_env"] == "BUBO_API_KEY"
            assert snapshot["pi_snippet"] is not None
            (proof.root / ("captured-page-facts-" + str(imported) + ".json")).write_text(
                json.dumps(rows[0]), encoding="utf-8")
            assert server.stop()[0].returncode == 0
            server.exit()

    proof.run("status_active_identity_and_structured_client_templates", page_facts)

    def quiet_warning():
        server = proof.server("quiet-warning", "--fake-event-burst", "1")
        server.wait_event("ready")
        warning = server.wait_event("warning")
        assert warning["payload"]["code"] == "fake_output_pressure"
        server.request()
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("serve_warning_event_survives_quiet_human_log_threshold", quiet_warning)

    def consumer_closed():
        server = proof.server("closed-output", "--fake-load-delay-ms", "150", drain=False)
        server.process.stdout.close()
        server.wait_discovery()
        for index in range(5):
            server.request(f"miss after close {index}")
        result, rows = server.status()
        assert result.returncode == 0 and rows[0]["payload"]["phase"] == "ready"
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("serve_stdout_consumer_closed_five_requests_complete", consumer_closed)

    def consumer_blocked():
        server = proof.server("blocked-output", "--fake-event-burst", "512", drain=False)
        server.wait_discovery()
        started = time.monotonic()
        for index in range(5):
            server.request(f"memory during blocked output {index}")
        assert time.monotonic() - started < 4
        assert server.status()[0].returncode == 0
        assert server.stop()[0].returncode == 0
        server.exit(timeout=8)

    proof.run("serve_stdout_blocked_five_requests_complete", consumer_blocked)

    def acl():
        server = proof.server("pipe-acl")
        server.wait_event("ready")
        dump = pipe_acl(server.instance["instance_id"])
        (proof.root / "captured-pipe-dacl.json").write_text(json.dumps(dump), encoding="utf-8")
        print("PIPE DACL " + json.dumps(dump), flush=True)
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("control_pipe_explicit_protected_owner_sid_only_dacl", acl)

    def remote():
        server = proof.server("pipe-remote")
        server.wait_event("ready")
        result = remote_pipe_probe(server.instance["instance_id"])
        print("REMOTE PIPE " + json.dumps(result), flush=True)
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("control_remote_pipe_client_rejected", remote)

    def tampered_identity(field: str, wrong, explicit_stop: bool):
        server = proof.server("tampered-" + field + str(explicit_stop))
        server.wait_event("ready")
        original = server.record_path.read_bytes()
        value = json.loads(original)
        identity_object = value.get("identity", value)
        identity_object[field] = wrong
        server.record_path.write_text(json.dumps(value), encoding="utf-8")
        try:
            if explicit_stop:
                result, rows = server.stop()
                error_exit(result, rows, 16, "control_access")
            else:
                result, rows = proof.command(server.folder, "status", "--json")
                assert result.returncode == 0
                assert rows[0]["payload"]["instances"] == [], rows
                assert rows[0]["payload"]["phase"] != "ready", rows
            server.request("miss after wrong identity")
        finally:
            server.record_path.write_bytes(original)
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("status_stale_discovery_pid_reuse_wrong_creation_ignored",
              lambda: tampered_identity("process_creation_time", "1", False))
    proof.run("stop_wrong_creation_time_exit_16",
              lambda: tampered_identity("process_creation_time", "1", True))
    proof.run("stop_wrong_canonical_executable_exit_16",
              lambda: tampered_identity("exe_path", str(proof.root / "other.exe"), True))

    def wrong_id():
        result, rows = proof.command(proof.root / "wrong-id", "stop", "--json",
                                     "--instance-id", "0" * 32, "--wait-ms", "100")
        error_exit(result, rows, 16, "control_access")

    proof.run("stop_wrong_instance_id_exit_16", wrong_id)

    def ack_exit():
        server = proof.server("stop-separation", "--fake-stop-delay-ms", "750")
        server.wait_event("ready")
        stopper = proof.popen([str(proof.exe), "stop", "--json", "--instance-id",
                               server.instance["instance_id"], "--wait-ms", "5000"],
                              server.env, stderr=subprocess.PIPE)
        ack = read_record_line(stopper)
        assert_stop([ack], exited=False)
        assert server.process.poll() is None, "Acknowledgement was emitted after exit"
        assert stopper.poll() is None, "Stop returned success before process exit"
        rest, stderr = stopper.communicate(timeout=8)
        assert stopper.returncode == 0, stderr
        assert_stop([ack, *records(rest, start_seq=2)])
        server.exit()

    proof.run("stop_acknowledgement_precedes_confirmed_process_exit", ack_exit)

    def deadline():
        server = proof.server("stop-deadline", "--fake-hang-stop")
        server.wait_event("ready")
        result, rows = server.stop(wait_ms=150)
        assert_stop(rows, exited=False)
        error_exit(result, rows, 17, "stop_timeout")
        assert all(row["type"] != "stopped" for row in rows)
        assert server.process.poll() is None

    proof.run("stop_deadline_exit_17_never_reports_stopped", deadline)

    def repeat_stop():
        server = proof.server("repeat-stop")
        server.wait_event("ready")
        assert server.stop()[0].returncode == 0
        server.exit()
        result, rows = server.stop()
        assert result.returncode == 0 and rows[-1]["type"] == "stopped", rows
        tombstone = json.loads(server.record_path.read_bytes())
        assert tombstone["phase"] == "stopped", tombstone
        text = json.dumps(tombstone).lower()
        for forbidden in ("api_key", "credentials", "prompt", "request_body", "token="):
            assert forbidden not in text, forbidden

    proof.run("stop_repeat_recorded_stopped_success_redacted_tombstone", repeat_stop)

    def conflict():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            assert port not in (8081, 8082, 8083, 8084, 8085, 8086)
            server = proof.server("conflict", "--fake-load-delay-ms", "1500", port=port)
            server.exit(expected=13, timeout=3)
            assert all(row["type"] != "loading" for row in server.events), server.events
            assert any(row["type"] == "error" and row["payload"]["code"] == "port_occupied"
                       for row in server.events), server.events

    proof.run("serve_port_conflict_exit_13_before_loading", conflict)

    def concurrent():
        folder = proof.root / "concurrent"
        port = ephemeral_port()
        servers = [proof.server("concurrent", "--fake-load-delay-ms", "350", folder=folder,
                                port=port) for _ in range(2)]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            exited = [server for server in servers if server.process.poll() is not None]
            ready = [server for server in servers if any(row["type"] == "ready" for row in server.events)]
            if exited and ready:
                break
            time.sleep(0.02)
        assert len(exited) == 1 and len(ready) == 1, [server.events for server in servers]
        loser, winner = exited[0], ready[0]
        loser.exit(expected=13)
        assert all(row["type"] != "loading" for row in loser.events), loser.events
        winner.wait_discovery()
        winner.request()
        assert winner.stop()[0].returncode == 0
        winner.exit()

    proof.run("serve_two_concurrent_starts_same_endpoint_one_owner", concurrent)

    def multiple():
        folder = proof.root / "multiple"
        servers = [proof.server("multiple", folder=folder) for _ in range(2)]
        for server in servers:
            server.wait_event("ready")
        result, rows = proof.command(folder, "status", "--json")
        assert result.returncode == 0 and len(rows) == 1
        snapshot = rows[0]["payload"]
        assert len(snapshot["instances"]) == 2, snapshot
        assert snapshot.get("identity") is None, "Status selected an arbitrary instance"
        assert snapshot.get("selection_required") is True, snapshot
        for server in servers:
            assert server.status()[0].returncode == 0
            assert server.stop()[0].returncode == 0
            server.exit()

    proof.run("status_multiple_instances_require_selection", multiple)

    def watch_eof():
        server = proof.server("watch-eof")
        server.wait_event("ready")
        watcher = proof.popen([str(proof.exe), "status", "--json", "--watch", "--instance-id",
                               server.instance["instance_id"]], server.env, stderr=subprocess.PIPE)
        row = read_record_line(watcher)
        assert row["type"] == "status", row
        started = time.monotonic()
        watcher.stdin.close()
        watcher.stdin = None
        watcher.communicate(timeout=2)
        assert watcher.returncode == 0 and time.monotonic() - started < 1.5
        server.request()
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("status_watch_stdin_eof_prompt_exit_server_keeps_serving", watch_eof)

    def metrics():
        server = proof.server("cache-completion-property")
        server.wait_event("ready")
        initial = server.status()[1][0]["payload"]
        assert initial["cache_hit_ratio"] is None
        assert initial["last_completed_request"]["age_ms"] is None
        assert initial["last_completed_request"]["decode_tokens_per_second"] is None
        assert initial["memory"]["unit"] == "bytes"
        for key in ("rss_bytes", "host_available_bytes"):
            assert initial["memory"][key] is None or (
                isinstance(initial["memory"][key], int) and initial["memory"][key] > 0)
        randomizer = random.Random(271828)
        expected = {"disk": 0, "memory": 0, "miss": 0}
        observed = []
        for index in range(37):
            mode = randomizer.choice(tuple(expected))
            server.request(f"{mode} property request {index}")
            expected[mode] += 1
            counters = server.status_when(lambda sample: sample["counters"]["completed_successful_requests_total"] == index + 1)[1][0]["payload"]["counters"]
            observed.append(counters)
            (proof.root / "cache-completion-counters.json").write_text(
                json.dumps(observed, indent=2), encoding="utf-8")
            assert counters["cache_disk_hits_total"] == expected["disk"], counters
            assert counters["cache_memory_hits_total"] == expected["memory"], counters
            assert counters["cache_misses_total"] == expected["miss"], counters
            assert counters["completed_successful_requests_total"] == index + 1
            assert counters["decode_tokens_total"] == 2 * (index + 1), counters
        final = server.status()[1][0]["payload"]
        (proof.root / "cache-completion-final.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8")
        counters = final["counters"]
        assert counters["cached_tokens_total"] == 4 * (expected["disk"] + expected["memory"])
        assert counters["decode_tokens_total"] == 74, counters
        assert counters["prefill_tokens_total"] == 4 * (expected["disk"] + expected["memory"]) + 8 * expected["miss"]
        assert abs(final["cache_hit_ratio"] - (expected["disk"] + expected["memory"]) / 37) < 1e-12
        gauge = final["last_completed_request"]
        assert gauge["age_ms"] >= 0 and gauge["age_unit"] == "milliseconds"
        assert gauge["prefill_tokens_per_second"] > 0
        assert gauge["decode_tokens_per_second"] > 0 and gauge["rate_unit"] == "tokens_per_second"
        assert final["sessions"] == 0 and final["queue"] == 0
        assert len(final["slots"]) == 1 and not final["slots"][0]["processing"]
        print("CACHE PROPERTY " + json.dumps({"requests": 37, "expected": expected}), flush=True)
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("serve_cache_completion_37_mixed_requests_exactly_once_typed_metrics", metrics)

    def watch_rates():
        server = proof.server("watch-rates")
        server.wait_event("ready")
        watcher = proof.popen([str(proof.exe), "status", "--json", "--watch", "--instance-id",
                               server.instance["instance_id"]], server.env, stderr=subprocess.PIPE)
        initial = read_record_line(watcher)["payload"]
        first = initial["live_throughput"]
        assert first["label"] == "aggregate wall throughput" and first["unit"] == "tokens_per_second"
        assert first["prefill_tokens_per_second"] is None and first["decode_tokens_per_second"] is None
        server.request("miss monotonic rate")
        later = read_record_line(watcher)["payload"]
        live = later["live_throughput"]
        assert live["interval_ms"] > 0
        elapsed_seconds = live["interval_ms"] / 1000
        for field, counter in (("prefill_tokens_per_second", "prefill_tokens_total"),
                               ("decode_tokens_per_second", "decode_tokens_total")):
            delta = later["counters"][counter] - initial["counters"][counter]
            assert delta > 0 and abs(live[field] - delta / elapsed_seconds) < 1e-9
        watcher.stdin.close()
        watcher.stdin = None
        watcher.communicate(timeout=2)
        assert watcher.returncode == 0
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("status_watch_rates_use_counter_deltas_over_monotonic_wall_time", watch_rates)

    def poll_rates():
        server = proof.server("poll-rates")
        server.wait_event("ready")
        # Each status() launches a fresh CLI process, as the GUI does. The
        # fake supplies results/counters via RuntimeStatusMetrics and the
        # real ServeLifecycle control publication, never a JSON fixture.
        initial = server.status()[1][0]["payload"]
        server.request("miss poll publication")
        later = server.status_when(lambda sample: sample["counters"]["decode_tokens_total"] > initial["counters"]["decode_tokens_total"])[1][0]["payload"]
        live = later["live_throughput"]
        assert live["interval_ms"] > 0, live
        assert live["label"] == "aggregate wall throughput"
        assert live["unit"] == "tokens_per_second"
        elapsed_seconds = live["interval_ms"] / 1000
        for field, counter in (("prefill_tokens_per_second", "prefill_tokens_total"),
                               ("decode_tokens_per_second", "decode_tokens_total")):
            delta = later["counters"][counter] - server.previous_status["counters"][counter]
            assert delta > 0 and abs(live[field] - delta / elapsed_seconds) < 1e-9
        assert isinstance(later["queue"], int) and later["queue"] == 0
        assert isinstance(later["sessions"], int) and later["sessions"] == 0
        last = later["last_completed_request"]
        assert last["age_ms"] >= 0 and last["age_unit"] == "milliseconds"
        assert last["decode_tokens_per_second"] > 0
        assert last["prefill_tokens_per_second"] > 0
        assert last["rate_unit"] == "tokens_per_second"
        # No work between polls is measured as zero, preserving a live gauge.
        idle = server.status_when(lambda sample: sample["live_throughput"]["interval_ms"] is not None)[1][0]["payload"]["live_throughput"]
        assert idle["decode_tokens_per_second"] == 0
        assert idle["prefill_tokens_per_second"] == 0
        (proof.root / "captured-poll-status.json").write_text(
            json.dumps(later), encoding="utf-8")
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("status_separate_process_polls_share_real_publication_baseline", poll_rates)

    def rejected_switch():
        server = proof.server("rejected-switch", "--fake-rejected-bias", "not-a-number")
        row = server.wait_event("error")
        assert row["payload"]["code"] == "invalid_arguments"
        assert row["payload"]["switch"] == "--think-marker-bias"
        assert row["payload"]["remedy"] == "Check the settings and model files."
        server.wait_event("stopped")
        server.exit(2)
        (proof.root / "captured-rejection.jsonl").write_bytes(server.raw)

    proof.run("serve_rejected_release_setting_names_switch", rejected_switch)

    def busy_status():
        folder = proof.root / "busy-status"
        folder.mkdir(parents=True, exist_ok=True)
        gate = folder / ("release-generation-" + uuid.uuid4().hex)
        assert not gate.exists(), "fresh gate root required"
        server = proof.server("busy-status", "--fake-generation-gate", str(gate),
                              "--fake-npu-status", folder=folder)
        ready = server.wait_event("ready")
        npu_status = ready["payload"]["npu_prefill_runtime"]
        assert npu_status["state"] == "off", npu_status
        assert npu_status["reason"] == "CPU fake startup denial", npu_status
        started = time.monotonic()
        idle = server.status()[1][0]
        idle_ms = (time.monotonic() - started) * 1000
        responses: queue.Queue[object] = queue.Queue()
        def generate():
            try:
                responses.put(server.request(timeout=45))
            except BaseException as error:
                responses.put(error)
        workers = [threading.Thread(target=generate, daemon=True) for _ in range(4)]
        watcher = None
        try:
            for worker in workers:
                worker.start()
            wait_busy_gate(gate, server, responses)
            assert all(worker.is_alive() for worker in workers)
            started = time.monotonic()
            busy = server.status()[1][0]
            busy_ms = (time.monotonic() - started) * 1000
            print("LATENCY " + json.dumps({"idle_ms": idle_ms, "busy_ms": busy_ms}), flush=True)
            assert busy_ms < 1000, ("busy status exceeded 1 s", busy_ms)
            assert idle_ms < 1000, ("idle status exceeded 1 s", idle_ms)
            assert busy["type"] == "status", busy
            assert busy["payload"]["npu_prefill_runtime"] == npu_status, busy
            assert busy["payload"]["sample_time"] <= busy["time"]
            watcher = proof.popen([str(proof.exe), "status", "--jsonl", "--watch", "--instance-id",
                                   server.instance["instance_id"]], server.env, stderr=subprocess.PIPE)
            assert read_record_line(watcher)["type"] == "capabilities"
            arrived = []
            samples = []
            for _ in range(4):
                row = read_record_line(watcher, timeout=1.2)
                assert row["type"] == "status", row
                assert row["payload"]["npu_prefill_runtime"] == npu_status, row
                arrived.append(time.monotonic())
                samples.append(row)
            gaps = [(right - left) * 1000 for left, right in zip(arrived, arrived[1:])]
            print("WATCH GAPS MS " + json.dumps(gaps), flush=True)
            assert all(gap <= 1000 for gap in gaps), gaps
            # A blocked observation retains its actual sample time. Receipt time
            # must never make stale scheduler metrics appear newly sampled.
            assert len({row["payload"]["sample_time"] for row in samples}) == 1
            ages = [row["payload"]["sample_age_ms"] for row in samples]
            assert ages[-1] - ages[0] >= 2400, ages
            assert len({row["payload"]["machine"]["sample_time"] for row in samples}) > 1
            assert responses.empty(), "all four generations must stay busy throughout status/watch"
            assert all(worker.is_alive() for worker in workers)
            timer = proof.popen([str(proof.exe), "control-timings", server.instance["instance_id"]],
                                server.env, stderr=subprocess.PIPE)
            output, error = timer.communicate(timeout=1)
            assert timer.returncode == 0, error
            print("CONTROL TIMINGS " + output.decode("utf-8").strip(), flush=True)
            (proof.root / "captured-busy-status.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in [busy, *samples]), encoding="utf-8")
        finally:
            gate.touch()
            for worker in workers:
                worker.join(timeout=5)
            assert all(not worker.is_alive() for worker in workers)
            for path in [gate, Path(str(gate) + ".sampling"),
                         *[Path(str(gate) + f".entered-{index}") for index in range(1, 5)]]:
                path.unlink(missing_ok=True)
            if watcher is not None:
                watcher.stdin.close()
                watcher.stdin = None
                watcher.communicate(timeout=3)
        assert responses.qsize() == 4
        for _ in workers:
            result = responses.get_nowait()
            assert not isinstance(result, BaseException), result
        last = server.status()[1][0]["payload"]["npu_prefill_runtime"]
        assert last["memory"] == {"min_available_mib": 1024, "max_commit_pct": 75}, last
        assert last["chip_power"]["w_mean"] == 12.5, last
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("status_busy_generation_one_shot_and_watch_latency", busy_status)

    def rotating_logs():
        folder = proof.root / "log-rotation"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "serve.log"
        server = proof.server("log-rotation", "--log-file", str(path), "--log-max-bytes", "4096",
                              "--log-retain", "2", "--fake-log-burst", "64", folder=folder)
        ready = server.wait_event("ready")
        assert Path(ready["payload"]["log_path"]) == path
        assert Path(server.status()[1][0]["payload"]["log_path"]) == path
        server.request()
        assert server.stop()[0].returncode == 0
        server.exit()
        files = [path, Path(str(path) + ".1"), Path(str(path) + ".2")]
        assert all(file.exists() and 0 < file.stat().st_size <= 4096 for file in files)
        assert not Path(str(path) + ".3").exists()
        discovery = json.loads(server.record_path.read_bytes())
        assert discovery["phase"] == "stopped" and Path(discovery["log_path"]) == path
        print("LOG ROTATION " + json.dumps({file.name: file.stat().st_size for file in files}), flush=True)

    proof.run("serve_existing_async_log_sink_bounded_rotation_reported_path", rotating_logs)

    def loading_stop():
        server = proof.server("loading-stop", "--fake-load-delay-ms", "2000")
        server.wait_event("loading")
        assert server.status()[1][0]["payload"]["endpoints"] is None
        result, rows = server.stop()
        assert result.returncode == 0
        assert_stop(rows)
        server.exit()
        assert all(row["type"] != "ready" for row in server.events)

    proof.run("stop_during_loading_no_ready_event", loading_stop)

    def no_http_shutdown():
        server = proof.server("no-http-shutdown")
        server.wait_event("ready")
        for path in ("/shutdown", "/v1/shutdown"):
            connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
            connection.request("POST", path, "{}", {"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read()
            assert response.status == 404
            connection.close()
        server.request()
        assert server.stop()[0].returncode == 0
        server.exit()

    proof.run("serve_no_public_http_shutdown_endpoint", no_http_shutdown)

    if options.case and not proof.passed and not proof.failed:
        parser.error("Unknown case: " + options.case)
    print(f"Serve/control process proof: {len(proof.passed)} passed, {len(proof.failed)} failed", flush=True)
    (proof.root / "case-counts.json").write_text(
        json.dumps({"passed": proof.passed, "failed": proof.failed}, indent=2), encoding="utf-8")
    for process in proof.owned:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    return 1 if proof.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
