"""Synthetic, no-server process proof for the bootstrap diagnose bundle route.

Usage: python tests/cli/contract/bundle_process_test.py PATH_TO_BUBO
The executable is supplied by the caller; this script does not build it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from bundle_synthetic_proof import verify

SECRETS = (
    r"C:\Users\SensitiveUser\Documents",
    "user-home-needle-7391",
    "203.0.113.77",
    "2001:db8::badd",
    "Bearer token-secret-7391",
    "http://proxy-user:proxy-pass@proxy.internal:3128",
    "prompt-private-7391",
    "response-private-7391",
    "serial-private-7391",
    "SensitiveUser",
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(root):
    gui = root / "Bubo" / "gui"
    logs = gui / "logs"
    logs.mkdir(parents=True)
    write_json(gui / "settings.json", {
        "version": 1,
        "profile": "default",
        "context": 262144,
        "sessions": 2,
        "port": 8080,
        "start_gui_at_login": False,
        "model_path": SECRETS[0],
        "runtime_path": SECRETS[1],
        "data_folder": SECRETS[3],
        "auth": SECRETS[4],
        "proxy": SECRETS[5],
        "prompt": SECRETS[6],
        "response": SECRETS[7],
        "serial": SECRETS[8],
        "last_error": {"code": "device_lost", "remedy": SECRETS[6]},
    })
    write_json(gui / "diagnose-state.json", {
        "status": "WARN",
        "message": SECRETS[7],
        "checks": [{"name": "gpu", "status": "WARN",
                    "details": {"serial": SECRETS[8], "host": SECRETS[2]}}],
    })
    write_json(gui / "diagnose-history.json", [
        {"type": "diagnose", "payload": {"status": "PASS",
          "prompt": SECRETS[6], "response": SECRETS[7], "serial": SECRETS[8]}}
        for _ in range(40)
    ])
    (logs / "stderr.2.log").write_text(
        f"{SECRETS[4]} {SECRETS[6]}\n", encoding="utf-8")
    (logs / "stderr.1.log").write_text(
        json.dumps({"event": "ready", "level": "info", "count": 2,
                    "prompt": SECRETS[6], "proxy": SECRETS[5]}) + "\n",
        encoding="utf-8")
    (logs / "stderr.0.log").write_text(
        json.dumps({"event": "unknown", "message": SECRETS[8]}) + "\n",
        encoding="utf-8")


def tree_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def command(exe, env, *options):
    result = subprocess.run([str(exe), "diagnose", "--json", *map(str, options)],
                            env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=45, check=False)
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert len(lines) == 1, (result.returncode, result.stdout, result.stderr)
    return result.returncode, lines[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exe", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bubo-bundle-proof-", dir=args.work_root) as temp:
        root = Path(temp)
        fixture(root)
        env = os.environ.copy()
        env["LOCALAPPDATA"] = str(root)
        env["BUBO_MANIFEST"] = str(root / SECRETS[6])
        before = tree_hashes(root)
        target = root / "bubo-support.zip"

        code, preview = command(exe, env, "--preview", "--bundle", target)
        assert code == 0 and preview["type"] == "support_bundle"
        assert not target.exists(), "preview created the target"
        assert tree_hashes(root) == before, "preview wrote a fixture file"

        code, saved = command(exe, env, "--bundle", target)
        assert code == 0 and target.is_file()
        verify(preview, saved, target, SECRETS)
        args.work_root.joinpath('preview.json').write_text(json.dumps(preview, indent=2), encoding='utf-8')
        args.work_root.joinpath('saved.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
        args.work_root.joinpath('proof.zip').write_bytes(target.read_bytes())
        import zipfile
        extracted = args.work_root / 'extracted'
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(extracted)
        hits = [(str(path.name), secret) for path in extracted.iterdir() for secret in SECRETS
                if secret.encode() in path.read_bytes()]
        assert not hits
        print('Planted list: ' + json.dumps(SECRETS))
        print(f'ZIP extracted-file scan: {len(SECRETS)} planted values, 7 extracted files, {len(hits)} hits')
        print('Preview/save member hashes and ZIP SHA256: identical ' + preview['payload']['bundle_sha256'])
        logs = json.loads(preview["payload"]["text_preview"]["logs.json"])
        assert logs["dropped_lines"] >= 2
        assert logs["entries"] == [{"event": "ready", "level": "info", "count": 2}]

        code, event = command(exe, env, "--bundle", target)
        assert code == 7 and event["type"] == "error"
        assert event["payload"]["code"] == "storage_failed"
        code, event = command(exe, env, "--preview", "--preview")
        assert code == 2 and event["payload"]["code"] == "invalid_arguments"
        code, event = command(exe, env, "--bundle", root / ".." / "escape.zip")
        assert code == 2 and event["payload"]["code"] == "invalid_arguments"
        if os.name == "nt":
            code, event = command(exe, env, "--bundle", root / "NUL.zip")
            assert code == 2 and event["payload"]["code"] == "invalid_arguments"
    print("Support bundle process, privacy, ZIP, preview, and errors: 9 passed, 0 failed")


if __name__ == "__main__":
    main()
