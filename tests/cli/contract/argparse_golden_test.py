"""Frozen argv/options and rejection records from the pre-ArgParser CLI.

The driver only calls pure parsers. Process cases use the production executable
and rejection-only lifecycle wrapper; none starts a listener or loads a model.
--record is explicit: use it before a parser migration or after an intentional
CLI contract extension, then review every changed record.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile

ID = "0123456789abcdef0123456789abcdef"


def cases():
    rows = []
    def add(command, *args):
        rows.append([command, *args])
    for command in ("prepare", "status", "stop", "models", "diagnose", "bundle", "serve"):
        for args in ([], ["--help"], ["-h"], ["--unknown"], [""], ["--json"],
                     ["--jsonl"], ["--json", "--jsonl"], ["--help", "--unknown"]):
            add(command, *args)
    for step in ("all", "runtime", "model", "bogus"):
        add("prepare", step)
        add("prepare", "--yes", step)
        add("prepare", step, "--help")
    for flag in ("--output", "--mtp-model", "--mmproj", "--cache-dir", "--runtime-dir", "--catalogue-id", "--model-set"):
        for args in ([flag], [flag, ""], [flag, "--yes"], [flag, "bad\npath"],
                     [flag, "folder", flag, "other"], [flag + "=folder"], [flag, "folder"]):
            add("prepare", *args)
    for args in (["--no-resume", "--resume"], ["--resume", "--no-resume"],
                 ["model", "--plan"], ["runtime", "--plan"], ["--npu-sidejobs", "--accept-gemma-terms"],
                 ["model", "--model-set", "original"], ["--help", "--jsonl"],
                 ["model", "--model-set", "lowmemory", "--plan", "--offline"],
                 ["model", "--catalogue-id", "huihui-qwen38-flash-next-lowmemory-v1", "--plan", "--offline"],
                 ["--yes", "--yes", "--offline", "--verify"]):
        add("prepare", *args)
    for args in (["runtime", "--mtp-model", "file"], ["runtime", "--mmproj", "file"],
                 ["model", "--model-set", "hybrid", "--plan", "--offline"],
                 ["model", "--catalogue-id", "huihui-qwen38-flash-next-ista-hybrid-v1", "--plan", "--offline"],
                 ["runtime", "--model-set", "huihui"], ["model", "--runtime-dir", "folder"],
                 ["--plan"], ["model", "--npu-sidejobs"], ["runtime", "--accept-gemma-terms"],
                 ["model", "--plan", "--npu-sidejobs"],
                 ["--catalogue-id", "qwen38-27b-ud-q4-k-xl", "--model-set", "original"],
                 ["--catalogue-id", "qwen38-27b-ud-q4-k-xl", "--mtp-model", "file"],
                 ["runtime", "--catalogue-id", "qwen38-27b-ud-q4-k-xl"],
                 ["--catalogue-id", "qwen38-27b-ud-q4-k-xl"]):
        add("prepare", *args)
    for args in (["--json", "--watch"], ["--jsonl", "--watch", "--watch"],
                 ["--json", "--instance-id", ID], ["--json", "--instance-id=" + ID],
                 ["--json", "--instance-id"], ["--json", "--instance-id="],
                 ["--json", "--instance-id", ID, "--instance-id", ID],
                 ["--json", "--instance-id", "--watch"], ["--json", "--watch=true"]):
        add("status", *args)
    for value in ("0", "1", "45000", "4294967294", "4294967295", "4294967296", "-1", "+1", " 1", "1x", ""):
        add("stop", "--json", "--instance-id", ID, "--wait-ms", value)
    for args in (["--json", "--instance-id", ID], ["--json", "--instance-id=" + ID],
                 ["--json", "--instance-id", ID, "--wait-ms=0"], ["--json", "--instance-id"],
                 ["--json", "--instance-id", ID, "--instance-id", ID],
                 ["--json", "--wait-ms", "0", "--wait-ms", "1"], ["--json", "--wait-ms"]):
        add("stop", *args)
    for verb in ("list", "estimate"):
        add("models", verb)
        add("models", verb, "--help", "--unknown")
        for flag in ("--scan-folder", "--models-root", "--model", "--context", "--sessions"):
            for args in ([flag], [flag, ""], [flag + "=folder"], [flag, "1"], [flag, "2", flag, "3"]):
                add("models", verb, *args)
    for value in ("0", "0001", "999999999999", "1000000000000", "-1", "+1", "1.0"):
        add("models", "estimate", "--model", "fixture", "--context", value)
    add("models", "list", "--scan-folder", "first", "--scan-folder", "second", "--json")
    add("models", "estimate", "--model", "fixture", "--context", "32", "--sessions", "4")
    for flag in ("--validate-artifact", "--benchmark", "--backends", "--warmup", "--repetitions", "--duration-ms", "--working-set-mib", "--output", "--section"):
        for args in ([flag], [flag, ""], [flag + "=1"], [flag, "1"], [flag, "2", flag, "3"]):
            add("diagnose", *args)
    for value in ("0", "4294967295", "4294967296", "-1", "+1", "1x"):
        add("diagnose", "--warmup=" + value)
    add("diagnose", "bubo", "diagnose", "--section=inventory")
    add("diagnose", "--help", "--warmup=-1")
    for args in (["--json", "--preview"], ["--json", "--bundle", "out.zip"],
                 ["--json", "--bundle=out.zip", "--preview"], ["--json", "--bundle="],
                 ["--json", "--preview", "--preview"], ["--json", "--json", "--preview"],
                 ["--json", "--bundle", "a", "--bundle=b"], ["--json", "--bundle"],
                 ["--json", "--bundle", "--preview"], ["--jsonl", "--preview"],
                 ["--json", "--bundle", "x" * 4097]):
        add("bundle", *args)
    for flag in ("--log-file", "--log-max-bytes", "--log-retain"):
        for value in ("", "0", "1", "16", "17", "-1", "x", "18446744073709551615", "18446744073709551616"):
            add("serve", flag, value)
            add("serve", flag + "=" + value)
        add("serve", flag)
        add("serve", flag, "1", flag + "=2")
    add("serve", "--jsonl", "--log-file=fixture.log", "llm", "--model", "unused", "--log-max-bytes=1024", "--log-retain", "2")
    add("serve", "--model", "--log-file", "value", "--jsonl")
    add("serve", "--jsonl=true")
    return rows


def process_cases():
    return [
        ["status", "--help"], ["status"], ["status", "--json", "--bad"],
        ["status", "--json", "--watch", "--watch"], ["status", "--json", "--instance-id="],
        ["stop"], ["stop", "--json"], ["stop", "--json", "--wait-ms", "4294967295"],
        ["stop", "--json", "--instance-id=" + ID],
        ["models", "list", "--help"], ["models", "estimate", "--json"],
        ["models", "list", "--scan-folder="], ["models", "estimate", "--model", "x", "--context", "-1", "--json"],
        ["prepare", "--help"], ["prepare", "--yes", "model", "--jsonl"],
        ["prepare", "--output=a", "--json"], ["prepare", "--output", "a", "--output", "b", "--jsonl"],
        ["diagnose", "--help"], ["diagnose", "--warmup"],
        ["diagnose", "--json", "--warmup=-1"], ["diagnose", "--json", "--section=nope"],
        ["diagnose", "--json", "--unknown"], ["diagnose", "--json", "--bundle="],
        ["diagnose", "--json", "--preview", "--preview"],
        ["diagnose", "--jsonl", "--preview"], ["diagnose", "--json", "--bundle", "--preview"],
        ["diagnose", "--json", "--preview=true"],
        ["diagnose", "--output", "--bundle", "a", "--json"],
        ["serve", "--jsonl", "--help", "--log-retain=bogus"],
        ["serve", "--json", "--help"], ["serve", "--list-switches", "--json", "--extra"],
        ["help", "prepare", "extra"], ["prepare", "-h"],
        ["status", "--json", "--instance-id=" + ID],
        ["status", "--json", "--instance-id", ID],
        ["stop", "--json", "--instance-id", ID, "--wait-ms", "0"],
        ["stop", "--json", "--instance-id", ID, "--wait-ms", "4294967294"],
    ]


# Every engine command with a help text: argv before --help. The same list
# feeds tools/cli/gen_offline_help.py, which writes src/cli/offline_help.inc.
HELP_TOPICS = [
    ["models"], ["models", "list"], ["models", "estimate"],
    ["serve"], ["serve", "llm"], ["serve", "image"], ["serve", "video"],
    ["serve", "tts"], ["serve", "asr"],
    ["prompt"], ["chat"], ["bench"], ["eval"], ["video"], ["transcribe"],
    ["diagnose"], ["probe"],
]


def offline_cases():
    """Help of every engine command, run by a bubo.exe that has no engine DLL and no runtime."""
    return [[*topic, "--help"] for topic in HELP_TOPICS] + [["help", "serve", "llm"], ["help", "models"]]


def run(exe, args, cwd, env):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    p = subprocess.run([str(exe), *args], cwd=cwd, env=env, input=b"", capture_output=True, timeout=15, creationflags=flags)
    out = p.stdout.decode("utf-8", errors="strict").replace("\r\n", "\n")
    err = p.stderr.decode("utf-8", errors="strict").replace("\r\n", "\n")
    return p.returncode, out, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--record", choices=("pure", "process", "machine", "offline"))
    parser.add_argument("--only", choices=("pure", "process", "machine", "offline"))
    args = parser.parse_args()
    golden = Path(__file__).with_name("argparse.golden.json")
    expected = json.loads(golden.read_text(encoding="utf-8")) if golden.exists() else {}
    actual = {}
    with tempfile.TemporaryDirectory(prefix="bubo-argparse-") as temporary:
        cwd = Path(temporary)
        env = dict(os.environ, LOCALAPPDATA=str(cwd / "local"), BUBO_RUNTIME_DIR=str(cwd / "runtime"))
        env.pop("BUBO_MANIFEST", None)
        env.pop("BUBO_RUNTIME_KIND", None)
        groups = [args.only] if args.only else ["pure", "process", "machine", "offline"]
        bare = None
        if "offline" in groups:
            # bubo.exe and its own libraries in an empty folder: no bubo-engine.dll, no AMD runtime.
            bare = cwd / "bare" / args.cli.name
            bare.parent.mkdir()
            shutil.copy2(args.cli, bare)
            for name in ("z.dll", "libcrypto-3-x64.dll", "libcurl.dll"):  # what bubo.exe itself imports
                shutil.copy2(args.cli.parent / name, bare.parent / name)
        for group in groups:
            if group == "pure":
                rows = cases()
            elif group == "process":
                rows = process_cases()
            elif group == "offline":
                rows = offline_cases()
            else:
                rows = []
                for r in expected["pure"]:
                    if r["exit"] != 2:
                        continue
                    command, *options = r["argv"]
                    if "--json" not in options and "--jsonl" not in options:
                        flag = "--jsonl" if command == "serve" else "--json"
                        first = 1 if command in ("prepare", "models") and options and not options[0].startswith("-") else 0
                        options.insert(first, flag)
                    rows.append(["run-" + command, *options])
            results = []
            for row in rows:
                exe = args.cli if group == "process" else bare if group == "offline" else args.driver
                code, out, err = run(exe, row, cwd, env)
                if group == "pure":
                    result = {"argv": row, "exit": code, "parsed": json.loads(out), "stderr": err}
                else:
                    if "--json" in row or "--jsonl" in row:
                        records = [json.loads(line) for line in out.splitlines()]
                        if group == "machine":
                            assert code == 2 and any(r["type"] == "error" and r["payload"]["code"] == "invalid_arguments" for r in records), (row, code, records)
                        if records and records[0]["type"] == "starting":
                            identity = records[0]["payload"]["identity"]
                            assert set(identity) == {"instance_id", "pid", "process_creation_time", "exe_path"}
                            assert len(identity["instance_id"]) == 32 and all(c in "0123456789abcdef" for c in identity["instance_id"])
                            assert isinstance(identity["pid"], int) and identity["pid"] > 0
                            assert identity["process_creation_time"].isdigit() and identity["exe_path"]
                            assert all(r["payload"]["identity"] == identity for r in records)
                            for r in records:
                                del r["payload"]["identity"]
                        # Process identity/time/revision varies; ordering and argument/error payloads are frozen.
                        out = [{"type": r["type"], "seq": r["seq"], "payload": r["payload"]} for r in records]
                    result = {"argv": row, "exit": code, "stdout": out, "stderr": err}
                if group == "offline":
                    # The help printed without the engine is the help the engine prints.
                    reference = run(args.cli, row, cwd, env)
                    assert code == 0 and (code, out, err) == reference, (row, code, reference[0], err)
                results.append(result)
            actual[group] = results
            if args.record == group:
                expected[group] = results
            else:
                assert len(expected[group]) == len(results), (group, len(expected[group]), len(results))
                for before, after in zip(expected[group], results):
                    assert before == after, json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=False)
            print(f"{group} golden: {len(results)} passed, 0 failed")
    if args.record:
        golden.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
