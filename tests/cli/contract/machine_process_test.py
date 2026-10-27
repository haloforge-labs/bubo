"""Real-process proof for the read-only machine fields in `bubo status --json`.

Usage: python machine_process_test.py PATH_TO_BUBO --work-root DIR
The test reads this computer. It passes with or without ADLX, an AMD driver or
a battery. It checks the field shapes and the unavailable-with-reason rule.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

READING_KEYS = {"value", "source", "api_version", "unit", "sample_time",
                "available", "reason"}


def run(exe, root, *args):
    env = dict(os.environ, LOCALAPPDATA=str(root / "local"))
    env.pop("BUBO_MANIFEST", None)
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
             if os.name == "nt" else 0)
    result = subprocess.run([str(exe), *args], cwd=root, env=env, input=b"",
                            capture_output=True, timeout=30,
                            creationflags=flags)
    assert result.returncode == 0, (args, result.returncode, result.stderr)
    assert result.stdout.endswith(b"\n") and result.stdout.count(b"\n") == 1
    assert not result.stdout.startswith(b"\xef\xbb\xbf") and b"\r" not in result.stdout
    return json.loads(result.stdout.decode("utf-8"))


def readings(node, path, found):
    if isinstance(node, dict):
        if "available" in node:
            assert READING_KEYS <= set(node), (path, sorted(node))
            found.append(path)
            if node["available"]:
                assert node["value"] is not None and node["reason"] is None, path
            else:
                assert node["value"] is None, path
                assert isinstance(node["reason"], str) and node["reason"], path
            assert node["source"] and node["sample_time"].endswith("Z"), path
        for key, child in node.items():
            readings(child, path + "." + key, found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exe")
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()
    root = Path(args.work_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    passed = 0
    first = run(args.exe, root, "status", "--json")
    machine = first["payload"]["machine"]
    assert machine["read_only"] is True
    passed += 1
    found = []
    readings(machine, "machine", found)
    assert len(found) == 20, found
    passed += 1
    mode = machine["windows"]["effective_power_mode"]
    assert mode["validated_mapping"] is None
    assert mode["reference_observation"]["confidence"] == \
        "harness-observed, ASUS PX13 only"
    assert mode["reference_observation"]["validated"] is False
    if mode["available"]:
        assert isinstance(mode["raw_value"], int) and mode["raw_value"] == mode["value"]
    else:
        assert mode["raw_value"] is None
    passed += 1
    carved = machine["variable_graphics_memory"]["carved"]
    assert carved["unit"] == "bytes" and carved["raw_unit"] == "GB"
    if carved["available"]:
        assert carved["value"] == round(carved["raw_value"] * 2**30)
        assert abs(carved["gib"] - carved["raw_value"]) < 1e-9
    passed += 1
    dxgi = machine["dxgi"]
    for name in ("dedicated_video_memory", "shared_system_memory", "budget",
                 "current_usage", "available_for_reservation"):
        assert dxgi[name]["unit"] == "bytes", name
        if dxgi[name]["available"]:
            assert dxgi[name]["gib"] == dxgi[name]["value"] / 2**30, name
        else:
            assert dxgi[name]["gib"] is None, name
    assert dxgi["dedicated_video_memory"]["source"] == \
        "DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory"
    assert dxgi["budget"]["source"] == "IDXGIAdapter3::QueryVideoMemoryInfo"
    assert dxgi["agree_tolerance_bytes"] == 1048576
    if dxgi["dedicated_video_memory"]["available"] and carved["available"]:
        assert dxgi["agree"] == (abs(dxgi["dedicated_video_memory"]["value"]
                                     - carved["value"]) <= 1048576)
    else:
        assert dxgi["agree"] is None and dxgi["agree_reason"]
    passed += 1
    assert machine["recipe_suggestion"]["text"] == \
        "No tested profile is available for this memory setting."
    assert machine["recipe_suggestion"]["tested_profile"] is None
    passed += 1
    # A second read must not change anything it reported (it writes nothing).
    for row_name in ("overlay_scheme",):
        before = machine["windows"][row_name]["value"]
        row = run(args.exe, root, "status", "--json")["payload"]["machine"]
        assert row["windows"][row_name]["value"] == before
    passed += 1
    assert first["payload"].get("capabilities", {}).get("machine_readings") is True
    passed += 1
    (root / "status-machine.stdout.jsonl").write_text(
        json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Machine fields in status --json: {passed} passed, 0 failed")


main()
