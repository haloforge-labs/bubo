"""CPU-only qualification import-policy controls; no device or DLL load."""

import importlib.util
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
source = Path(__file__).resolve().parents[2] / "tools/npu/qualification.py"
if len(sys.argv) > 1:
    source = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("qualification", source)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
symbols = ["symbol" + str(i) for i in range(26)]
edge = {"Name": "xrt_coreutil.dll", "Kind": "delay"}
q.validate_imports("ON", {"Edges": [edge]}, symbols, symbols)
q.validate_imports("OFF", {"Edges": []}, [], symbols)
print("PASS: ON delay-only and OFF no-XRT positive controls")
variants = [
    ("OFF-edge", "OFF", [edge], []),
    ("OFF-symbol", "OFF", [], symbols),
    ("ON-static-import", "ON", [{**edge, "Kind": "import"}], symbols),
    ("ON-missing-edge", "ON", [], symbols),
    ("ON-duplicate-edge", "ON", [edge, edge], symbols),
    ("ON-other-XRT", "ON", [{**edge, "Name": "xrt_core.dll"}], symbols),
    ("ON-symbol-drift", "ON", [edge], symbols[:-1]),
    ("ON-unresolved-symbol", "ON", [edge], symbols[:-1] + ["missing"]),
    ("ON-abort", "ON", [edge], symbols[:-1] + ["?abort@run@xrt"]),
]
for name, mode, edges, variant_symbols in variants:
    try:
        q.validate_imports(mode, {"Edges": edges}, variant_symbols,
                           symbols + ["?abort@run@xrt"])
    except ValueError as error:
        print(f"PASS: failing variant {name} rejected: {error}")
    else:
        raise AssertionError("variant unexpectedly accepted: " + name)

policy = json.loads((source.parent / "os-import-dispositions.json").read_text())
edges = [{"From": d["from"], "Name": d["name"], "Kind": d["kind"], "Resolution": "missing"}
         for d in policy["dispositions"]]
nodes = [{"Path": d["from"], "Sha256": d["parent_sha256"]} for d in policy["dispositions"]]
closure = {"Nodes": nodes, "Edges": edges}
assert len(q.disposition_imports(closure, policy)) == 5
print("PASS: all five pinned OS delay edges dispositioned; raw evidence retained")
for case in ("normal", "new-name", "new-parent", "hash-drift"):
    mutant = copy.deepcopy(closure)
    if case == "normal": mutant["Edges"][0]["Kind"] = "import"
    if case == "new-name": mutant["Edges"][0]["Name"] = "Unknown.dll"
    if case == "new-parent": mutant["Edges"][0]["From"] = "C:/Windows/System32/other.dll"
    if case == "hash-drift":
        for node in mutant["Nodes"]: node["Sha256"] = "0" * 64
    try:
        q.disposition_imports(mutant, policy)
    except ValueError as error:
        print(f"PASS: OS disposition mutant {case} refused: {error}")
    else:
        raise AssertionError(case)
