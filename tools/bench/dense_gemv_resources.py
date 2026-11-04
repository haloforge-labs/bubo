#!/usr/bin/env python3
"""Extract compiler resource evidence from HIP device assembly, without a GPU.

Use the production compile command with --cuda-device-only -S. Occupancy is
LLVM's static upper bound in waves/SIMD, NOT measured resident occupancy.
Instruction counts are static, not traffic or cache-miss measurements.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess


FIELDS = {
    "TotalNumSgprs": "sgpr", "NumVgprs": "vgpr", "ScratchSize": "scratch_bytes",
    "LDSByteSize": "lds_bytes", "Occupancy": "compiler_waves_per_simd",
    "VGPRBlocks": "vgpr_alloc_blocks", "SGPRBlocks": "sgpr_alloc_blocks",
}
META = {
    "sgpr_spill_count": "sgpr_spills", "vgpr_spill_count": "vgpr_spills",
    "wavefront_size": "wave_size", "max_flat_workgroup_size": "max_threads",
}
FORMATS = {23: "IQ4_XS", 20: "IQ4_NL", 12: "Q4_K", 13: "Q5_K",
           7: "Q5_1", 14: "Q6_K", 21: "IQ3_S", 18: "IQ3_XXS", 17: "IQ2_XS"}


def template(row, name):
    match = re.search(r"::" + name + r"<([^>]+)>", row.get("kernel", ""))
    params = [value.strip() for value in match[1].split(",")] if match else None
    # The loads-only flag (bench-only 8th parameter of the reuse kernel) is
    # false in every production specialization; hide it so those match as before.
    if params and name == "mul_mat_vec_dense_reuse" and len(params) == 8 and params[7] == "false":
        params = params[:7]
    return params


def width_table(records):
    """Bind compiled specializations to existing dense dispatch, not timings.

    Baseline moe kernels take N at runtime; their resource counts repeat at
    each width even though launch geometry changes. K=640 NL is a separate
    branch. The IQ4 batch control is the ragged one-tile/eight-wave WMMA path.
    """
    table = []
    for type_id, name in FORMATS.items():
        for n in range(1, 9):
            for variant in ("baseline", "reuse1", "reuse2", "iq4_wmma"):
                if variant == "iq4_wmma" and type_id not in (20, 23):
                    continue
                conditions = ("K!=640", "K=640") if type_id == 20 and n > 1 and variant == "baseline" else ("all supported K",)
                for condition in conditions:
                    matches = []
                    threads = 128
                    for row in records:
                        if variant.startswith("reuse"):
                            params = template(row, "mul_mat_vec_dense_reuse")
                            expected = [str(type_id), str(n), "1" if variant == "reuse1" else "2"]
                            if params and params[:3] == expected and (len(params) == 3 or params[3:] == ["false", "false", "1", "4"]):
                                matches.append(row)
                        elif variant == "iq4_wmma":
                            params = template(row, "mul_mat_iq4_decode_batch")
                            if params == [f"(qfn_mmq::ggml_type){type_id}", "1", "true", "8"]:
                                matches.append(row)
                            threads = 256
                        elif type_id in (20, 23):
                            tile = 4 if condition == "K=640" else 2
                            params = template(row, "mul_mat_vec_iq4_dense")
                            if params == [f"(qfn_mmq::ggml_type){type_id}", str(n), str(tile), "4"]:
                                matches.append(row)
                        else:
                            params = template(row, "mul_mat_vec_q_moe")
                            real_type = 4 if type_id == 7 else type_id
                            iq_variant = "1" if type_id in (17, 18, 21) else "0"
                            if params and len(params) == 4 and params[:3] == [f"(qfn_mmq::ggml_type){real_type}", "2", "false"] and params[3].endswith(")" + iq_variant):
                                matches.append(row)
                            threads = 32 * min(n, 4 if type_id in (12, 13) else 8)
                    if len(matches) != 1:
                        raise ValueError(f"expected one resource binding: {name} {variant} N={n} {condition}, found {len(matches)}")
                    table.append(dict(format=name, variant=variant, N=n, condition=condition,
                                      pass_index=1, pass_width=n, launch_threads=threads, **matches[0]))
    return table


def candidate_width_table(records):
    specializations = []
    for row in records:
        params = template(row, "mul_mat_vec_dense_reuse")
        if params and len(params) == 7:
            specializations.append((params, row))
    if not any(p[3] == "true" for p, _ in specializations):
        return []  # Step (a)'s untouched source has no candidates.
    table = []
    settings = {
        "reuse1_cap6": ["false", "false", "1", "4"],
        "reuse1_lds": ["true", "false", "1", "4"],
        "reuse1_ldsgrid": ["false", "true", "1", "4"],
        "reuse1_splitk_lds": ["true", "false", "4", "8"],
    }
    for type_id in (21, 18, 17):
        for variant, flags in settings.items():
            for n in range(1, 9):
                widths = [4, n - 4] if variant == "reuse1_cap6" and n > 6 else [n]
                for index, width in enumerate(widths, 1):
                    expected = [str(type_id), str(width), "1", *flags]
                    matches = [r for p, r in specializations if p == expected]
                    if len(matches) != 1:
                        raise ValueError(f"expected one candidate specialization: {expected}")
                    table.append(dict(format=FORMATS[type_id], variant=variant, N=n,
                                      condition="all supported K", pass_index=index, pass_width=width,
                                      launch_threads=1024 if flags[2] == "4" else 128, **matches[0]))
    return table


def extract(path):
    records = {}
    current = None
    metadata = False
    pending = {}
    # Stream the assembly: debug information can make this hundreds of MB.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            begin = re.search(r"-- Begin function (\S+)", line)
            if begin:
                current = begin[1]
                records[current] = dict(symbol=current, global_load_ops=0, lds_read_ops=0,
                                        lds_write_ops=0, wmma_ops=0, barrier_ops=0)
            if current:
                code_size = re.match(r"; codeLenInByte = (\d+)", line)
                if code_size:
                    records[current]["code_bytes"] = int(code_size[1])
                for key, field in FIELDS.items():
                    value = re.match(r"; " + key + r": (\d+)", line)
                    if value:
                        records[current][field] = int(value[1])
                if re.match(r"\s+(?:global|flat|buffer)_load_", line):
                    records[current]["global_load_ops"] += 1
                # gfx11 names LDS accesses ds_load_* / ds_store_*; older ISA ds_read_* / ds_write_*.
                if re.match(r"\s+ds_(?:read|load)", line):
                    records[current]["lds_read_ops"] += 1
                if re.match(r"\s+ds_(?:write|store)", line):
                    records[current]["lds_write_ops"] += 1
                if re.match(r"\s+v_wmma_", line):
                    records[current]["wmma_ops"] += 1
                if re.match(r"\s+s_barrier\b", line):
                    records[current]["barrier_ops"] += 1
            if line.strip() == ".amdgpu_metadata":
                metadata = True
            if metadata:
                if re.match(r"  - \.args:", line):
                    pending = {}
                item = re.match(r"    \.(\w+):\s+(\S+)", line)
                if item:
                    pending[item[1]] = item[2]
                    if item[1] == "workgroup_processor_mode":
                        name = pending.get("name")
                        if name in records:
                            for key, field in META.items():
                                if key in pending:
                                    records[name][field] = int(pending[key])
    return list(records.values())


def kquant_width_table(records):
    """The candidate takes N at runtime; widths 1..3 retain reuse1."""
    table = []
    present = any(template(r, "mul_mat_dense_kquant_wmma") for r in records)
    if not present:
        return table
    for type_id in (12, 13, 14):
        # The production kernel is the full-stage instantiation (stage 8).
        matches = [r for r in records
                   if template(r, "mul_mat_dense_kquant_wmma") == [str(type_id), str(STAGE_FULL)]]
        if len(matches) != 1:
            raise ValueError(f"expected one K-quant matrix specialization: {type_id}")
        for n in range(4, 9):
            table.append(dict(format=FORMATS[type_id], variant="kquant_wmma", N=n,
                              condition="four large trunk shapes only", pass_index=1,
                              pass_width=n, launch_threads=256, **matches[0]))
    return table


# Bench-only ablation stages of the K-quant matrix-unit kernel (kquant_stage.hpp).
STAGES = {1: "A", 2: "A0", 3: "B", 4: "B0", 5: "C1", 6: "C2", 7: "C3", 8: "D"}
STAGE_FULL = 8


def kquant_stage_table(records):
    """One row per ablation stage and K-quant format; stage D must equal the
    production kernel (same record) in VGPR/SGPR/LDS or the ladder is invalid."""
    table = []
    for type_id in (12, 13, 14):
        for stage, label in STAGES.items():
            matches = [r for r in records
                       if template(r, "mul_mat_dense_kquant_wmma") == [str(type_id), str(stage)]]
            if len(matches) != 1:
                continue
            table.append(dict(format=FORMATS[type_id], variant="stage:" + label, N=0,
                              condition="bench-only ablation", pass_index=1,
                              pass_width=0, launch_threads=256, **matches[0]))
    for type_id in FORMATS:
        for row in records:
            params = template(row, "mul_mat_vec_dense_reuse")
            if params and len(params) == 8 and params[0] == str(type_id) and params[-1] == "true":
                table.append(dict(format=FORMATS[type_id], variant="reuse1_loads", N=1,
                                  condition="bench-only loads rung", pass_index=1,
                                  pass_width=1, launch_threads=128, **row))
    return table


def dot4_table(records):
    """One row per (format, width) of the few-row dot4 kernel (dense_dot4.hpp,
    DenseDot4Kernel<GgmlType, Batch>; one width per translation unit, so one
    assembly holds one width for every instantiated format)."""
    table = []
    for row in records:
        params = template(row, "DenseDot4Kernel")
        if not params or len(params) != 2:
            continue
        type_id = int(re.search(r"(\d+)\s*$", params[0])[1])
        width = int(re.match(r"(\d+)", params[1])[1])
        table.append(dict(format=FORMATS.get(type_id, str(type_id)), variant="dot4", N=width,
                          condition="K % 256 == 0, any M", pass_index=1, pass_width=width,
                          launch_threads=256, **row))
    return table


def linked_metadata(path):
    data = path.read_bytes()
    source = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8")
    records = {}
    for block in source.split("  - .args:")[1:]:
        values = dict(re.findall(r"^    \.(\w+):\s+(\S+)", block, re.MULTILINE))
        if "name" in values:
            records[values["name"]] = values
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output prefix for CSV and JSON")
    parser.add_argument("--cxxfilt", default=shutil.which("llvm-cxxfilt"))
    parser.add_argument("--width-table", action="store_true", help="require and bind all existing N=1..8 variants")
    parser.add_argument("--dot4-table", action="store_true", help="bind the dot4 kernel's (format, width) records (dense_dot4_w*.hip.cpp)")
    parser.add_argument("--metadata", type=Path, help="llvm-readobj --notes output from the built device image")
    args = parser.parse_args()
    rows = extract(args.assembly)
    if args.cxxfilt:
        names = subprocess.check_output(
            [args.cxxfilt], input="\n".join(r["symbol"] for r in rows) + "\n", text=True).splitlines()
        if len(names) != len(rows):
            raise ValueError("demangler output count differs")
        for row, name in zip(rows, names):
            row["kernel"] = name
    rows = [r for r in rows if "vgpr" in r]
    if args.metadata:
        metadata = linked_metadata(args.metadata)
        for row in rows:
            row["resource_source"] = "standalone ISA"
            if row["symbol"] not in metadata:
                continue
            values = metadata[row["symbol"]]
            for src, dst in (("vgpr_count", "vgpr"), ("sgpr_count", "sgpr"),
                             ("group_segment_fixed_size", "lds_bytes"),
                             ("private_segment_fixed_size", "scratch_bytes"),
                             ("vgpr_spill_count", "vgpr_spills"), ("sgpr_spill_count", "sgpr_spills")):
                row[dst] = int(values[src])
            # Occupancy and these encoded allocation fields still come from
            # standalone ISA, not linked metadata. Keep that provenance explicit.
            row["isa_vgpr_alloc_blocks"] = row.pop("vgpr_alloc_blocks", None)
            row["isa_sgpr_alloc_blocks"] = row.pop("sgpr_alloc_blocks", None)
            row["resource_source"] = "linked metadata; occupancy/instruction counts from standalone ISA"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["kernel", "vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills",
              "sgpr_spills", "compiler_waves_per_simd", "vgpr_alloc_blocks", "sgpr_alloc_blocks", "wave_size", "max_threads",
              "global_load_ops", "lds_read_ops", "lds_write_ops", "wmma_ops", "barrier_ops", "code_bytes", "resource_source",
              "isa_vgpr_alloc_blocks", "isa_sgpr_alloc_blocks", "symbol"]
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.width_table:
        bound = width_table(rows) + candidate_width_table(rows) + kquant_width_table(rows) + kquant_stage_table(rows)
        if args.metadata and any(not r.get("resource_source", "").startswith("linked") for r in bound):
            raise ValueError("a width/pass route has no linked-device metadata")
        path = args.out.with_name(args.out.name + "-widths.csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["format", "variant", "N", "condition", "pass_index", "pass_width", "launch_threads", *fields])
            writer.writeheader()
            writer.writerows(bound)
        print(f"Bound {len(bound)} format/variant/width/shape/pass routes to compiler records")
    if args.dot4_table:
        bound = dot4_table(rows)
        if not bound:
            raise ValueError("no DenseDot4Kernel record in this assembly")
        path = args.out.with_name(args.out.name + "-dot4.csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["format", "variant", "N", "condition", "pass_index", "pass_width", "launch_threads", *fields])
            writer.writeheader()
            writer.writerows(bound)
        print(f"Bound {len(bound)} dot4 format/width records")
    print(f"Extracted {len(rows)} compiler resource records; no GPU queried")


if __name__ == "__main__":
    main()
