"""CPU-only recovery of the production rank-one writer edit; never loads HIP.

GGUF contract v1: general.architecture=refusal-direction, refusal.version="1",
refusal.sites=newline-separated exact tensor names, refusal.fitted_strength is
the median through-origin fit (with its range and estimator in string KVs),
and one F32 direction.r[2560].
The filename suffix is SHA256 of the little-endian F32 vector (file SHA printed).
Scratch is on S:. D is streamed in row chunks; randomized subspace iteration
uses a fixed seed and rank 16, followed by a small exact SVD via its Gram matrix.
"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import argparse
from collections import Counter
from enum import IntEnum
import hashlib
import json
from pathlib import Path
import re
import time
import sys

import numpy as np
import psutil
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize
import gguf.gguf_reader as reader_module
from gguf.constants import GGMLQuantizationType

def dequant_q2_0(raw):
    """uint8 encoded rows -> float32 rows; 64 values per 18-byte block.

    Derived from src/core/quant/ggml_dequant.hpp and ggml_dequant.cpp
    (not a Q2_K layout):
      d = fp16 bytes 0..1; qs = bytes 2..17
      code[j] = (qs[j//4] >> (2*(j%4))) & 3
      y[j] = float32(code[j] - 1) * float32(d).
    Flatten byte then low-to-high two-bit lane, preserving C++ signed zeros.
    """
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[1] % 18:
        raise ValueError("Q2_0 requires rows of complete 18-byte blocks")
    blocks = np.ascontiguousarray(raw).reshape(-1, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    codes = (blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    levels = codes.reshape(-1, 64).astype(np.float32) - np.float32(1)
    return (levels * d).reshape(raw.shape[0], raw.shape[1] // 18 * 64)

# Installed gguf-py predates Q2_0. Use the NumPy dequantizer above only for
# that type; all others use gguf.quants.dequantize.
if 42 not in GGMLQuantizationType._value2member_map_:
    ReaderType = IntEnum("ReaderType", [(t.name, int(t)) for t in GGMLQuantizationType] + [("Q2_0", 42)])
    reader_module.GGMLQuantizationType = ReaderType
    GGML_QUANT_SIZES[ReaderType.Q2_0] = (64, 18)

SITES = [f"blk.{i}.{('attn_output' if i % 4 == 3 else 'ssm_out')}.weight" for i in range(48)] + [f"blk.{i}.ffn_down_shexp.weight" for i in range(48)]
EDITED_EXPERTS = (2, 4, 30, 46, 47)

def guard():
    while psutil.virtual_memory().available < 6 * 2**30:
        print("paused: less than 6 GiB free RAM", flush=True)
        time.sleep(5)

def load(paths):
    readers = [GGUFReader(p) for p in paths]
    tensors = {}
    for reader in readers:
        for t in reader.tensors:
            if t.name in tensors: raise ValueError(f"duplicate tensor: {t.name}")
            tensors[t.name] = t
    return readers, tensors

def rows(t, start, stop, expert=0):
    guard()
    cols, nrows = map(int, t.shape[:2])
    block, size = GGML_QUANT_SIZES[t.tensor_type]
    data = t.data.reshape(-1, cols // block * size)
    raw = data[expert*nrows+start:expert*nrows+stop]
    out = dequant_q2_0(raw) if int(t.tensor_type) == 42 else dequantize(raw, t.tensor_type)
    return out.astype(np.float32, copy=False)

def matrix(t, expert=0):
    out = np.empty((int(t.shape[1]), int(t.shape[0])), np.float32)
    for start in range(0, len(out), 128):
        out[start:start+128] = rows(t, start, min(start+128, len(out)), expert)
    return out

def stats(h, w, r):
    a, b = r.astype(np.float64) @ h, r.astype(np.float64) @ w
    denom = np.linalg.norm(b)
    signed = float(np.dot(a, b)/np.dot(b, b))
    return {"delta_ratio": float(np.linalg.norm(a-b)/denom),
            "remaining_ratio": float(np.linalg.norm(a)/denom),
            "signed_remaining": signed, "fitted_strength": 1.0-signed}

def fit_summary(dense):
    if len(dense) != len(SITES) or {item["name"] for item in dense} != set(SITES):
        raise ValueError("fit requires each of the 96 writer sites exactly once")
    for item in dense:
        item["fitted_strength"] = 1.0-float(item["signed_remaining"])
    values = np.array([item["fitted_strength"] for item in dense])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 4):
        raise ValueError("site fit is nonfinite or outside [0, 4]")
    by_role = {}
    for role in ("attn_output", "ssm_out", "ffn_down_shexp"):
        group = [item["fitted_strength"] for item in dense if item["name"].split(".")[2] == role]
        by_role[role] = dict(zip(("min", "median", "max"), map(float, np.quantile(group, [0, .5, 1]))))
    return {"estimator": "median of 96 per-site through-origin regressions",
            "strength": float(np.median(values)),
            "range": [float(values.min()), float(values.max())], "by_role": by_role}

def write_direction(outdir, r, report):
    fit = fit_summary(report["edited"])
    r = np.asarray(r, dtype=np.float32).reshape(-1)
    if r.shape != (2560,) or not np.isfinite(r).all() or abs(float(r @ r)-1.0) > 1e-5:
        raise ValueError("direction must be a finite 2560-wide unit vector")
    suffix = hashlib.sha256(r.astype("<f4").tobytes()).hexdigest()[:8]
    out = outdir / f"refusal-r-{suffix}-fitted.gguf"
    writer = GGUFWriter(out, "refusal-direction")
    writer.add_string("refusal.version", "1")
    writer.add_string("refusal.sites", "\n".join(SITES))
    writer.add_string("refusal.evidence", "unit-projection-match" if report["unit_projection_matches"] else "non-unit-edit-observed")
    writer.add_string("refusal.fitted_strength", repr(fit["strength"]))
    writer.add_string("refusal.fitted_strength_min", repr(fit["range"][0]))
    writer.add_string("refusal.fitted_strength_max", repr(fit["range"][1]))
    writer.add_string("refusal.fitted_strength_estimator", fit["estimator"])
    writer.add_tensor("direction.r", r)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    report["fitted_strength"] = fit
    print("fitted strength", json.dumps(fit), flush=True)
    print("direction", out, "sha256", hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
    return out

def distribution(items):
    return {k: dict(zip(("min", "p05", "median", "p95", "max"),
                       map(float, np.quantile([x[k] for x in items], [0, .05, .5, .95, 1]))))
            for k in ("delta_ratio", "remaining_ratio", "signed_remaining")}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--huihui-dir", type=Path, help="directory with the production Huihui *-of-*.gguf shards (full derivation)")
    p.add_argument("--ista", type=Path, help="shard 1 of the IQ3_S ISTA model (full derivation)")
    p.add_argument("--cross-check", type=Path, help="shard 1 of the IQ3_XXS ISTA model (full derivation)")
    p.add_argument("--out", type=Path, required=True, help="output directory for the direction file and reports")
    p.add_argument("--scratch", type=Path, help="scratch directory for the D matrices (full derivation)")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--fit-existing", action="store_true", help="fit from the saved derivation and existing unit vector without rereading model weights")
    p.add_argument("--direction-file", type=Path, help="existing version-1 direction file (--fit-existing)")
    a = p.parse_args()
    if a.fit_existing:
        if a.direction_file is None: p.error("--fit-existing requires --direction-file")
    else:
        for flag in ("huihui_dir", "ista", "cross_check", "scratch"):
            if getattr(a, flag) is None: p.error("--" + flag.replace("_", "-") + " is required without --fit-existing")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.scratch is not None: a.scratch.mkdir(parents=True, exist_ok=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if a.fit_existing:
        report = json.loads((a.out / "derivation.json").read_text())
        reader = GGUFReader(a.direction_file)
        tensors = [tensor for tensor in reader.tensors if tensor.name == "direction.r"]
        def field(name):
            value = reader.get_field(name)
            return value.contents() if value else None
        if (field("general.architecture") != "refusal-direction" or
                field("refusal.version") != "1" or
                field("refusal.sites") != "\n".join(SITES) or
                len(reader.tensors) != 1 or len(tensors) != 1 or
                tensors[0].tensor_type.name != "F32" or list(tensors[0].shape) != [2560]):
            raise ValueError("existing file must be a version-1 direction with the 96 sites and one F32 direction.r")
        write_direction(a.out, tensors[0].data, report)
        (a.out / "derivation-fitted.json").write_text(json.dumps(report, indent=2))
        return
    hr, ht = load(sorted(a.huihui_dir.glob("*-of-*.gguf")))
    ir, it = load([a.ista])
    xr, xt = load([a.cross_check])
    names = SITES + [f"blk.{i}.ffn_down_exps.weight" for i in EDITED_EXPERTS]
    inventory = [{"name": n, "huihui": ht[n].tensor_type.name, "ista": it[n].tensor_type.name,
                  "cross_check": xt[n].tensor_type.name, "shape": list(map(int, ht[n].shape))} for n in names]
    for item in inventory: print(json.dumps(item), flush=True)
    print("inventory counts", Counter((x["huihui"], x["ista"]) for x in inventory), flush=True)
    (a.out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    if a.inventory_only: return
    rng = np.random.default_rng(26030)
    rank = 16
    y = np.zeros((2560, rank), np.float64)
    paths = []
    for n in SITES:
        if ht[n].tensor_type.name != "Q8_0" or int(ht[n].shape[1]) != 2560:
            raise ValueError(f"unexpected edited tensor {n}")
        h, w = matrix(ht[n]), matrix(it[n])
        d = h-w
        path = a.scratch / (n + ".npy")
        np.save(path, d); paths.append(path)
        omega = rng.standard_normal((d.shape[1], rank), dtype=np.float32)
        y += d @ omega
        print("sketch", n, "rss_GiB", psutil.Process().memory_info().rss/2**30, flush=True)
        del h, w, d, omega
    # One power iteration: top left subspace of D D^T, streamed per tensor.
    q, _ = np.linalg.qr(y)
    y.fill(0)
    for path in paths:
        guard(); d = np.load(path, mmap_mode="r")
        y += d @ (d.T @ q)
        del d
    q, _ = np.linalg.qr(y)
    gram = np.zeros((rank, rank), np.float64)
    for path in paths:
        guard(); d = np.load(path, mmap_mode="r")
        b = q.T @ d; gram += b @ b.T
        del b, d
    eig, u = np.linalg.eigh(gram)
    r = (q @ u[:, -1]).astype(np.float32)
    r /= np.linalg.norm(r)
    if r[np.argmax(abs(r))] < 0: r *= -1
    np.save(a.scratch / "direction-candidate.npy", r)
    singular = np.sqrt(np.maximum(eig[::-1], 0))
    print("singular values", singular.tolist(), "gap", singular[0]/singular[1], flush=True)
    dense, control, cross, real = [], [], [], []
    for n in SITES:
        h, w = matrix(ht[n]), matrix(it[n])
        item = {"name": n, **stats(h, w, r)}; dense.append(item)
        print("edited", json.dumps(item), flush=True)
        # Independent random-input output check for each writer role.
        role = n.split(".")[2]
        if role not in [v["role"] for v in real]:
            x = rng.standard_normal(w.shape[1]).astype(np.float32)
            yh, yi = h @ x, w @ x
            yp = yi - r * (r.astype(np.float64) @ yi)
            real.append(dict(role=role, name=n, baseline_l2=float(np.linalg.norm(yh-yi)),
                             projected_l2=float(np.linalg.norm(yh-yp)),
                             baseline_direction=abs(float(r @ (yh-yi))),
                             projected_direction=abs(float(r @ (yh-yp)))))
        del h, w
        # Cross-quant consistency of statistics on all 96 sites.
        h, w = matrix(ht[n]), matrix(xt[n])
        cross.append({"name": n, **stats(h, w, r)}); del h, w
    for i in range(48):
        if i in EDITED_EXPERTS: continue
        n = f"blk.{i}.ffn_down_exps.weight"
        if ht[n].tensor_type.name != "Q5_1": raise ValueError(f"control type changed: {n}")
        for expert in (0, 173, 511):
            h, w = matrix(ht[n], expert), matrix(it[n], expert)
            item = {"name": n, "expert": expert, **stats(h, w, r)}
            control.append(item); print("control", json.dumps(item), flush=True)
            del h, w
    disagreements = [v for v in dense if v["delta_ratio"] < .7 or v["remaining_ratio"] > .3]
    separated = min(v["delta_ratio"] for v in dense) > max(v["delta_ratio"] for v in control)
    remaining_separated = max(v["remaining_ratio"] for v in dense) < min(v["remaining_ratio"] for v in control)
    report = dict(singular_values=singular.tolist(), singular_gap=float(singular[0]/singular[1]),
                  edited=dense, controls=control, cross_check=cross, real_tensor=real,
                  distributions={"edited": distribution(dense), "control": distribution(control),
                                 "cross_check": distribution(cross)},
                  disagreements=disagreements, delta_separated=separated,
                  remaining_separated=remaining_separated, unit_projection_matches=not disagreements,
                  peak_rss_GiB=psutil.Process().memory_info().peak_wset/2**30)
    report["fitted_strength"] = fit_summary(dense)
    (a.out/"derivation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("distributions", "real_tensor", "disagreements", "delta_separated", "remaining_separated", "peak_rss_GiB")}, indent=2), flush=True)
    if not separated:
        raise ValueError("direction does not distinguish dense edits from controls; inspect derivation.json")
    if disagreements:
        print("FINDING: unit projection does not reproduce the measured Huihui edit. "
              "Emitting the unit direction for the explicitly requested idempotent runtime projection; "
              "all disagreements remain in derivation.json.", flush=True)
    write_direction(a.out, r, report)

if __name__ == "__main__": main()
