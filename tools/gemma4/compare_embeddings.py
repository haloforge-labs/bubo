r"""Compare the Bubo GPU encoder's stage dumps with llama.cpp b11316's (multislot 2026-10-17). Both sides write u32 dims[4] + f32 data per stage.
Usage: python -B compare_emb.py --gpu DIR --llama DIR [--json OUT]   (DIR = a folder per image with <stage>.f32)
Reports rel RMS = sqrt(sum((gpu-llama)^2) / sum(llama^2)) per stage; for the embedding also the worst per-row cosine and max abs."""
import argparse, json, sys
from pathlib import Path

import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    dims = raw[:16].view(np.uint32)
    width, rows = int(dims[0]), int(dims[1])
    data = raw[16:].view(np.float32)
    if data.size != width * rows:
        # the scalar reference writes width 1152 for every stage; trust the file size
        return data.reshape(1, -1)
    return data.reshape(rows, width)


def rel_rms(a, b):
    return float(np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2) / np.sum(b.astype(np.float64) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    gpu_dir, llama_dir = Path(a.gpu), Path(a.llama)
    names = sorted(p.stem for p in gpu_dir.glob("*.f32") if (llama_dir / p.name).exists())
    order = ["l0.ln1", "l0.attn", "l0.o", "l0.h1", "l0.gate_up", "l0.act", "l0.down"] + [f"layer{i}" for i in range(27)] + ["pooled", "embedding"]
    out = {}
    for name in [n for n in order if n in names]:
        g, l = load(gpu_dir / f"{name}.f32"), load(llama_dir / f"{name}.f32")
        if g.shape != l.shape:
            out[name] = {"error": f"shape gpu {g.shape} vs llama {l.shape}"}
            print(f"{name:12s} SHAPE MISMATCH gpu {g.shape} llama {l.shape}")
            continue
        entry = {"rel_rms": rel_rms(g, l), "max_abs": float(np.max(np.abs(g.astype(np.float64) - l)))}
        if name == "embedding":
            gd, ld = g.astype(np.float64), l.astype(np.float64)
            cos = np.sum(gd * ld, axis=1) / (np.linalg.norm(gd, axis=1) * np.linalg.norm(ld, axis=1))
            entry["worst_row_cosine"] = float(cos.min())
            entry["rows"] = int(g.shape[0])
        out[name] = entry
        print(f"{name:12s} rel rms {entry['rel_rms']:.3e}  max abs {entry['max_abs']:.3e}" + (f"  worst row cosine {entry['worst_row_cosine']:.6f}  rows {entry['rows']}" if name == "embedding" else ""))
    missing = [n for n in order if n not in names]
    if missing:
        print("not compared (absent on one side):", ", ".join(missing))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
