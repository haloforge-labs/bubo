"""Independent random-input check of three real writer roles, CPU only.

Usage: python -B refusal_real_test.py --direction S:/.../refusal-r-<sha8>.gguf
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derive_refusal_r as d
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", type=Path, required=True)
    a=p.parse_args()
    reader=d.GGUFReader(a.direction)
    t,=reader.tensors
    assert t.name=="direction.r" and t.tensor_type.name=="F32" and list(t.shape)==[2560]
    fitted=float(reader.get_field("refusal.fitted_strength").contents())
    assert np.isfinite(fitted) and 0 <= fitted <= 4
    r=t.data.copy().reshape(-1)
    assert abs(float(r.astype(np.float64) @ r)-1)<1e-5
    hr,ht=d.load(sorted(d.HUI.glob("*-of-*.gguf")))
    ir,it=d.load([d.ISTA])
    rng=np.random.default_rng(99131)
    for name in ("blk.0.ssm_out.weight","blk.3.attn_output.weight","blk.0.ffn_down_shexp.weight"):
        h,w=d.matrix(ht[name]),d.matrix(it[name])
        x=rng.standard_normal((w.shape[1],8)).astype(np.float32)
        yh,yi=h@x,w@x
        projected=r[:,None]*(r.astype(np.float64)@yi)[None,:]
        unit=yi-projected
        fit=yi-fitted*projected
        before,one,after=(np.linalg.norm(yh-v) for v in (yi,unit,fit))
        rb,ru,rf=(np.linalg.norm(r@(yh-v)) for v in (yi,unit,fit))
        print(f"{name}: L2 baseline={before:.9g} unit={one:.9g} fitted={after:.9g}; direction baseline={rb:.9g} unit={ru:.9g} fitted={rf:.9g}",flush=True)
        assert after<one<before and rf<ru<rb, "fitted projection did not improve the real tensor comparison"
        del h,w
    print(f"PASS three real roles; fitted strength={fitted:.9g}")

if __name__=="__main__": main()
