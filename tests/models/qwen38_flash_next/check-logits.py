"""CPU-only validation of saved FP32 logits and optional byte comparison."""
import argparse
import array
import hashlib
import math
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('dump', type=Path)
p.add_argument('--compare', type=Path)
p.add_argument('--allow-difference', action='store_true')
a = p.parse_args()

def inspect(path):
    data = path.read_bytes()
    if not data or len(data) % 4:
        p.error(f'empty or non-FP32 dump: {path}')
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        p.error(f'nonfinite saved logit: {path}')
    print(f'{path.name}: {len(values)} finite logits, sha256={hashlib.sha256(data).hexdigest()}')
    return data

left = inspect(a.dump)
if a.compare:
    right = inspect(a.compare)
    same = left == right
    print(f'bit_identical={int(same)} expected=1')
    if not same and not a.allow_difference:
        p.error('full-logit bytes differ (both hashes printed above)')
