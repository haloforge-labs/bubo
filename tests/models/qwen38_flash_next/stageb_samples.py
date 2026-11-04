"""Decode Stage B compact sample records into the original named field order.

UNORM16 gates/max use [0,2], gate L2 [0,4], masses [0,1]. Five
nonnegative norms use BF16 RNE. top4 = 1 - omitted mass at k4. Token and
layer are losslessly carried by per-layer metadata, rather than repeated.
Totals/scalars never use this sample-only quantization.
"""
import math
import struct

ROW = struct.Struct('<16H')
MAGIC = b'GUFOPFB2'


def decode(data, offset, *, token, layer):
    q = ROW.unpack_from(data, offset)
    gates = [x*2/65535 for x in q[:4]]
    norms = [struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in q[7:12]]
    if any(not math.isfinite(x) for x in norms):
        raise ValueError('nonfinite BF16 sample norm')
    omitted = [x/65535 for x in q[12:]]
    return (token, layer, *gates, q[4]*4/65535, q[5]*2/65535,
            q[6]/65535, 1-omitted[2], *norms, *omitted)
