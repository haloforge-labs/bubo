"""Compact sample codec accuracy/shape tests, independent of GPU arithmetic."""
import math
import struct
import unittest

import stageb_samples as samples


class CodecTests(unittest.TestCase):
    def test_known_layout_and_precision(self):
        # Gates 0,1,2,.5; L2=sqrt(5.25), max2; top1 .4;
        # F/error norms 1,2,4,8,16; omitted masses .4,.3,.2,.1.
        gates = [0, 32768, 65535, 16384]
        encoded = struct.pack('<16H', *gates, round(math.sqrt(5.25)/4*65535),
                              65535, round(.4*65535),
                              *[0x3f80, 0x4000, 0x4080, 0x4100, 0x4180],
                              *[round(m*65535) for m in (.4,.3,.2,.1)])
        self.assertEqual(len(encoded), 32)
        token, layer, *v = samples.decode(encoded, 0, token=101, layer=3)
        self.assertEqual((token, layer), (101, 3))
        for a, b in zip(v[:4], [0,1,2,.5]): self.assertLess(abs(a-b), 2/65535)
        self.assertLess(abs(v[4]-math.sqrt(5.25)), 4/65535)
        self.assertEqual(v[8:13], [1.,2.,4.,8.,16.])
        self.assertLess(abs(v[7]-.8), 1/65535)
        self.assertLess(abs(v[6]-.4), 1/65535)

    def test_bad_size_and_nonfinite_norm(self):
        with self.assertRaises((ValueError, struct.error)):
            samples.decode(b'bad', 0, token=1, layer=0)
        bad = bytearray(32)
        struct.pack_into('<H', bad, 14, 0x7f80)
        with self.assertRaises(ValueError): samples.decode(bad, 0, token=1, layer=0)


if __name__ == '__main__': unittest.main()
