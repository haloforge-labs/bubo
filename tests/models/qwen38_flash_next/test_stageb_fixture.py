"""Independent fixture and token-zero addressing challenges; CPU only."""
import json
import math
from pathlib import Path
import struct
import unittest

from stageb_reference import reference

HERE = Path(__file__).resolve().parent


def pool(rows, start, stride, field, k='3'):
    return math.fsum(row['expected']['totals']['k'][k][field]
                     for t, row in enumerate(rows) if (start + t) % stride == 0)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((HERE / 'stageb-synthetic.json').read_bytes())

    def test_saved_expectations_match_independent_cpu_oracle(self):
        for case in self.cases:
            for row in case.get('tokens', [case]):
                self.assertEqual(row['expected'], reference(
                    row['experts'], row['weights'], row['inject'], row['parts'], row['ids']))

    def test_multirow_rejects_token_zero_addressing(self):
        multirow = [c['tokens'] for c in self.cases if 'tokens' in c]
        self.assertTrue(multirow, 'repeated-row fixtures cannot detect token-zero reads')
        rows = multirow[0]
        self.assertEqual(len(rows), 33)
        for field in ('experts', 'weights', 'ids', 'inject'):
            self.assertEqual(len({json.dumps(r[field]) for r in rows}), len(rows), field)
        for row in rows:
            for expert in row['experts']:
                for value in expert:
                    self.assertEqual(struct.unpack('e', struct.pack('e', value))[0], value)
            for value in row['weights'] + row['inject']:
                self.assertEqual(struct.unpack('f', struct.pack('f', value))[0], value)
        # Deliberately wrong CPU implementation: preserve output token/count,
        # but obtain every numerical input from token zero, as the old fixture
        # accidentally allowed. Test actual absolute accumulation positions.
        wrong = [rows[0]] * len(rows)
        repeated = [self.cases[0]] * len(rows)
        for stride in (1, 3, 128):
            self.assertEqual(pool(repeated, 101, stride, 'rr'),
                             pool([repeated[0]] * 33, 101, stride, 'rr'))
            for field in ('rf', 'rr', 'ff', 'error2', 'omitted_mass'):
                self.assertNotEqual(pool(rows, 101, stride, field),
                                    pool(wrong, 101, stride, field), (stride, field))
        for t in range(33):
            if (101 + t) % 4 == 0:
                for field in ('gates', 'gate_l2', 'gate_max', 'f_norm'):
                    self.assertNotEqual(rows[t]['expected'][field],
                                        rows[0]['expected'][field], (t, field))


if __name__ == '__main__':
    unittest.main()
