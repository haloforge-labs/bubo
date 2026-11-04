"""Publication, binary sample and full-prompt coverage failures before checker."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import check_stageb as check
from test_stageb import fixture


def capture(root):
    records = fixture(1)
    rows = bytearray(b'GUFOPFB2')
    for record in records:
        record.update(sample_every=1, row_file='stageb-pass-1-1/rows.bin',
                      row_offset=len(rows), row_count=1, record_bytes=32)
        norms = [math.sqrt(5)] + [math.sqrt(record['k'][str(k)]['error2']) for k in (2,3,4,6)]
        bf16 = []
        for v in norms:
            bits = struct.unpack('<I', struct.pack('<f', v))[0]
            bf16.append((bits + 0x7fff + ((bits>>16)&1)) >> 16)
        rows.extend(struct.pack('<16H', *([32768]*4 + [32768,32768,16384] + bf16 +
                                  [round(record['k'][str(k)]['omitted_mass']*65535) for k in (2,3,4,6)])))
    path = root / 'stageb-pass-1-1/rows.bin'
    path.parent.mkdir()
    path.write_bytes(rows)
    (root / 'stageb-totals.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), newline='\n')
    return records


class CaptureTests(unittest.TestCase):
    def test_complete_and_bad_samples(self):
        for failure in ('none', 'short', 'nonfinite', 'missing', 'orphan', 'count', 'offset'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = capture(root)
                path = root / rows[0]['row_file']
                data = bytearray(path.read_bytes())
                if failure == 'short': path.write_bytes(data[:-1])
                if failure == 'nonfinite': struct.pack_into('<H', data, 22, 0x7f80); path.write_bytes(data)
                if failure == 'missing': path.unlink()
                if failure == 'orphan': (root / 'stageb-pass-2-1').mkdir()
                if failure == 'count': rows[0]['row_count'] = 2
                if failure == 'offset': rows[0]['row_offset'] += 32
                if failure == 'none': self.assertEqual(check.validate_samples(root, rows), 48)
                else:
                    with self.assertRaises((ValueError, OSError)):
                        check.validate_samples(root, rows)

    def test_prompt_coverage_and_final_trim(self):
        rows = fixture(1)
        for row in rows:
            row['tokens'] = row['rows_available'] = row['rows'] = 4096
        self.assertEqual(check.validate_coverage(rows, [4096], allow_final_trim=False), 1)
        rows[47]['row_start'] = 3968
        rows[47]['rows_available'] = rows[47]['rows'] = 128
        self.assertEqual(check.validate_coverage(rows, [4096], allow_final_trim=True), 1)
        with self.assertRaises(ValueError): check.validate_coverage(rows, [4096])
        with self.assertRaises(ValueError): check.validate_coverage(rows, [8192], allow_final_trim=True)
        with self.assertRaises(ValueError): check.validate_coverage(rows, [4096, 4096], allow_final_trim=True)


if __name__ == '__main__': unittest.main()
