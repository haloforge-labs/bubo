"""CPU calibration oracle/tool challenges; no model or GPU dependency."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import stageb_reference as ref
import derive_fold as fold

HERE = Path(__file__).resolve().parent
WEIGHTS = [.25, .125, .125, .125, .0625, .0625, .0625, .0625, .0625, .0625]


def fixture(passes=2):
    row = ref.reference([[1., -2.]] * 10, WEIGHTS, [1., -1.] * 4, 2)
    records = []
    for p in range(passes):
        for layer in range(48):
            records.append(dict(schema='gufo-prefill-stageb-v1', pass_index=p + 1,
                                start=p, tokens=1, layer=layer, row_start=p,
                                rows_available=1, stride=1, sample_every=64,
                                router_weight_basis='original-top10-normalized',
                                **copy.deepcopy(row['totals'])))
    return records


class ReferenceTests(unittest.TestCase):
    def test_equal_stored_weights_keep_supplied_router_rank(self):
        # Normalization can erase distinct pre-normalization probabilities.
        # IDs deliberately disagree with the router's supplied slot rank.
        weights = [.125] * 8 + [0.] * 2
        experts = [[float(s + 1)] for s in range(10)]
        r = ref.reference(experts, weights, [0.] * 4, 1,
                          ids=list(reversed(range(10))))
        full = math.fsum(w * e[0] for w, e in zip(weights, experts))
        for k in (2, 3, 4, 6):
            with self.subTest(k=k):
                kept = math.fsum(weights[s] * experts[s][0] for s in range(k))
                self.assertEqual(r['totals']['k'][str(k)]['rr'], kept * kept)
                self.assertEqual(r['totals']['k'][str(k)]['rf'], kept * full)

    def test_finite_gate_saturation(self):
        r = ref.reference([[1., -2.]] * 10, WEIGHTS, [4000., -4000., 0., 0.], 1)
        self.assertEqual(r['gates'], [2., 0., 1., 1.])

    def test_known_scalar_and_identity(self):
        r = ref.reference([[1., -2.]] * 10, WEIGHTS, [1., -1.] * 4, 2)
        for k, mass in [(2, .375), (3, .5), (4, .625), (6, .75), (10, 1.)]:
            t = r['totals']['k'][str(k)]
            self.assertAlmostEqual(t['rf'] / t['rr'], 1 / mass)
            self.assertAlmostEqual(t['omitted_mass'], 1 - mass)
        identity = r['totals']['k']['10']
        self.assertEqual(identity['rf'], identity['rr'])
        self.assertEqual(identity['rf'], identity['ff'])
        self.assertEqual(identity['error2'], 0)
        self.assertEqual(r['gates'], [1.] * 4)
        self.assertEqual(r['gate_l2'], 2.)

    def test_unsorted_and_cancellation(self):
        experts = [[float(i - 5), float(10 - i)] for i in range(10)]
        a = ref.reference(experts, WEIGHTS, [0.] * 4, 1)
        order = [4, 0, 1, 2, 3, 5, 6, 7, 8, 9]
        # Move unequal weights, preserving the supplied rank within ties.
        b = ref.reference([experts[i] for i in order], [WEIGHTS[i] for i in order],
                          [0.] * 4, 1, ids=order)
        self.assertEqual(a, b)
        r = ref.reference([[1e12, 1.], [-1e12, 1.]] + [[1., 1.]] * 8,
                          [.125] * 2 + [.09375] * 8, [0.] * 4, 1)
        self.assertTrue(all(math.isfinite(t['rf']) for t in r['totals']['k'].values()))

    def test_bad_shapes_and_nonfinite(self):
        for rows, weights, inject, parts in [([], WEIGHTS, [0.] * 4, 1),
                                             ([[1.]] * 10, WEIGHTS, [0.], 1),
                                             ([[float('nan')]] * 10, WEIGHTS, [0.] * 4, 1),
                                             ([[1.]] * 10, [1.] * 10, [0.] * 4, 1)]:
            with self.assertRaises(ValueError):
                ref.reference(rows, weights, inject, parts)


class DeriveTests(unittest.TestCase):
    def run_tool(self, records, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            totals = root / 'stageb-totals.jsonl'
            totals.write_bytes(b''.join((json.dumps(r) + '\n').encode() for r in records))
            result = subprocess.run([sys.executable, str(HERE / 'derive_fold.py'),
                                     str(totals), '--out', str(root / 'fold'), *extra],
                                    capture_output=True, text=True)
            files = {p.name: p.read_bytes() for p in (root / 'fold').glob('*')}
            return result, files

    def test_known_c_files_lf_hashes_and_warnings(self):
        p, files = self.run_tool(fixture())
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('renorm', p.stdout)
        summary = json.loads(files['summary.json'])
        for k, mass in [(2, .375), (3, .5), (4, .625), (6, .75), (10, 1.)]:
            name = f'fold-L{k}.txt'
            data = files[name]
            self.assertNotIn(b'\r', data)
            self.assertEqual(data.count(b'\n'), 48)
            self.assertAlmostEqual(float(data.splitlines()[0]), 1 / mass, places=7)
            self.assertEqual(summary['files'][str(k)]['sha256'], hashlib.sha256(data).hexdigest())
        self.assertIn('outside [0.5, 2.0]', p.stdout)

    def test_pooled_sums_not_mean_ratios(self):
        rows = fixture()
        for row in rows[48:]:
            for k, t in row['k'].items():
                if k != '10':
                    t['rf'] *= 2
                    t['rr'] *= 4
                    t['error2'] = t['rr'] - 2*t['rf'] + t['ff']
        p, files = self.run_tool(rows)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertAlmostEqual(float(files['fold-L3.txt'].splitlines()[0]), 1.2)

    def test_reject_corrupt_complete_records(self):
        for change in ['identity', 'missing_layer', 'duplicate', 'nan', 'zero',
                       'negative_c', 'missing_k', 'basis', 'count', 'error']:
            with self.subTest(change=change):
                rows = fixture(1)
                t = rows[0]['k']['3']
                if change == 'identity': rows[0]['k']['10']['rr'] *= 1.01
                if change == 'missing_layer': rows.pop()
                if change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
                if change == 'nan': t['rf'] = float('nan')
                if change == 'zero': t['rr'] = 0
                if change == 'negative_c': t['rf'] = -1
                if change == 'missing_k': del rows[0]['k']['6']
                if change == 'basis': rows[0]['router_weight_basis'] = 'survivors'
                if change == 'count': rows[0]['rows'] = 0
                if change == 'error': t['error2'] += 1
                p, files = self.run_tool(rows)
                self.assertNotEqual(p.returncode, 0, change)
                self.assertFalse(any(n.startswith('fold-L') for n in files), change)

    def test_partial_and_empty_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.jsonl'
            for data in [b'', b'{}', b'{}\r', b'\n']:
                path.write_bytes(data)
                with self.assertRaises(ValueError): fold.read_totals([path])

    def test_independent_capture_files_have_local_pass_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'capture-{i}.jsonl' for i in range(2)]
            data = ''.join(json.dumps(r)+'\n' for r in fixture(1))
            for path in paths: path.write_text(data, newline='\n')
            pooled = fold.derive(fold.read_totals(paths))
            self.assertAlmostEqual(pooled[3][3]['c'], 2.)
            self.assertEqual(pooled[3][3]['rows'], 2)
            with self.assertRaises(ValueError): fold.read_totals([paths[0], paths[0]])


if __name__ == '__main__':
    unittest.main()
