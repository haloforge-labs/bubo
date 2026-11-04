"""CPU-only checks: missing evidence must never become a measured zero."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('budget', ROOT / 'tools/bench/decode_tensor_budget.py')
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class BudgetTest(unittest.TestCase):
    def test_roles_and_missing_cells(self):
        self.assertEqual(budget.classify('output.weight', 2), 'target_head')
        self.assertEqual(budget.classify('per_layer_token_embd.weight', 2), 'ple_gather')
        self.assertEqual(budget.classify('blk.2.ffn_down_exps.weight', 3), 'routed_down')
        record = dict(model='native', name='output.weight', tensor_class='target_head',
                      type='Q6_K', shape='2560x248320', layer=-1, bytes=521472000,
                      K=2560, M=248320)
        text, rows = budget.floor_report([record], {}, {}, {},
                                         {('native', 4): (0.24, 54.4)}, 210, 'assumed')
        self.assertIn('whole-pass BW floor = NA', text)
        head = next(r for r in rows if r['tensor_class'] == 'target_head' and r['N'] == 4)
        self.assertEqual(head['pass_bytes'], 521472000)
        self.assertAlmostEqual(head['floor_ms'], 521472000 / 210e6)
        self.assertIsNone(head['measured_ms'])
        self.assertIsNone(head['lost_ms'])
        self.assertIn('| NA | both | No eligible measured cells', text)
        self.assertIn('| 54.400 | 54.400 |', text)

    def test_discard_and_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log.txt'
            meta = ' weights=real evict_bytes=268435456\n'
            path.write_text('BUDGET name=output.weight N=4 us=5 valid=0' + meta +
                            'BUDGET name=output.weight N=4 us=3000 valid=1' + meta +
                            'BUDGET name=output.weight N=4 us=3200 valid=1' + meta +
                            'BUDGET name=output.weight N=5 us=nan valid=1' + meta +
                            'BUDGET name=output.weight N=4 us=1 valid=1 weights=real evict_bytes=0\n')
            result = budget.timing_logs([('prod', path)], True)
            self.assertEqual(result, {('prod', 'output.weight', 4): 3100.0})

    def test_synthetic_routing_cannot_fill_real_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log.txt'
            path.write_text('REAL_CELL layer=0 N=8 kind=chain us=100 valid=1 source=independent_synthetic unique_bytes=20000000\n'
                            'REAL_CELL layer=1 N=8 kind=chain us=100 valid=0 source=real_replay unique_bytes=20000000\n'
                            'REAL_CELL layer=2 N=8 kind=chain us=100 valid=1 source=real_replay unique_bytes=20000000\n'
                            'REAL_CELL layer=3 N=8 kind=chain us=1 valid=1 source=real_replay unique_bytes=20000000\n')
            result = budget.expert_logs([('prod', path)])
            self.assertEqual(result, {('prod', 2, 8): (100.0, 20000000.0)})

    def test_routing_floor_and_width_weighted_loss(self):
        records = [dict(model='prod', name=f'blk.0.ffn_{role}_exps.weight',
                        tensor_class='routed_' + role, type='IQ3_S',
                        shape='2560x640x512', layer=0, bytes=512000000,
                        K=2560, M=640, E=512) for role in ('gate', 'up')]
        records.append(dict(model='prod', name='output.weight', tensor_class='target_head',
                            type='Q6_K', shape='2560x248320', layer=-1,
                            bytes=420000000, K=2560, M=248320, E=1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'routing.txt'
            path.write_text('0 2 ' + ' '.join(map(str, list(range(10)) * 2)) + '\n')
            routing = budget.routing_logs([('prod', path)])
            self.assertEqual(routing['prod', 0, 2], 10)
            text, rows = budget.floor_report(records,
                {('prod', 'output.weight', 2): 4000.0},
                {('prod', 0, 2): (1000.0, 20000000)}, routing,
                {('prod', 2): (0.25, 10.0)}, 210, 'test')
            gate = next(r for r in rows if r['tensor_class'] == 'routed_gate' and r['N'] == 2)
            self.assertEqual(gate['pass_bytes'], 10000000)
            self.assertFalse(gate['additive_time'])
            self.assertIn('| 5.000 | 10.000 | 5.000 |', text)
            self.assertIn('| 0.500 | 1/1 |', text)  # (4 ms - 2 ms) * 0.25
            path.write_text('0 2 ' + ' '.join(['-2'] * 20))
            with self.assertRaises(ValueError):
                budget.routing_logs([('prod', path)])

    def test_timeline_and_control_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.txt'
            path.write_text('verify-body w3 13.22 23.7% 0.35\n'
                            'verify-prefix w3 0.36 0.6% 0.35\n')
            mix = budget.timelines([('prod', path)])
            self.assertAlmostEqual(mix['prod', 3][1], 13.22 / 0.35)
            self.assertEqual(len(mix), 1)
            prefix = 'BUDGET fmt=Q4_K K=2560 M=6144 N=1 valid=1 evict_bytes=268435456 '
            path.write_text(prefix + 'weights=real us=44\n' +
                            prefix + 'weights=real us=40\n' +
                            prefix + 'weights=real us=1\n' +
                            prefix + 'weights=synthetic us=42\n')
            rate, count = budget.control_bandwidth([('prod', path)])
            self.assertEqual(count, 2)
            self.assertAlmostEqual(rate, (8847360/44/1000 + 8847360/40/1000)/2)

    def test_constant_fit_and_chain_interval_count(self):
        records = [dict(model='native', name=f'p{i}', tensor_class='shared_expert',
                        type='Q4_K', shape='2560x640', layer=i, bytes=100000+i*1000,
                        K=2560, M=640, E=1) for i in range(8)]
        times = {('native',r['name'],n): 30+r['bytes']/210/1000
                 for r in records for n in range(1,9)}
        fits = budget.fit_call_constants(records,times,210)
        self.assertAlmostEqual(fits['native',3]['us'],30)
        rows = budget.ledger(records,times,{}, {},210)
        budget.corrected_ledger(rows,fits)
        row = next(r for r in rows if r['tensor_class']=='shared_expert' and r['N']==3)
        self.assertAlmostEqual(row['overhead_ms'],.24)
        self.assertAlmostEqual(row['net_lost_ms'],0)
        self.assertEqual(row['q8_reference_gbps'],229)
        budget.attach_interval_rows(rows,[dict(row,measured_ms=.004)])
        self.assertAlmostEqual(row['interval_ms'],.004)
        self.assertIsNone(next(r for r in rows if r['N']==2 and
                              r['tensor_class']=='shared_expert')['interval_ms'])
        chain = dict(row, tensor_class='routed_chain',count=2,measured_ms=.05)
        budget.corrected_ledger([chain],fits)
        self.assertAlmostEqual(chain['overhead_ms'],.06)
        self.assertAlmostEqual(chain['net_ms'],-.01)  # Never silently clamp.

    def test_single_call_cannot_pool_with_distinct_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mixed.txt'
            prefix='BUDGET name=p N=3 us=50 valid=1 weights=real evict_bytes=268435456 '
            path.write_text(prefix+'sets=1\n'+prefix+'sets=16\n')
            with self.assertRaises(ValueError):
                budget.timing_logs([('native',path)],True)
            path.write_text(prefix+'sets=1\n')
            with self.assertRaises(ValueError):
                budget.timing_logs([('native',path)],True,16)


if __name__ == '__main__':
    unittest.main()
