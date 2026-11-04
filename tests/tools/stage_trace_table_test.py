"""CPU-only aggregation contract. Fixture stage values are synthetic.

The old phase-line format is excerpted from hcinject-operator/iq3-trace/
serve-20261031-032730-default-r1.err; it must not be parsed as stage data.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage_trace_table", ROOT / "tools/bench/stage_trace_table.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)
FIXTURE = Path(__file__).parent / "fixtures/stage_trace.err"


class StageTraceTest(unittest.TestCase):
    def test_weighted_windows_and_scope(self):
        body = TRACE.read_windows(FIXTURE)
        self.assertEqual(body[3][0], 8)
        self.assertEqual(body[3][1]["hc_attn"], 2.5)
        self.assertEqual(body[3][1]["body"], 41.5)
        self.assertEqual(body[4][0], 1)
        prefix = TRACE.read_windows(FIXTURE, "verify-prefix")
        self.assertEqual(prefix[3][1]["embedding"], 0.5)
        self.assertEqual(prefix[3][1]["head"], 0)

    def test_difference_and_missing_width(self):
        body = TRACE.read_windows(FIXTURE)
        native = {3: (2, {key: value + 1 for key, value in body[3][1].items()})}
        output = TRACE.table(body, native)
        self.assertIn("w3 production\tw3 native\tw3 difference", output)
        self.assertIn("hc_attn\t2.500\t3.500\t+1.000\t3.000\t-\t-", output)
        self.assertIn("passes\t8\t2\t-\t1\t-\t-", output)
        self.assertNotIn("native", TRACE.table(body))

    def test_invalid_window_is_not_silently_dropped(self):
        for change in ("0x:", "2x: hc_attn nan", "2x: bogus 1"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.err"
                path.write_text("stages verify-body w3 " + change, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bad.err:1:"):
                    TRACE.read_windows(path)


if __name__ == "__main__":
    unittest.main()
