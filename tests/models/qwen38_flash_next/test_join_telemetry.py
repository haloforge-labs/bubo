"""CPU contract tests for the prefill telemetry / continuation-loss join."""

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("join_telemetry.py")
HEADER = struct.Struct("<8sIIIIQ")
RECORD = struct.Struct("<II7f")
HASH = "a" * 64


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def population_rows(layer, row_start, length, chunk_start=0, chunk_tokens=2):
    return [{"layer": value,
             "row_start": row_start if value == layer else chunk_start,
             "rows": length if value == layer else chunk_tokens}
            for value in range(48)]


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.maps = []
        self.ref = []
        self.cand = []

    def add_session(self, name, signal, delta, *, start=2, compact=False,
                    layer=3, rows=None):
        binary = self.root / f"{name}.bin"
        index = self.root / f"{name}.index.json"
        layers = self.root / f"{name}.layers.jsonl"
        records = rows if rows is not None else [(0, layer, signal), (1, layer, signal)]
        binary.write_bytes(
            HEADER.pack(b"GUFOPFT1", 1, 32, 36, 0, len(self.maps) + 1)
            + b"".join(RECORD.pack(token, lyr, value, value / 2, value + 1,
                                   0.1, 0.2, 0.3, 0.4)
                       for token, lyr, value in records)
        )
        jsonl(layers, population_rows(layer, 0, len(records), chunk_tokens=start))
        index.write_text(json.dumps({
            "schema": "gufo-prefill-telemetry-v1", "binary": binary.name,
            "pass": len(self.maps) + 1, "start": 0, "tokens": start,
            "layers": [layer], "records": len(records),
            "policy_identity": "reference:10", "record_bytes": 36,
            "header_bytes": 32, "layer_records": layers.name,
            "population_layers": 48,
        }), encoding="utf-8")
        self.maps.append({"session": name, "manifest_sha256": HASH,
                          "prompt_tokens": start, "start": 0,
                          "indexes": [str(index.resolve())]})
        # First continuation target is deliberately excluded from the score.
        for position, loss in [(start - 1, 99), (start, 1)]:
            base = {"session": name, "manifest_sha256": HASH,
                    "position": position, "target_position": position + 1,
                    "target": 42 + position, "prompt_tokens": start,
                    "start": start}
            if compact:
                self.ref.append({**base, "lse": loss + 2, "target_logit": 2})
                self.cand.append({**base, "nll": loss + (delta if position == start else 200)})
            else:
                self.ref.append({**base, "nll": loss})
                self.cand.append({**base, "nll": loss + (delta if position == start else 200)})

    def run_join(self, extra=()):
        mp, rp, cp = (self.root / x for x in ("map.jsonl", "ref.jsonl", "cand.jsonl"))
        jsonl(mp, self.maps)
        jsonl(rp, self.ref)
        jsonl(cp, self.cand)
        return subprocess.run([sys.executable, str(SCRIPT), "--telemetry-map", str(mp),
                               "--reference-nll", str(rp), "--candidate-nll", str(cp),
                               *extra], capture_output=True, text=True)

    def test_roundtrip_scores_and_broadcast_view(self):
        for name, signal, delta in [("low", 1, -2), ("mid", 2, 1), ("high", 3, 3)]:
            self.add_session(name, signal, delta, compact=True)
        view = self.root / "view.jsonl"
        result = self.run_join(("--prompt-token-out", str(view)))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        layer = report["layers"]["3"]
        self.assertEqual(layer["rows"], 3)
        self.assertAlmostEqual(layer["top1_mass"]["spearman"], 1)
        self.assertAlmostEqual(layer["top1_mass"]["auc"], 1)
        self.assertAlmostEqual(layer["top1_mass"]["auc_flipped"], 0)
        self.assertEqual(sorted(r["delta_nll"] for r in report["sessions"]), [-2, 1, 3])
        rows = [json.loads(line) for line in view.read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["target_position"], 1)
        self.assertEqual(rows[0]["label_scope"], "session_broadcast_not_token_causal")
        self.assertIn("gate_l2", rows[0])

    def test_inverse_signal_and_tied_ranks(self):
        for name, signal, delta in [("a", 2, -1), ("b", 2, -1), ("c", 1, 3)]:
            self.add_session(name, signal, delta)
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)
        metric = json.loads(result.stdout)["layers"]["3"]["top1_mass"]
        self.assertAlmostEqual(metric["spearman"], -1)
        self.assertAlmostEqual(metric["auc"], 0)
        self.assertAlmostEqual(metric["auc_flipped"], 1)

    def test_tied_auc_gets_half_credit(self):
        self.add_session("negative", 2, -1)
        self.add_session("positive", 2, 1)
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)
        metric = json.loads(result.stdout)["layers"]["3"]["top1_mass"]
        self.assertEqual(metric["auc"], 0.5)
        self.assertEqual(metric["auc_flipped"], 0.5)
        self.assertIsNone(metric["spearman"])

    def test_corrupt_and_missing_telemetry_fail(self):
        self.add_session("a", 1, 1)
        index = Path(self.maps[0]["indexes"][0])
        binary = index.with_suffix("").with_suffix(".bin")
        binary.write_bytes(binary.read_bytes()[:-1])
        self.assertIn("truncated or oversized binary", self.run_join().stderr)
        binary.unlink()
        self.assertIn("No such file", self.run_join().stderr)
        self.maps[0]["indexes"] = []
        self.assertIn("missing telemetry indexes", self.run_join().stderr)

    def test_mismatched_targets_hash_and_keys_fail(self):
        self.add_session("a", 1, 1)
        self.cand[1]["target"] += 1
        self.assertIn("target mismatch", self.run_join().stderr)
        self.cand[1]["target"] -= 1
        self.cand[1]["manifest_sha256"] = "b" * 64
        self.assertIn("NLL key mismatch", self.run_join().stderr)
        self.cand[1]["manifest_sha256"] = HASH
        self.cand.pop()
        self.assertIn("NLL key mismatch", self.run_join().stderr)

    def test_trimmed_layer_coverage_and_invalid_position(self):
        self.add_session("trim", 1, 1, layer=47, rows=[(1, 47, 1)])
        layers = self.root / "trim.layers.jsonl"
        jsonl(layers, population_rows(47, 1, 1))
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)
        coverage = json.loads(result.stdout)["layers"]["47"]["coverage"]
        self.assertEqual(coverage["observed_tokens"], 1)
        self.assertEqual(coverage["expected_tokens"], 2)
        row = json.loads(result.stdout)["layers"]["47"]["session_rows"][0]
        self.assertEqual((row["observed_tokens"], row["expected_tokens"]), (1, 2))
        blob = self.root / "trim.bin"
        data = bytearray(blob.read_bytes())
        struct.pack_into("<I", data, 32, 2)
        blob.write_bytes(data)
        self.assertIn("record outside layer coverage", self.run_join().stderr)

    def test_nonfinal_layer_cannot_hide_missing_prefix(self):
        self.add_session("missing", 1, 1, layer=3, rows=[(1, 3, 1)])
        jsonl(self.root / "missing.layers.jsonl", population_rows(3, 1, 1))
        self.assertIn("invalid trimmed layer coverage", self.run_join().stderr)

    def test_nonfinite_duplicate_and_unknown_cli_fail(self):
        self.add_session("a", 1, 1)
        self.cand[1]["nll"] = math.nan
        self.assertIn("nonfinite JSON constant", self.run_join().stderr)
        self.cand[1]["nll"] = 2
        self.ref.append(dict(self.ref[1]))
        self.assertIn("duplicate NLL key", self.run_join().stderr)
        self.ref.pop()
        self.assertIn("unrecognized arguments", self.run_join(("--unknown", "1")).stderr)

    def test_population_rows_include_unselected_layers(self):
        self.add_session("all_layers", 1, 1)
        jsonl(self.root / "all_layers.layers.jsonl", [
            {"layer": layer, "row_start": 0, "rows": 2}
            for layer in range(48)])
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)["layers"]), {"3"})

    def test_pruned_reference_is_rejected(self):
        self.add_session("wrong_policy", 1, 1)
        path = Path(self.maps[0]["indexes"][0])
        index = json.loads(path.read_text())
        index["policy_identity"] = "prefill_experts=renorm:4"
        path.write_text(json.dumps(index))
        self.assertIn("reference top-10", self.run_join().stderr)

    def test_each_chunk_may_trim_its_final_layer_prefix(self):
        self.add_session("chunks", 1, 1, start=4, layer=47,
                         rows=[(1, 47, 1)])
        first = Path(self.maps[0]["indexes"][0])
        index = json.loads(first.read_text())
        index["tokens"] = 2
        first.write_text(json.dumps(index))
        jsonl(self.root / "chunks.layers.jsonl", population_rows(47, 1, 1))
        second = self.root / "second.index.json"
        index.update(binary="second.bin", start=2)
        index["pass"] = 2
        index["layer_records"] = "second.layers.jsonl"
        second.write_text(json.dumps(index))
        (self.root / "second.bin").write_bytes(
            HEADER.pack(b"GUFOPFT1", 1, 32, 36, 0, 2) +
            RECORD.pack(3, 47, 1, .5, 2, 1, 1, 1, 1))
        jsonl(self.root / "second.layers.jsonl", population_rows(47, 3, 1, 2))
        self.maps[0]["indexes"].append(str(second.resolve()))
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["layers"]["47"]["coverage"]
                         ["observed_tokens"], 2)

    def test_unselected_population_truncation_is_rejected(self):
        self.add_session("truncated_population", 1, 1)
        jsonl(self.root / "truncated_population.layers.jsonl",
              population_rows(3, 0, 2)[:-1])
        self.assertIn("missing population layer metadata", self.run_join().stderr)

    def test_separate_processes_can_reuse_numeric_pass(self):
        self.add_session("first_process", 1, -1)
        self.add_session("second_process", 2, 1)
        path = Path(self.maps[1]["indexes"][0])
        index = json.loads(path.read_text())
        index["pass"] = 1
        path.write_text(json.dumps(index))
        binary = path.parent / index["binary"]
        data = bytearray(binary.read_bytes())
        struct.pack_into("<Q", data, 24, 1)
        binary.write_bytes(data)
        result = self.run_join()
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
