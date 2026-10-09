"""CPU contracts: known KL, exact union-tail math, fixed ragged schedules/gates."""
import math
import copy
import random
import struct
import unittest

import batch_quality as q


class QualityTest(unittest.TestCase):
    def record(self, probabilities, top):
        logits = [math.log(p) for p in probabilities]
        return {"lse": 0, "vocab": len(logits), "top": [(i, logits[i]) for i in top]}, logits

    def test_known_kl_and_union_not_intersection(self):
        p, pl = self.record([.5, .3, .2], [0])
        r, rl = self.record([.25, .5, .25], [1])
        kl, ta, tb = q.union_kl(p, r, pl, rl)
        expected = .5 * math.log(2) + .3 * math.log(.6) + .2 * math.log(.8)
        self.assertAlmostEqual(kl, expected, places=14)
        self.assertAlmostEqual(ta, .2)
        self.assertAlmostEqual(tb, .25)
        self.assertEqual(q.union_kl(p, p, pl, pl)[0], 0)

    def test_lumped_tail_lower_bound(self):
        p, pl = self.record([.5, .2, .2, .1], [0])
        r, rl = self.record([.6, .2, .1, .1], [0])
        kl, ta, tb = q.union_kl(p, r, pl, rl)
        expected = .5 * math.log(.5 / .6) + .5 * math.log(.5 / .4)
        self.assertAlmostEqual(kl, expected, places=14)
        full = sum(a * math.log(a / b) for a, b in zip([.5, .2, .2, .1], [.6, .2, .1, .1]))
        self.assertLess(kl, full)

    def test_negative_offset_float32_self_kl_exact_zero(self):
        rng = random.Random(101)
        logits = [struct.unpack("<f", struct.pack("<f", rng.gauss(-100, 3)))[0]
                  for _ in range(600)]
        peak = max(logits)
        lse = peak + math.log(math.fsum(math.exp(x - peak) for x in logits))
        ids = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:512]
        row = {"lse": lse, "vocab": len(logits), "top": [(i, logits[i]) for i in ids]}
        self.assertEqual(q.union_kl(row, row, logits, logits)[0], 0)

    def test_invalid_thresholds(self):
        args = [[0.] * 4000, [0.] * 4000, [0.] * 4000, 4000, 4000, 4000, 1, 1, 1, 4000, 8, True]
        self.assertEqual(q.gates(*args), "PASS")
        for index, value in ((9, 3999), (10, 7), (11, False)):
            invalid = args.copy(); invalid[index] = value
            self.assertEqual(q.gates(*invalid), "INVALID")
        fail = args.copy(); fail[0] = [.00011] * 4000
        self.assertEqual(q.gates(*fail), "FAIL")

    def test_session_schedules(self):
        sessions = [{"start": 256, "stop": 768} for _ in range(8)]
        for c in (1, 2, 4):
            ns, widths = set(), set()
            total = 0
            for group in q.schedule(sessions, c):
                positions = [256] * c
                for row in group["passes"]:
                    self.assertEqual(len(row), c)
                    for j, chunk in enumerate(row):
                        self.assertEqual(chunk["position"], positions[j])
                        self.assertTrue(1 <= chunk["width"] <= 8)
                        positions[j] += chunk["width"]
                        widths.add(chunk["width"])
                        total += chunk["width"]
                    ns.add(sum(x["width"] for x in row))
                self.assertEqual(positions, [768] * c)
            self.assertEqual(total, 4096)
            self.assertEqual(widths, set(range(1, 9)))
            self.assertTrue(any(n % 2 for n in ns))
            if c == 4:
                self.assertTrue(any(n % 16 for n in ns))
                self.assertTrue({9, 17, 25, 31, 32}.issubset(ns))

    def test_paired_bootstrap_known_ratio(self):
        self.assertEqual(q.bootstrap_ratio([2., 4.], [1., 2.], 200), [2., 2.])

    def test_run_metadata_rejects_wrong_models_and_dispatch(self):
        common = {"model": "base.gguf", "mtp_model": "mtp.gguf", "probe_sha256": "a" * 64}
        metas = [dict(common, twin="twin.gguf", env={"GUFO_DENSE_TWIN_MODEL": "twin.gguf",
                 "GUFO_DENSE_TWIN_BATCH": "1", "GUFO_IQ4_WIDE_VARIANT": "4", "GUFO_IQ4_WMMA_WAVES": "8"}),
                 dict(common, twin="twin.gguf", env={"GUFO_DENSE_TWIN_MODEL": "twin.gguf"}),
                 dict(common, twin=None, env={}),
                 dict(common, twin="twin.gguf", env={"GUFO_DENSE_TWIN_MODEL": "twin.gguf"})]
        q.validate_arm_metadata(metas)
        for arm, key in ((2, "model"), (1, "mtp_model"), (3, "probe_sha256"), (1, "twin")):
            changed = copy.deepcopy(metas)
            changed[arm][key] = "wrong"
            with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                q.validate_arm_metadata(changed)
        for arm, key, value in ((0, "GUFO_IQ4_WIDE_VARIANT", "0"),
                                (0, "GUFO_IQ4_WMMA_WAVES", ""),
                                (1, "GUFO_DENSE_TWIN_BATCH", "1"),
                                (2, "GUFO_DENSE_TWIN_MODEL", "twin.gguf")):
            changed = copy.deepcopy(metas)
            changed[arm]["env"][key] = value
            with self.subTest(arm=arm, key=key), self.assertRaises(ValueError):
                q.validate_arm_metadata(changed)

    def test_expected_schedule_checks_target_and_width(self):
        manifest = {"sessions": [{"name": "a", "tokens": [10, 11, 12, 13, 14]},
                                  {"name": "b", "tokens": [20, 21, 22, 23, 24]}],
                    "schedules": {"2": [{"sessions": [0, 1], "passes": [
                        [{"position": 1, "width": 2}, {"position": 1, "width": 1}]]}]}}
        expected = q.expected_rows(manifest, 2)
        self.assertEqual(expected[("a", 1)], (12, 2, 3))
        self.assertEqual(expected[("a", 2)], (13, 2, 3))
        self.assertEqual(expected[("b", 1)], (22, 1, 3))
        manifest["schedules"]["2"][0]["passes"][0][0]["width"] = 9
        with self.assertRaises(ValueError):
            q.expected_rows(manifest, 2)


if __name__ == "__main__":
    unittest.main()
