"""Forced public-session manifest, separate engine arms, and exact union KL.

Compact top-512 alone cannot reveal another arm's logits on their union.
FP32 sidecars are required: seek and read one full row at a time, then score
the union with one tail bucket. No missing logit is guessed or imputed.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

PUBLIC = Path(os.environ.get("BUBO_BATCH_QUALITY_SESSIONS", "sessions"))
JOB = Path(os.environ.get("BUBO_BATCH_QUALITY_JOB", "job"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from gufo.control_tokens import kImStart as IM_START, kImEnd as IM_END


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to(JOB.resolve()):
        raise ValueError("all quality outputs must remain in the named job folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def render_public(path):
    if path.parent.resolve() != PUBLIC.resolve() or path.suffix != ".jsonl":
        raise ValueError("quality data must be converted public sessions only")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [r["message"] for r in rows if r.get("type") == "message"]
    if not any("public source:" in json.dumps(m) for m in messages):
        raise ValueError("session lacks converted-public provenance note: " + path.name)
    result = []
    for m in messages:
        role = m.get("role", "user")
        if role == "toolResult":
            role = "tool"
        if role not in ("user", "assistant", "system", "tool"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") if x.get("type") == "text"
                                else json.dumps(x, ensure_ascii=False, sort_keys=True) for x in content)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        result.append(IM_START + role + "\n" + content + IM_END + "\n")
    return "".join(result)


def source_manifest(out, count=8):
    files = sorted(PUBLIC.glob("*.jsonl"))[:count]
    if len(files) < 8:
        raise ValueError("need at least eight converted public sessions")
    write_json(destination(out), {"format": "GF-BATCH-QUALITY-1", "source_root": str(PUBLIC),
                                 "sessions": [{"name": p.stem, "source": str(p),
                                               "source_sha256": sha(p), "text": render_public(p)} for p in files]})


def schedule(sessions, c):
    if len(sessions) % c:
        raise ValueError("session count must divide concurrency")
    groups = []
    for base in range(0, len(sessions), c):
        indices = list(range(base, base + c))
        positions = [sessions[i]["start"] for i in indices]
        ends = [sessions[i]["stop"] for i in indices]
        passes = []
        turn = 0
        rounds = max((e - p + 3) // 4 for p, e in zip(positions, ends))
        while any(p < e for p, e in zip(positions, ends)):
            # Always retain exactly c peers: reserve one row for each remaining
            # pass, and make all chains finish together. No silent c1 tail.
            remain = [e - p for p, e in zip(positions, ends)]
            if min(remain) <= 0:
                raise ValueError("schedule would change concurrency at the tail")
            rounds_left = rounds - turn
            if rounds_left < 1:
                raise ValueError("schedule exhausted rounds")
            probes = {2: [[1, 8], [8, 8], [7, 8], [1, 1]],
                      4: [[1, 2, 3, 3], [4, 4, 4, 5], [8, 8, 8, 1],
                          [7, 8, 8, 8], [8, 8, 8, 8]]}.get(c, [])
            desired = probes[turn] if turn < len(probes) else [1 + (turn + 3 * j) % 8 for j in range(c)]
            widths = [min(r - (rounds_left - 1), w) for r, w in zip(remain, desired)]
            widths = [max(w, max(1, r - 8 * (rounds_left - 1))) for w, r in zip(widths, remain)]
            passes.append([{"position": p, "width": w} for p, w in zip(positions, widths)])
            positions = [p + w for p, w in zip(positions, widths)]
            turn += 1
        groups.append({"sessions": indices, "passes": passes})
    return groups


def freeze(tokenized, out, positions=512, prefix=256):
    value = json.loads(Path(tokenized).read_text(encoding="utf-8"))
    if value.get("source_root") != str(PUBLIC):
        raise ValueError("wrong source root")
    for s in value["sessions"]:
        if sha(s["source"]) != s["source_sha256"] or render_public(Path(s["source"])) == "":
            raise ValueError("source changed or lacks public provenance")
        if len(s["tokens"]) < prefix + positions + 1:
            raise ValueError("session too short for fixed positions: " + s["name"])
        s.update(start=prefix, stop=prefix + positions)
        s.pop("text", None)
    value["context"] = max(prefix + positions + 8, 4096)
    value["schedules"] = {str(c): schedule(value["sessions"], c) for c in (1, 2, 4)}
    value["positions_per_session"] = positions
    write_json(destination(out), value)


def smoke_source(out, text_out):
    path = sorted(PUBLIC.glob("*.jsonl"))[0]
    text = render_public(path)[:2048]
    # newline="": no Windows "\n" -> "\r\n" translation. gufo bench tokenizes this file and the
    # probe tokenizes the JSON text; a translated newline became token 317 instead of 198.
    destination(text_out).write_text(text, encoding="utf-8", newline="")
    write_json(destination(out), {"format": "GF-BATCH-QUALITY-1", "source_root": str(PUBLIC),
                                 "sessions": [{"name": path.stem, "source": str(path),
                                               "source_sha256": sha(path), "text": text}]})


def smoke_freeze(tokenized, out):
    value = json.loads(Path(tokenized).read_text())
    session = value["sessions"][0]
    session.update(start=1, stop=len(session["tokens"]) - 1)
    session.pop("text", None)
    passes, position, turn = [], 1, 0
    while position < session["stop"]:
        width = min(1 + turn % 8, session["stop"] - position)
        passes.append([{"position": position, "width": width}])
        position += width
        turn += 1
    value.update(context=max(32768, len(session["tokens"]) + 8),
                 schedules={"1": [{"sessions": [0], "passes": passes}]})
    write_json(destination(out), value)


def run_arm(args):
    out = destination(args.out)
    if out.with_suffix(".jsonl").exists() or out.with_suffix(".f32").exists():
        raise ValueError("refusing to overwrite an existing arm")
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("GUFO_DENSE_TWIN") or k.startswith("GUFO_IQ4_"):
            env.pop(k)
    if args.arm != "q8":
        if not args.twin:
            raise ValueError("twin arm requires --twin")
        env["GUFO_DENSE_TWIN_MODEL"] = str(args.twin)
    if args.arm == "wmma" and args.c > 1:
        env.update(GUFO_DENSE_TWIN_BATCH="1", GUFO_IQ4_WIDE_VARIANT="4")
        env["GUFO_IQ4_WMMA_WAVES"] = str(args.waves or 8)
    cmd = [str(args.probe), "--model", str(args.model), "--mtp-model", str(args.mtp_model),
           "--manifest", str(args.manifest), "--manifest-sha256", sha(args.manifest),
           "--c", str(args.c), "--arm", args.arm, "--out", str(out)]
    meta = {"command": cmd, "manifest_sha256": sha(args.manifest),
            "probe_sha256": sha(args.probe), "model": str(args.model), "mtp_model": str(args.mtp_model),
            "twin": str(args.twin) if args.arm != "q8" else None,
            "env": {k: v for k, v in env.items() if k.startswith("GUFO_")}, "complete": False}
    write_json(str(out) + ".run.json", meta)
    with open(str(out) + ".log", "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    meta["exit_code"] = result.returncode
    if result.returncode == 0:
        meta.update(complete=True, compact_sha256=sha(str(out) + ".jsonl"), full_sha256=sha(str(out) + ".f32"))
    write_json(str(out) + ".run.json", meta)
    if result.returncode:
        raise ValueError("quality arm failed; inspect " + str(out) + ".log")


INT_FIELDS = ("c", "K", "N", "position", "target", "vocab", "offset")


def as_int(value):
    # The probe's JSON writer emits the shortest number form, so an integral
    # double such as token 100000 arrives as 1e+05 (float). Accept only exact
    # integral values.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric integer field")
    if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
        raise ValueError("non-integral integer field")
    return int(value)


def normalize_row(r):
    for k in INT_FIELDS:
        if k in r:
            r[k] = as_int(r[k])
    r["top"] = [[as_int(i), v] for i, v in r["top"]]
    return r


class Dump:
    def __init__(self, prefix):
        self.prefix = str(prefix)
        meta = json.loads(Path(self.prefix + ".run.json").read_text())
        if not meta.get("complete"):
            raise ValueError("incomplete engine arm")
        if sha(self.prefix + ".jsonl") != meta["compact_sha256"] or sha(self.prefix + ".f32") != meta["full_sha256"]:
            raise ValueError("dump checksum mismatch")
        self.meta = meta
        self.rows = {}
        for line in Path(self.prefix + ".jsonl").read_text().splitlines():
            r = normalize_row(json.loads(line))
            key = (r["session"], r["position"])
            if key in self.rows or r["manifest_sha256"] != meta["manifest_sha256"]:
                raise ValueError("duplicate position or manifest mismatch")
            if not all(math.isfinite(r[k]) for k in ("lse", "target_logit")):
                raise ValueError("nonfinite compact logit")
            if len(r["top"]) != min(512, r["vocab"]) or len({i for i, v in r["top"]}) != len(r["top"]):
                raise ValueError("bad top512")
            self.rows[key] = r
        self.full = open(self.prefix + ".f32", "rb")

    def read(self, key):
        r = self.rows[key]
        self.full.seek(r["offset"])
        raw = self.full.read(4 * r["vocab"])
        if len(raw) != 4 * r["vocab"]:
            raise ValueError("truncated full logit sidecar")
        values = array.array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        for i, v in r["top"]:
            if i < 0 or i >= len(values) or values[i] != v:
                raise ValueError("compact/full top mismatch")
        if values[r["target"]] != r["target_logit"]:
            raise ValueError("compact/full target mismatch")
        return r, values, raw

    def close(self):
        self.full.close()


def union_kl(p, q, plogits, qlogits):
    if p["vocab"] != q["vocab"]:
        raise ValueError("vocabulary mismatch")
    ids = sorted({i for i, _ in p["top"]} | {i for i, _ in q["top"]})
    a = [math.exp(plogits[i] - p["lse"]) for i in ids]
    b = [math.exp(qlogits[i] - q["lse"]) for i in ids]
    ta, tb = 1 - math.fsum(a), 1 - math.fsum(b)
    if min(ta, tb) < -1e-12:
        raise ValueError("negative tail mass")
    ta, tb = max(ta, 0), max(tb, 0)
    terms = [x * ((plogits[i] - p["lse"]) - (qlogits[i] - q["lse"]))
             for i, x in zip(ids, a)]
    if ta:
        if not tb:
            return math.inf, ta, tb
        terms.append(ta * (math.log(ta) - math.log(tb)))
    kl = math.fsum(terms)
    if kl < -1e-12:
        raise ValueError("negative KL beyond numerical roundoff")
    return max(kl, 0), ta, tb


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def bootstrap_ratio(a, b, rounds=2000, seed=9409):
    rng = random.Random(seed)
    ratios = []
    for _ in range(rounds):
        indices = [rng.randrange(len(a)) for _ in a]
        den = math.fsum(b[i] for i in indices)
        num = math.fsum(a[i] for i in indices)
        ratios.append(num / den if den else (0 if num == 0 else math.inf))
    return [percentile(ratios, .025), percentile(ratios, .975)]


def gates(primary, secondary_w, secondary_n, top_primary, top_w, top_n,
          nll_w, nll_n, nll_q, positions, sessions, self_zero):
    if positions < 4000 or sessions < 8 or not self_zero:
        return "INVALID"
    ok = (math.fsum(primary) / positions <= 1e-4 and percentile(primary, .99) <= 1e-3 and
          top_primary / positions >= .999 and
          math.fsum(secondary_w) <= 1.10 * math.fsum(secondary_n) and
          top_w / positions >= top_n / positions - .002 and
          nll_w / nll_q <= nll_n / nll_q + .002)
    return "PASS" if ok else "FAIL"


def validate_arm_metadata(metas):
    w, n, q, s = metas
    for key in ("model", "mtp_model", "probe_sha256"):
        if any(m.get(key) != w.get(key) for m in metas) or not w.get(key):
            raise ValueError("arm metadata mismatch: " + key)
    if not w.get("twin") or n.get("twin") != w["twin"] or s.get("twin") != w["twin"]:
        raise ValueError("twin arms did not load the same twin")
    if q.get("twin"):
        raise ValueError("Q8 metadata names a twin")
    env = w.get("env", {})
    if env.get("GUFO_DENSE_TWIN_BATCH") != "1" or env.get("GUFO_IQ4_WIDE_VARIANT") != "4" or env.get("GUFO_IQ4_WMMA_WAVES") not in ("4", "8", "16"):
        raise ValueError("WMMA arm lacks pinned batch/variant/waves environment")
    for meta in (w, n, s):
        if meta.get("env", {}).get("GUFO_DENSE_TWIN_MODEL") != meta["twin"]:
            raise ValueError("loaded twin environment disagrees with run metadata")
    for meta in (n, s):
        if meta.get("env", {}).get("GUFO_DENSE_TWIN_BATCH"):
            raise ValueError("narrow/self arm enabled twin batch")
    if q.get("env", {}).get("GUFO_DENSE_TWIN_MODEL"):
        raise ValueError("Q8 arm loaded a twin")


def expected_rows(manifest, c):
    expected = {}
    sessions = manifest["sessions"]
    for group in manifest["schedules"][str(c)]:
        indices = group["sessions"]
        if len(indices) != c or len(set(indices)) != c:
            raise ValueError("frozen group has wrong concurrency")
        for row in group["passes"]:
            if len(row) != c:
                raise ValueError("frozen pass has wrong concurrency")
            total = sum(chunk["width"] for chunk in row)
            for i, chunk in zip(indices, row):
                session = sessions[i]
                p, k = chunk["position"], chunk["width"]
                if not 1 <= k <= 8:
                    raise ValueError("frozen width out of range")
                for t in range(k):
                    key = (session["name"], p + t)
                    if key in expected:
                        raise ValueError("duplicate frozen row")
                    expected[key] = (session["tokens"][p + t + 1], k, total)
    return expected


def score(args):
    dumps = [Dump(getattr(args, arm)) for arm in ("wmma", "narrow", "q8", "self")]
    try:
        w, n, q, s = dumps
        validate_arm_metadata([d.meta for d in dumps])
        keys = set(w.rows)
        if any(set(d.rows) != keys for d in dumps):
            raise ValueError("arm position sets differ")
        manifest = json.loads(Path(args.manifest).read_text())
        if any(d.meta["manifest_sha256"] != sha(args.manifest) for d in dumps):
            raise ValueError("arms did not share frozen manifest")
        for session in manifest["sessions"]:
            if sha(session["source"]) != session["source_sha256"]:
                raise ValueError("public source checksum changed")
            render_public(Path(session["source"]))
        expected = {(r["name"], p) for r in manifest["sessions"] for p in range(r["start"], r["stop"])}
        if keys != expected:
            raise ValueError("position set does not match manifest")
        expected_schedule = expected_rows(manifest, args.c)
        if set(expected_schedule) != expected:
            raise ValueError("frozen schedule does not cover its declared positions")
        primary, sw, sn, tails, self_kls = [], [], [], [], []
        tp = tw = tn = 0
        nw = nn = nq = 0.0
        self_zero = True
        worst = []
        for key in sorted(keys):
            wr, wl, wb = w.read(key)
            nr, nl, nb = n.read(key)
            qr, ql, qb = q.read(key)
            sr, sl, sb = s.read(key)
            for r, expected_arm in zip((wr, nr, qr, sr), ("wmma", "narrow", "q8", "self")):
                if r["arm"] != expected_arm or r["c"] != args.c:
                    raise ValueError("arm label or concurrency mismatch")
                if (r["target"], r["K"], r["N"]) != expected_schedule[key]:
                    raise ValueError("forced target or pass schedule disagrees with frozen manifest")
            self_kl, _, _ = union_kl(nr, sr, nl, sl)
            self_kls.append(self_kl)
            self_zero &= nb == sb and nr["lse"] == sr["lse"] and self_kl == 0
            a, ta, tb = union_kl(nr, wr, nl, wl)
            b, tq, tww = union_kl(qr, wr, ql, wl)
            d, tq2, tnn = union_kl(qr, nr, ql, nl)
            primary.append(a); sw.append(b); sn.append(d)
            tails.append((ta, tb, tq, tww, tq2, tnn))
            tp += nr["top"][0][0] == wr["top"][0][0]
            tw += qr["top"][0][0] == wr["top"][0][0]
            tn += qr["top"][0][0] == nr["top"][0][0]
            nw += wr["lse"] - wr["target_logit"]
            nn += nr["lse"] - nr["target_logit"]
            nq += qr["lse"] - qr["target_logit"]
            worst.append({"session": key[0], "position": key[1], "KL": a, "K": wr["K"], "N": wr["N"]})
        count = len(keys)
        if not count or nq <= 0:
            raise ValueError("empty positions or invalid Q8 NLL")
        report = {"status": gates(primary, sw, sn, tp, tw, tn, nw, nn, nq, count,
                                    len({k[0] for k in keys}), self_zero),
                  "c": args.c, "positions": count, "sessions": len({k[0] for k in keys}),
                  "method": "exact logits on pairwise top512 union + one tail bucket; KL lower bound",
                  "self_KL": math.fsum(self_kls) / count, "self_full_rows_identical": self_zero,
                  "primary_mean_KL": math.fsum(primary) / count, "primary_p99_KL": percentile(primary, .99),
                  "primary_top1": tp / count, "secondary_WMMA_mean_KL": math.fsum(sw) / count,
                  "secondary_narrow_mean_KL": math.fsum(sn) / count,
                  "secondary_KL_ratio_CI95": bootstrap_ratio(sw, sn, args.bootstrap),
                  "Q8_top1_WMMA": tw / count, "Q8_top1_narrow": tn / count,
                  "NLL_WMMA_over_Q8": nw / nq, "NLL_narrow_over_Q8": nn / nq,
                  "tail_mass_max": max(max(t) for t in tails),
                  "tail_mass_mean": math.fsum(v for t in tails for v in t) / (6 * count),
                  "worst_primary": sorted(worst, key=lambda r: r["KL"], reverse=True)[:20]}
        write_json(destination(args.out), report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    finally:
        for dump in dumps:
            dump.close()


def smoke(gfle, compact):
    """Compare existing gufo bench --logit-eval GFLE rows to c1 probe rows."""
    rows = [normalize_row(json.loads(x)) for x in Path(compact).read_text().splitlines()]
    by_position = {r["position"]: r for r in rows}
    count = 0
    with open(gfle, "rb") as f:
        magic, version, vocab, top = struct.unpack("<4sIII", f.read(16))
        if (magic, version, top) != (b"GFLE", 1, 64):
            raise ValueError("unknown existing single-session harness dump")
        while head := f.read(24):
            if len(head) != 24:
                raise ValueError("truncated GFLE row")
            p, width, target, top1, target_lp, lse = struct.unpack("<iiiiff", head)
            ids = struct.unpack("<64i", f.read(256))
            lp = struct.unpack("<64f", f.read(256))
            r = by_position.get(p)
            f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
            if not r or r["target"] != target or r["K"] != width or r["vocab"] != vocab:
                raise ValueError("smoke positions, width or target mismatch")
            if top1 != r["top"][0][0]:
                raise ValueError("smoke top1 mismatch")
            if lse != f32(r["lse"]) or target_lp != f32(r["target_logit"] - r["lse"]):
                raise ValueError("smoke LSE/target mismatch")
            if list(ids) != [i for i, v in r["top"][:64]] or list(lp) != [f32(v - r["lse"]) for i, v in r["top"][:64]]:
                raise ValueError("smoke top64 mismatch")
            count += 1
    if count != len(rows) or not count:
        raise ValueError("smoke row count mismatch")
    print(f"SMOKE PASS rows={count} existing_GFLE_different=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("source"); p.add_argument("--out", required=True); p.add_argument("--sessions", type=int, default=8)
    p = sub.add_parser("freeze"); p.add_argument("--tokens", required=True); p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=512); p.add_argument("--prefix", type=int, default=256)
    p = sub.add_parser("smoke-source"); p.add_argument("--out", required=True); p.add_argument("--text", required=True)
    p = sub.add_parser("smoke-freeze"); p.add_argument("--tokens", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    for flag in ("probe", "model", "mtp-model", "manifest", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--twin", type=Path); p.add_argument("--c", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--arm", choices=("wmma", "narrow", "q8", "self"), required=True)
    p.add_argument("--waves", type=int, choices=(4, 8, 16), help="pin selected WMMA A/B launch")
    p = sub.add_parser("score")
    for flag in ("wmma", "narrow", "q8", "self", "manifest", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--c", type=int, choices=(2, 4), required=True); p.add_argument("--bootstrap", type=int, default=2000)
    p = sub.add_parser("smoke"); p.add_argument("--gfle", type=Path, required=True); p.add_argument("--compact", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "source": source_manifest(args.out, args.sessions)
        elif args.cmd == "freeze": freeze(args.tokens, args.out, args.positions, args.prefix)
        elif args.cmd == "smoke-source": smoke_source(args.out, args.text)
        elif args.cmd == "smoke-freeze": smoke_freeze(args.tokens, args.out)
        elif args.cmd == "run": run_arm(args)
        elif args.cmd == "score": return score(args)
        else: smoke(args.gfle, args.compact)
        return 0
    except (ValueError, KeyError, OSError, OverflowError) as e:
        print("INVALID: " + str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
