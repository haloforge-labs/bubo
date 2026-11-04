#!/usr/bin/env python3
"""Compare in-graph stage windows, weighted by passes (all values ms/pass).

Use production.err [native.err]. Default scope is verify-body; use
--phase verify-prefix for the separate prefix. Stamp overhead is included;
compare differences between equally instrumented runs, not absolute costs.

Widths with fewer than --min-passes passes in an input are dropped from that
input: the untimed warm-up request and the first pass of each width are eager
and would otherwise read as kernel cost. The "all" columns weight each input by
its own pass counts; the "mix" columns weight both inputs by the production
pass counts over the widths both have, which separates per-stage cost from the
width mix. launch+tail is body minus the stage sum, so a negative value is an
error (exit status 1); sum/body is the real coverage check.
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import re
import sys


STAGES = ("hc_attn", "mixer", "combine_hc_ffn", "router", "shared", "routed",
          "combine_epilogue", "embedding", "head", "sum", "body", "launch+tail")
LINE = re.compile(r"\bstages (verify-body|verify-prefix) w(\d+) (\d+)x:(.*)$")


def read_windows(path, phase="verify-body"):
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            match = LINE.search(line)
            if not match or match[1] != phase:
                continue
            width, count = int(match[2]), int(match[3])
            fields = match[4].split()
            try:
                if count <= 0 or width <= 0 or len(fields) != 2 * len(STAGES):
                    raise ValueError("invalid width, count or field count")
                values = dict(zip(fields[::2], map(float, fields[1::2])))
                if set(values) != set(STAGES) or not all(
                        math.isfinite(value) for value in values.values()):
                    raise ValueError("invalid stage names or values")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            counts[width] += count
            for stage, value in values.items():
                totals[width][stage] += value * count
    return {width: (counts[width], {stage: total / counts[width]
                                  for stage, total in values.items()})
            for width, values in totals.items()}


def keep(data, min_passes):
    return {width: entry for width, entry in data.items() if entry[0] >= min_passes}


def weighted(data, weights):
    """Mean of each stage over the widths in weights, weighted by weights."""
    total = sum(weights.values())
    return (total, {stage: sum(weights[width] * data[width][1][stage] for width in weights) / total
                    for stage in STAGES})


def cell(values, stage):
    if stage == "sum/body":
        return f"{values['sum'] / values['body']:.3f}" if values["body"] > 0 else "-"
    return f"{values[stage]:.3f}"


def table(production, native=None):
    widths = sorted(set(production) | set(native or {}))
    columns = []  # (header, production values or None, native values or None)
    for width in widths:
        p = production[width][1] if width in production else None
        n = native[width][1] if native and width in native else None
        columns.append((f"w{width}", p, n, production.get(width, (0,))[0], (native or {}).get(width, (0,))[0]))
    groups = [("all", weighted(production, {w: production[w][0] for w in production})[1] if production else None,
               weighted(native, {w: native[w][0] for w in native})[1] if native else None,
               sum(v[0] for v in production.values()), sum(v[0] for v in (native or {}).values()))]
    shared = [w for w in widths if w in production and native and w in native]
    if native is not None and shared:
        mix = {w: production[w][0] for w in shared}
        groups.append(("mix", weighted(production, mix)[1], weighted(native, mix)[1], sum(mix.values()), sum(mix.values())))
    lines = []
    header = ["stage"]
    for name, _, _, _, _ in columns + groups:
        header += ([f"{name} production", f"{name} native", f"{name} difference"]
                   if native is not None else [f"{name} production"])
    lines.append("\t".join(header))
    for stage in ("passes", *STAGES, "sum/body"):
        row = [stage]
        for name, p, n, pc, nc in columns + groups:
            if stage == "passes":
                row.append(str(pc) if p else "-")
                if native is not None:
                    row += [str(nc) if n else "-", "-"]
                continue
            row.append(cell(p, stage) if p else "-")
            if native is not None:
                row.append(cell(n, stage) if n else "-")
                if not p or not n or stage == "sum/body":
                    row.append("-")
                else:
                    row.append(f"{n[stage] - p[stage]:+.3f}")
        lines.append("\t".join(row))
    return "\n".join(lines)


def negative_remainders(data, label):
    return [f"{label} w{width}: launch+tail {values['launch+tail']:.3f} ms < 0"
            for width, (_, values) in sorted(data.items()) if values["launch+tail"] < -0.001]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("native", type=Path, nargs="?")
    parser.add_argument("--phase", choices=("verify-body", "verify-prefix"),
                        default="verify-body")
    parser.add_argument("--min-passes", type=int, default=20,
                        help="drop widths with fewer passes in an input (default 20)")
    args = parser.parse_args()
    try:
        production = keep(read_windows(args.production, args.phase), args.min_passes)
        native = keep(read_windows(args.native, args.phase), args.min_passes) if args.native else None
        if not production or (native is not None and not native):
            parser.error(f"no {args.phase} stage windows with {args.min_passes}+ passes in an input")
        names = f"{args.production}" + (f" {args.native}" if args.native else "")
        print(f"# {args.phase}: ms/pass; difference = native - production; stamp overhead included; "
              f"widths under {args.min_passes} passes dropped; inputs: {names}")
        print(table(production, native))
        bad = negative_remainders(production, "production") + (negative_remainders(native, "native") if native else [])
        for message in bad:
            print(f"ERROR: {message}", file=sys.stderr)
        if bad:
            sys.exit(1)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
