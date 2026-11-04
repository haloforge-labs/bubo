#!/usr/bin/env python3
"""Join prefill router telemetry to paired continuation NLL, using only the stdlib.

The statistical unit is a manifest/session. Prompt-token JSONL repeats that
unit's label for inspection and must not be treated as token-level evidence.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct
import sys


HEADER = struct.Struct("<8sIIIIQ")
RECORD = struct.Struct("<II7f")
SIGNALS = ("top1_mass", "top4_mass", "gate_l2")
FIELDS = ("top1_mass", "top4_mass", "gate_l2", "gate0", "gate1", "gate2", "gate3")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}: {value!r}")
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"nonfinite or invalid {label}: {value!r}")
    return float(value)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON member: {key}")
        obj[key] = value
    return obj


def no_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def rows(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(line.strip(), f"blank JSONL row: {path}:{line_number}")
            try:
                obj = json.loads(line, parse_constant=no_constant,
                                 object_pairs_hook=unique_object)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"bad JSONL {path}:{line_number}: {exc}") from exc
            require(type(obj) is dict, f"non-object JSONL row: {path}:{line_number}")
            yield obj


def read_object(path):
    with open(path, encoding="utf-8") as stream:
        obj = json.load(stream, parse_constant=no_constant,
                        object_pairs_hook=unique_object)
    require(type(obj) is dict, f"non-object index: {path}")
    return obj


def identity(obj):
    key = (obj["manifest_sha256"], obj["session"])
    require(all(type(x) is str and x for x in key), f"invalid session identity: {key}")
    return key


def load_nll(path):
    result = {}
    for row in rows(path):
        key = (*identity(row), integer(row["position"], "position"))
        require(key not in result, f"duplicate NLL key {key} in {path}")
        target = integer(row["target"], "target")
        target_position = integer(row.get("target_position", key[2] + 1),
                                  "target_position")
        require(target_position == key[2] + 1, f"bad target position {key}")
        if "nll" in row:
            nll = finite(row["nll"], "nll")
            if "lse" in row or "target_logit" in row:
                require("lse" in row and "target_logit" in row, f"partial logits {key}")
                expected = finite(row["lse"], "lse") - finite(row["target_logit"], "target_logit")
                require(math.isclose(nll, expected, rel_tol=1e-6, abs_tol=1e-6),
                        f"inconsistent NLL {key}")
        else:
            nll = finite(row["lse"], "lse") - finite(row["target_logit"], "target_logit")
        require(math.isfinite(nll), f"nonfinite NLL {key}")
        result[key] = (target, target_position, nll,
                       row.get("prompt_tokens"), row.get("start"))
    require(result, f"empty NLL file {path}")
    return result


def load_map(path):
    result = {}
    used_indexes = set()
    for row in rows(path):
        key = identity(row)
        require(key not in result, f"duplicate telemetry map session {key}")
        prompt = integer(row["prompt_tokens"], "prompt_tokens", 1)
        require(integer(row["start"], "map start") == 0, f"map start must be zero: {key}")
        indexes = row["indexes"]
        require(type(indexes) is list and indexes, f"missing telemetry indexes: {key}")
        for name in indexes:
            require(type(name) is str and Path(name).is_absolute(),
                    f"index must be an absolute path: {name}")
            normalized = str(Path(name).resolve())
            require(normalized not in used_indexes, f"duplicate telemetry index: {name}")
            used_indexes.add(normalized)
        result[key] = (prompt, indexes)
    require(result, f"empty telemetry map {path}")
    return result


def load_index(path, prompt, seen_passes):
    index_path = Path(path)
    obj = read_object(index_path)
    require(obj["schema"] == "gufo-prefill-telemetry-v1", f"bad schema: {path}")
    name = obj["binary"]
    require(type(name) is str and name and Path(name).name == name, f"bad binary name: {path}")
    layer_name = obj["layer_records"]
    require(type(layer_name) is str and layer_name and Path(layer_name).name == layer_name,
            f"bad layer_records name: {path}")
    pass_id = integer(obj["pass"], "pass")
    # Sequence numbers restart in each process. Paths are unique across
    # processes; the numeric pass still has to match the binary header.
    pass_key = str(index_path.resolve())
    require(pass_key not in seen_passes, f"duplicate telemetry index {path}")
    seen_passes.add(pass_key)
    start = integer(obj["start"], "chunk start")
    count = integer(obj["tokens"], "chunk tokens", 1)
    require(start + count <= prompt, f"chunk outside prompt: {path}")
    nrecords = integer(obj["records"], "records")
    require(obj["header_bytes"] == HEADER.size and obj["record_bytes"] == RECORD.size,
            f"bad record dimensions: {path}")
    policy = obj["policy_identity"]
    # A top-10 router tap inside a pruned run still sees changed activations
    # and gates. This analysis deliberately needs the unpruned k=10 run.
    require(policy in ("reference:10", "prefill_experts=renorm:10",
                       "prefill_experts=ref10:10"),
            f"telemetry must be the reference top-10 policy: {path}")
    layers = obj["layers"]
    require(integer(obj["population_layers"], "population_layers") == 48,
            f"invalid population layer count: {path}")
    require(type(layers) is list and layers and len(layers) == len(set(layers)) and
            all(type(x) is int and 0 <= x < 48 for x in layers), f"bad layers: {path}")
    blob = (index_path.parent / name).read_bytes()
    require(len(blob) == HEADER.size + nrecords * RECORD.size,
            f"truncated or oversized binary: {path}")
    magic, version, header_bytes, record_bytes, reserved, binary_pass = HEADER.unpack_from(blob)
    require((magic, version, header_bytes, record_bytes, reserved, binary_pass) ==
            (b"GUFOPFT1", 1, 32, 36, 0, pass_id), f"bad binary header: {path}")
    metadata = {}
    seen_layers = set()
    for row in rows(index_path.parent / layer_name):
        if "pass" in row:
            require(integer(row["pass"], "layer pass") == pass_id,
                    f"layer pass mismatch: {path}")
        layer = integer(row["layer"], "layer")
        row_start = integer(row["row_start"], "row_start")
        length = integer(row["rows"], "layer rows")
        require(layer < 48 and layer not in seen_layers, f"duplicate/invalid layer row: {path}")
        seen_layers.add(layer)
        require(start <= row_start <= start + count and row_start + length <= start + count,
                f"layer row outside chunk: {path}")
        require(length > 0 and row_start + length == start + count and
                (row_start == start or layer == 47),
                f"invalid trimmed layer coverage: {path}")
        if layer in layers:
            metadata[layer] = set(range(row_start, row_start + length))
    require(seen_layers == set(range(48)), f"missing population layer metadata: {path}")
    require(set(metadata) == set(layers), f"missing layer row metadata: {path}")
    found = defaultdict(set)
    result = []
    for offset in range(HEADER.size, len(blob), RECORD.size):
        token, layer, *values = RECORD.unpack_from(blob, offset)
        require(layer in metadata and token in metadata[layer],
                f"record outside layer coverage: {path} token={token} layer={layer}")
        require(token not in found[layer], f"duplicate telemetry record: {path} {layer}:{token}")
        found[layer].add(token)
        result.append({"position": token, "layer": layer,
                       **{field: finite(value, field) for field, value in zip(FIELDS, values)}})
    require(all(found[layer] == metadata[layer] for layer in layers),
            f"layer row count/positions mismatch: {path}")
    return result, {layer: set(range(start, start + count)) for layer in layers}, policy


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        average = (left + right - 1) / 2 + 1
        for i in order[left:right]:
            result[i] = average
        left = right
    return result


def spearman(x, y):
    require(len(x) == len(y), "rank vector mismatch")
    a, b = ranks(x), ranks(y)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((u - mean_a) * (v - mean_b) for u, v in zip(a, b))
    denominator = math.sqrt(sum((u - mean_a) ** 2 for u in a) *
                            sum((v - mean_b) ** 2 for v in b))
    return numerator / denominator if denominator else None


def auc(scores, labels):
    pos = [s for s, label in zip(scores, labels) if label]
    neg = [s for s, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return None
    return sum(1 if p > n else 0.5 if p == n else 0
               for p in pos for n in neg) / (len(pos) * len(neg))


def metrics(session_rows, threshold):
    labels = [row["delta_nll"] > threshold for row in session_rows]
    result = {}
    for signal in SIGNALS:
        values = [row[signal] for row in session_rows]
        deltas = [row["delta_nll"] for row in session_rows]
        result[signal] = {"spearman": spearman(values, deltas),
                          "auc": auc(values, labels),
                          "auc_flipped": auc([-v for v in values], labels)}
    return result


def join(map_path, reference_path, candidate_path, threshold=0, token_out=None):
    threshold = finite(threshold, "threshold")
    mapping = load_map(map_path)
    reference, candidate = load_nll(reference_path), load_nll(candidate_path)
    require(reference.keys() == candidate.keys(),
            f"NLL key mismatch: reference-only={list(reference.keys() - candidate.keys())[:3]} "
            f"candidate-only={list(candidate.keys() - reference.keys())[:3]}")
    require({key[:2] for key in reference} == set(mapping), "telemetry/NLL session key mismatch")
    grouped = defaultdict(list)
    for key in reference:
        ref, cand = reference[key], candidate[key]
        require(ref[:2] == cand[:2], f"target mismatch: {key}")
        prompt = mapping[key[:2]][0]
        for item in (ref, cand):
            if item[3] is not None:
                require(integer(item[3], "NLL prompt_tokens") == prompt,
                        f"prompt boundary mismatch: {key}")
            if item[4] is not None:
                require(integer(item[4], "NLL start") == prompt,
                        f"NLL start mismatch: {key}")
        if key[2] >= prompt:  # target_position >= prompt start + 1
            grouped[key[:2]].append(cand[2] - ref[2])
    require(all(grouped.get(key) for key in mapping), "missing scored continuation NLL")
    seen_passes = set()
    by_layer = defaultdict(list)
    sessions = []
    prompt_view = []
    policies = set()
    for key, (prompt, indexes) in mapping.items():
        delta = sum(grouped[key]) / len(grouped[key])
        sessions.append({"manifest_sha256": key[0], "session": key[1],
                         "prompt_tokens": prompt, "scored_targets": len(grouped[key]),
                         "delta_nll": delta})
        records, expected = defaultdict(list), defaultdict(set)
        for index in indexes:
            chunk, coverage, policy = load_index(index, prompt, seen_passes)
            policies.add(policy)
            for layer, positions in coverage.items():
                require(not expected[layer].intersection(positions),
                        f"overlapping chunks: {key} layer {layer}")
                expected[layer].update(positions)
            for record in chunk:
                records[record["layer"]].append(record)
        require(records, f"missing telemetry records: {key}")
        for layer, entries in records.items():
            positions = [entry["position"] for entry in entries]
            require(len(positions) == len(set(positions)), f"duplicate telemetry position: {key} {layer}")
            coverage = expected[layer]
            require(coverage == set(range(prompt)), f"incomplete prompt chunks: {key} layer {layer}")
            actual = set(positions)
            require(actual.issubset(coverage), f"position outside prompt: {key} {layer}")
            # Each chunk's complete observed range is checked against its
            # layer record above. Final-layer prefix trims can repeat in every
            # chunk, so the union need not be one suffix of the whole prompt.
            summary = {signal: sum(entry[signal] for entry in entries) / len(entries)
                       for signal in SIGNALS}
            by_layer[layer].append({"manifest_sha256": key[0], "session": key[1],
                                    "delta_nll": delta, **summary,
                                    "observed_tokens": len(entries),
                                    "expected_tokens": len(coverage)})
            for entry in entries:
                prompt_view.append({"manifest_sha256": key[0], "session": key[1],
                                    "position": entry["position"],
                                    "target_position": entry["position"] + 1,
                                    "delta_nll": delta, "label_scope":
                                    "session_broadcast_not_token_causal", **entry})
        require(set(records) == set(expected), f"layer without telemetry records: {key}")
    require(len(policies) == 1, f"mixed telemetry policy identities: {sorted(policies)}")
    layers = {}
    for layer, entries in sorted(by_layer.items()):
        layers[str(layer)] = {"rows": len(entries), "session_rows": entries,
                              "coverage": {
            "observed_tokens": sum(r["observed_tokens"] for r in entries),
            "expected_tokens": sum(r["expected_tokens"] for r in entries),
            "sessions": len(entries)}, **metrics(entries, threshold)}
    report = {"label": "candidate_minus_reference_mean_continuation_nll",
              "excluded_target_position": "prompt_tokens",
              "positive_label": f"delta_nll>{threshold}",
              "policy_identity": next(iter(policies)), "sessions": sessions,
              "layers": layers}
    if token_out is not None:
        with open(token_out, "w", encoding="utf-8") as stream:
            for row in prompt_view:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-map", required=True)
    parser.add_argument("--reference-nll", required=True)
    parser.add_argument("--candidate-nll", required=True)
    parser.add_argument("--prompt-token-out")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        report = join(args.telemetry_map, args.reference_nll, args.candidate_nll,
                      args.threshold, args.prompt_token_out)
        print(json.dumps(report, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        parser.exit(2, f"INVALID: {exc}\n")


if __name__ == "__main__":
    main()
