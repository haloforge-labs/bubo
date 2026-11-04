"""CPU completion checks. No probe is imported or executed by this module."""
import argparse
import hashlib
import json
from pathlib import Path

PINNED_SHA256 = 'cbb55a66c0a43866a6b203252af389b39b5b91b2eaf90f889c6c9b4cc72daab9'
DEFAULT_LAYERS = (3, 15, 27, 39, 47)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_manifest(path, digest=PINNED_SHA256, c=4):
    from join_telemetry import integer, read_object
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(actual == digest.lower(), f'manifest SHA256 mismatch: expected={digest} actual={actual}')
    manifest = read_object(path)
    sessions, expected = {}, {}
    for session in manifest['sessions']:
        name = session['name']
        require(isinstance(name, str) and name and name not in sessions, 'duplicate/invalid manifest session')
        start = integer(session['start'], 'manifest start', 1)
        stop = integer(session['stop'], 'manifest stop', start + 1)
        tokens = session['tokens']
        require(isinstance(tokens, list) and stop < len(tokens), f'manifest tokens too short: {name}')
        sessions[name] = start
        for position in range(start, stop):
            expected[(actual, name, position)] = integer(tokens[position + 1], 'manifest target')
    require(sessions, 'empty manifest sessions')
    schedule = set()
    grouped_indices = []
    all_sessions = manifest['sessions']
    for group in manifest['schedules'][str(c)]:
        indices = group['sessions']
        require(isinstance(indices, list) and 0 < len(indices) <= c and
                all(type(i) is int and 0 <= i < len(all_sessions) for i in indices) and
                len(indices) == len(set(indices)), 'frozen group has wrong concurrency/indices')
        # The final incomplete group is deliberately legal (43 sessions at c=4).
        offset = len(grouped_indices)
        require(indices == list(range(offset, min(offset + c, len(all_sessions)))),
                'frozen group order/coverage mismatch')
        grouped_indices.extend(indices)
        cursors = {i: all_sessions[i]['start'] for i in indices}
        for row in group['passes']:
            require(len(row) == len(indices), 'frozen pass has wrong concurrency')
            for i, chunk in zip(indices, row):
                position = integer(chunk['position'], 'frozen position')
                width = integer(chunk['width'], 'frozen width', 1)
                session = all_sessions[i]
                require(width <= 8 and position == cursors[i] and position + width <= session['stop'],
                        f'frozen schedule does not cover declared range: {session["name"]} position={position}')
                for p in range(position, position + width):
                    key = (actual, session['name'], p)
                    require(key not in schedule, f'duplicate frozen row: {key}')
                    schedule.add(key)
                cursors[i] += width
    missing = set(expected) - schedule
    require(schedule == set(expected) and grouped_indices == list(range(len(all_sessions))),
            f'frozen schedule does not cover declared positions: missing={sorted(missing)[:8]}')
    if actual == PINNED_SHA256:
        require(len(sessions) == 43 and len(expected) == 5504 and
                all(s['stop'] - s['start'] == 128 for s in all_sessions), 'pinned manifest dimensions changed')
    return actual, sessions, expected


def validate_nll(path, manifest, loaded=None):
    from join_telemetry import load_nll
    digest, sessions, expected = manifest
    actual = load_nll(path) if loaded is None else loaded
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    require(not missing and not extra,
            f'NLL key mismatch in {path}: missing NLL rows ({len(missing)})={sorted(missing)[:12]}; '
            f'extra NLL rows ({len(extra)})={sorted(extra)[:8]}')
    for key, row in actual.items():
        require(row[0] == expected[key] and row[1] == key[2] + 1,
                f'manifest target mismatch: {path} {key} expected={expected[key]} actual={row[0]}')
        require(type(row[3]) is int and type(row[4]) is int and
                row[3] == sessions[key[1]] and row[4] == sessions[key[1]],
                f'prompt boundary mismatch: {path} {key} expected={sessions[key[1]]}')
    return {'file': str(path), 'bytes': Path(path).stat().st_size, 'records': len(actual),
            'expected_records': len(expected), 'manifest_sha256': digest}


def validate_map(path, manifest, loaded=None):
    from join_telemetry import load_map
    digest, sessions, _ = manifest
    mapping = load_map(path) if loaded is None else loaded
    expected = {(digest, name) for name in sessions}
    require(set(mapping) == expected,
            f'map session mismatch: missing={sorted(expected-set(mapping))[:12]} extra={sorted(set(mapping)-expected)[:8]}')
    for key, (prompt, _) in mapping.items():
        require(prompt == sessions[key[1]], f'map prompt boundary mismatch: {key}')
    return mapping


def publication_check(index_path, obj):
    """Index must be complete, last, and have no abandoned partial sibling."""
    path = Path(index_path)
    data = path.read_bytes()
    require(data and data.endswith(b'\n') and len(data.splitlines()) == 1,
            f'partial/multiline telemetry index: {path}')
    require(not list(path.parent.glob('*.partial')), f'unpublished partial telemetry index: {path}')
    for name in (obj['binary'], obj['layer_records']):
        dependency = path.parent / name
        require(dependency.stat().st_size > 0, f'empty telemetry dependency: {dependency}')
        require(path.stat().st_mtime_ns >= dependency.stat().st_mtime_ns,
                f'index-last publication violation: {path} predates {dependency.name}')


def validate_telemetry(mapping, *, chunk_size=None, full_prompt=False, layers=None):
    from join_telemetry import load_index, read_object
    seen, total_records, total_indexes = set(), 0, 0
    for key, (prompt, indexes) in mapping.items():
        chunks, found = [], {}
        for path in indexes:
            records, coverage, _ = load_index(path, prompt, seen)
            obj = read_object(path)
            publication_check(path, obj)
            if layers is not None:
                require(set(obj['layers']) == set(layers), f'selected layer mismatch: {path}')
            chunks.append((obj['start'], obj['tokens']))
            for layer, positions in coverage.items():
                require(not found.setdefault(layer, set()).intersection(positions), f'overlapping chunks: {key}')
                found[layer].update(positions)
            if full_prompt:
                expected_count = obj['tokens'] * len(obj['layers'])
                require(len(records) == expected_count,
                        f'incomplete full-prompt telemetry: {path} expected={expected_count} actual={len(records)}')
            total_records += len(records)
            total_indexes += 1
        require(found and all(v == set(range(prompt)) for v in found.values()),
                f'incomplete prompt chunks: {key}')
        if chunk_size:
            expected_chunks = [(start, min(chunk_size, prompt-start)) for start in range(0, prompt, chunk_size)]
            require(sorted(chunks) == expected_chunks, f'manifest chunk/index count mismatch: {key} expected={expected_chunks} actual={chunks}')
    return {'sessions': len(mapping), 'indexes': total_indexes, 'binary_records': total_records,
            'population_records': 48 * total_indexes}


def validate_root(root, mapping):
    root = Path(root)
    require(root.is_dir(), f'missing telemetry directory: {root}')
    mapped = {Path(path).resolve() for _, indexes in mapping.values() for path in indexes}
    published = {path.resolve() for path in root.rglob('telemetry.index.json')}
    require(mapped == published, f'telemetry root/index set mismatch: unmapped={sorted(map(str, published-mapped))[:8]} missing={sorted(map(str,mapped-published))[:8]}')
    require(not list(root.rglob('*.partial')), f'unpublished partial telemetry index: {root}')
    for name in ('tokens.bin', 'layers.jsonl'):
        for path in root.rglob(name):
            require((path.parent/'telemetry.index.json').resolve() in mapped,
                    f'unpublished telemetry dependency: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', default=PINNED_SHA256)
    parser.add_argument('--manifest-only', action='store_true', help='explicit hash/schedule preflight; validates no output files')
    parser.add_argument('--nll', action='append', default=[])
    parser.add_argument('--telemetry-map')
    parser.add_argument('--telemetry-root')
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        require(args.manifest_only or args.nll, 'no NLL outputs requested; use --manifest-only for preflight')
        require(not args.manifest_only or not (args.nll or args.telemetry_map or args.telemetry_root),
                '--manifest-only cannot validate outputs')
        require(not args.telemetry_root or args.telemetry_map, '--telemetry-root requires --telemetry-map')
        require(args.chunk_size > 0, 'invalid chunk size')
        manifest = read_manifest(args.manifest, args.manifest_sha256)
        report = {'status': 'PASS', 'manifest_sha256': manifest[0], 'sessions': len(manifest[1]),
                  'expected_nll_records': len(manifest[2]),
                  'nll': [validate_nll(path, manifest) for path in args.nll]}
        if args.telemetry_map:
            mapping = validate_map(args.telemetry_map, manifest)
            if args.telemetry_root:
                validate_root(args.telemetry_root, mapping)
            report['telemetry'] = validate_telemetry(mapping, chunk_size=args.chunk_size,
                                                      full_prompt=True, layers=DEFAULT_LAYERS)
            report['map'] = {'records': len(mapping), 'bytes': Path(args.telemetry_map).stat().st_size}
        print(json.dumps(report, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(2, f'INVALID: {exc}\n')


if __name__ == '__main__':
    main()
