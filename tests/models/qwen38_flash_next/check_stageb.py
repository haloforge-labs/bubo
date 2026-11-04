"""CPU-only completeness, sample-file and token-range gate for Stage B."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from derive_fold import read_totals, count_samples
from stageb_samples import ROW, MAGIC, decode

PINNED_SHA256 = 'cbb55a66c0a43866a6b203252af389b39b5b91b2eaf90f889c6c9b4cc72daab9'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_samples(root, records):
    root = Path(root).resolve()
    files, samples = {}, 0
    for row in records:
        path = (root / row['row_file']).resolve()
        require(path.is_relative_to(root) and path.name == 'rows.bin', 'sample path escapes capture root')
        files.setdefault(path, []).append(row)
    require({p.resolve() for p in root.glob('stageb-pass-*')} == {p.parent for p in files},
            'orphan/unpublished stage B pass directory')
    for path, rows in files.items():
        data = path.read_bytes()
        require(data[:8] == MAGIC, 'bad sample magic')
        cursor = 8
        for row in rows:
            count = count_samples(row['row_start'], row['rows_available'], row['sample_every'])
            require(row['row_count'] == count and row['record_bytes'] == ROW.size and
                    row['row_offset'] == cursor, 'sample count/offset/layout mismatch')
            first = ((row['row_start'] + row['sample_every'] - 1) // row['sample_every']) * row['sample_every']
            for i in range(count):
                require(cursor + ROW.size <= len(data), 'truncated sample record')
                token, layer, *v = decode(data, cursor, token=first + i*row['sample_every'], layer=row['layer'])
                require(token == first + i*row['sample_every'] and layer == row['layer'],
                        'sample token/layer mismatch')
                require(all(math.isfinite(x) for x in v), 'nonfinite sample')
                require(all(0 <= x <= 2 for x in v[:4]) and 0 <= v[4] <= 4.00001 and
                        abs(v[4] - math.sqrt(sum(g*g for g in v[:4]))) < 1e-4 and
                        abs(v[5] - max(v[:4])) < 4e-5, 'invalid sample write gates')
                require(0 <= v[6] <= v[7]+1e-5 <= 1.00002 and min(v[8:13]) >= 0 and
                        all(-1e-5 <= x <= 1.00001 for x in v[13:]) and
                        abs(v[7] - (1-v[15])) < 1e-5, 'invalid sample masses/norms')
                cursor += ROW.size
                samples += 1
        require(cursor == len(data), 'extra or incomplete sample bytes')
    return samples


def validate_coverage(records, prompt_lengths, allow_final_trim=False):
    passes = {}
    for row in records:
        passes.setdefault(row['pass_index'], []).append(row)
    ordered = iter(sorted(passes.items()))
    total_passes = 0
    for prompt in prompt_lengths:
        cursor = 0
        while cursor < prompt:
            try:
                _, rows = next(ordered)
            except StopIteration as error:
                raise ValueError('missing prompt calibration pass') from error
            tokens = min(4096, prompt-cursor)
            require(len(rows) == 48 and {r['layer'] for r in rows} == set(range(48)), 'missing calibration layers')
            for r in rows:
                require(r['start'] == cursor and r['tokens'] == tokens and r['stride'] == 1,
                        'calibration prompt/chunk/stride coverage mismatch')
                if allow_final_trim and r['layer'] == 47:
                    require(r['row_start'] + r['rows_available'] == cursor+tokens and
                            (r['row_start']-cursor) % 128 == 0 and
                            r['rows_available'] >= min(tokens, 96), 'invalid final-layer trim')
                else:
                    require(r['row_start'] == cursor and r['rows_available'] == tokens,
                            'full-prompt calibration row missing')
            cursor += tokens
            total_passes += 1
    require(next(ordered, None) is None, 'unexpected extra calibration passes')
    return total_passes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tokens', type=int)
    group.add_argument('--manifest', type=Path)
    p.add_argument('--manifest-sha256', default=PINNED_SHA256)
    p.add_argument('--allow-final-trim', action='store_true')
    p.add_argument('--map-out', type=Path)
    args = p.parse_args()
    try:
        records = read_totals([args.directory / 'stageb-totals.jsonl'])
        if args.manifest:
            require(hashlib.sha256(args.manifest.read_bytes()).hexdigest() == args.manifest_sha256,
                    'manifest hash mismatch')
            manifest = json.loads(args.manifest.read_bytes())
            sessions = manifest['sessions']
            lengths = [s['start'] for s in sessions]
            require(sum(s['stop']-s['start'] for s in sessions) == 5504 and len(sessions) == 43,
                    'wrong pinned NLL manifest dimensions')
        else:
            require(args.tokens > 0, 'positive prompt token count required')
            lengths = [args.tokens]
        passes = validate_coverage(records, lengths, args.allow_final_trim)
        samples = validate_samples(args.directory, records)
        if args.map_out:
            require(args.manifest is not None, 'map output requires manifest')
            by_pass = {}
            for row in records:
                by_pass.setdefault(row['pass_index'], row)
            pending = iter(sorted(by_pass.items()))
            mapping = []
            for session in sessions:
                ids, cursor = [], 0
                while cursor < session['start']:
                    pass_id, row = next(pending)
                    ids.append(pass_id); cursor += row['tokens']
                mapping.append(dict(manifest_sha256=args.manifest_sha256, session=session['name'],
                                    start=0, prompt_tokens=session['start'], passes=ids,
                                    totals=str((args.directory/'stageb-totals.jsonl').resolve()),
                                    router_weight_basis='original-top10-normalized'))
            args.map_out.write_text(''.join(json.dumps(m)+'\n' for m in mapping), encoding='utf-8', newline='\n')
        print(json.dumps(dict(status='PASS', k10_identity=True, passes=passes, layer_records=len(records),
                              samples=samples, prompt_tokens=sum(lengths),
                              binary_bytes=sum(p.stat().st_size for p in args.directory.glob('stageb-pass-*/rows.bin')))))
        return 0
    except (OSError, ValueError, KeyError, struct.error, StopIteration) as error:
        print(f'INVALID: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
