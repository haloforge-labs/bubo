"""Pool Stage B routed-sum totals and write per-layer closed-form fold scalars."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

KS = (2, 3, 4, 6, 10)
BASIS = 'original-top10-normalized'


def count_samples(start, rows, stride):
    return (start + rows - 1) // stride - (start - 1) // stride


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label}: expected finite number')
    return value


def read_totals(paths):
    records, seen, passes = [], set(), {}
    for path in map(Path, paths):
        data = path.read_bytes()
        if not data or not data.endswith(b'\n'):
            raise ValueError(f'{path}: empty or partial final JSONL record')
        for number, line in enumerate(data.splitlines(), 1):
            try:
                row = json.loads(line)
                if row['schema'] != 'gufo-prefill-stageb-v1' or row['router_weight_basis'] != BASIS:
                    raise ValueError('wrong schema/router weight basis')
                for key in ('pass_index', 'layer', 'start', 'tokens', 'row_start',
                            'rows_available', 'rows', 'stride', 'sample_every'):
                    if type(row[key]) is not int or row[key] < 0:
                        raise ValueError(f'invalid {key}')
                if row['layer'] >= 48 or not row['tokens'] or not row['stride'] or not row['sample_every']:
                    raise ValueError('invalid pass/layer/sampling')
                if not (row['start'] <= row['row_start'] < row['start'] + row['tokens'] and
                        row['row_start'] + row['rows_available'] == row['start'] + row['tokens']):
                    raise ValueError('invalid layer token range')
                expected = count_samples(row['row_start'], row['rows_available'], row['stride'])
                if row['rows'] != expected:
                    raise ValueError('wrong sampled row count')
                selected = row.get('selected_layers', list(range(48)))
                if (not selected or any(type(l) is not int or not 0 <= l < 48 for l in selected) or
                        len(set(selected)) != len(selected) or row['layer'] not in selected):
                    raise ValueError('invalid selected layers')
                # Pass counters are process-local. Separate capture files may
                # both start at pass 1; the same input file twice is rejected.
                source = str(path.resolve())
                key = source, row['pass_index'], row['layer']
                if key in seen:
                    raise ValueError(f'duplicate pass/layer {key}')
                seen.add(key)
                signature = (row['start'], row['tokens'], row['stride'], row['sample_every'], tuple(selected))
                previous, layers = passes.setdefault((source, row['pass_index']), (signature, set()))
                if previous != signature:
                    raise ValueError('pass metadata disagrees')
                layers.add(row['layer'])
                if set(row['k']) != set(map(str, KS)):
                    raise ValueError('missing/extra candidate k')
                for field in ('top1_mass', 'top4_mass', 'gate_l2'):
                    finite(row[field], field)
                n = row['rows']
                if not (0 <= row['top1_mass'] <= row['top4_mass'] + 1e-6*n and
                        row['top4_mass'] <= n*(1+1e-5) and 0 <= row['gate_l2'] <= n*4.00001):
                    raise ValueError('invalid cheap-field totals')
                for k in KS:
                    t = row['k'][str(k)]
                    for field in ('rf', 'rr', 'ff', 'error2', 'omitted_mass'):
                        finite(t[field], f'k={k} {field}')
                    if min(t['rr'], t['ff'], t['error2']) < 0 or not -1e-5*n <= t['omitted_mass'] <= n*(1+1e-5):
                        raise ValueError('negative energy or invalid omitted mass')
                    scale = max(t['ff'], t['rr'], abs(t['rf']), 1e-30)
                    if abs(t['error2'] - (t['ff'] + t['rr'] - 2*t['rf'])) > 1e-5*scale:
                        raise ValueError('error energy disagrees with dot products')
                    if abs(t['rf']) > math.sqrt(t['rr']*t['ff']) + 1e-5*scale:
                        raise ValueError('Cauchy-Schwarz identity violated')
                full = row['k']['10']
                scale = max(full['ff'], 1e-30)
                if (abs(full['rf'] - full['ff']) > 1e-4*scale or
                        abs(full['rr'] - full['ff']) > 1e-4*scale or
                        full['error2'] > 1e-8*scale or abs(full['omitted_mass']) > 1e-5*max(n, 1)):
                    raise ValueError('k=10 identity failed')
                if any(abs(row['k'][str(k)]['ff'] - full['ff']) > 1e-8*scale for k in KS):
                    raise ValueError('candidate full energy disagrees')
                row['_source'] = source
                records.append(row)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    for p, (signature, layers) in passes.items():
        if layers != set(signature[-1]):
            raise ValueError(f'incomplete pass {p}: missing layers {set(signature[-1]) - layers}')
    if not records:
        raise ValueError('no calibration records')
    return records


def derive(records):
    by_layer = {}
    for row in records:
        layer = by_layer.setdefault(row['layer'], {k: {field: [] for field in
                    ('rf', 'rr', 'ff', 'error2', 'omitted_mass', 'rows')} for k in KS})
        for k in KS:
            for field in layer[k]:
                layer[k][field].append(row['rows'] if field == 'rows' else row['k'][str(k)][field])
    if set(by_layer) != set(range(48)):
        raise ValueError('fold derivation requires all 48 layers')
    result = {}
    for layer, candidates in sorted(by_layer.items()):
        result[layer] = {}
        for k, terms in candidates.items():
            t = {field: math.fsum(values) for field, values in terms.items()}
            if t['rr'] <= 0 or t['rows'] <= 0:
                raise ValueError(f'layer {layer} k={k}: zero denominator/rows')
            c = t['rf'] / t['rr']
            mass = 1 - t['omitted_mass'] / t['rows']
            if not math.isfinite(c) or c <= 0 or not 0 < mass <= 1.00001:
                raise ValueError(f'layer {layer} k={k}: nonpositive/nonfinite scalar or mass')
            if k == 10 and abs(c - 1) > 1e-4:
                raise ValueError('pooled k=10 scalar identity failed')
            result[layer][k] = dict(c=c, renorm=1/mass, **t)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('totals', nargs='+', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        records = read_totals(args.totals)
        if args.check_only:
            print(json.dumps(dict(status='PASS', identity_k10=True, records=len(records),
                                  passes=len({(r['_source'], r['pass_index']) for r in records}))))
            return 0
        if args.out is None:
            raise ValueError('--out is required to write fold files')
        result = derive(records)  # Validate everything before writing any fold.
        summary = dict(schema='gufo-fold-stageb-v1', inputs=list(map(str, args.totals)),
                       layers=result, files={}, statistics={})
        args.out.mkdir(parents=True, exist_ok=True)
        for k in KS:
            values = [result[l][k]['c'] for l in range(48)]
            data = ''.join(f'{value:.9g}\n' for value in values).encode('ascii')
            name = f'fold-L{k}.txt'
            (args.out / name).write_bytes(data)
            stats = dict(min=min(values), median=statistics.median(values), max=max(values))
            summary['statistics'][k] = stats
            summary['files'][k] = dict(path=name, sha256=hashlib.sha256(data).hexdigest())
            print(f'k={k}: min={stats["min"]:.9g} median={stats["median"]:.9g} max={stats["max"]:.9g}')
            for l, value in enumerate(values):
                print(f'layer={l:02d} k={k} c*={value:.9g} renorm=1/mean(top-k mass)={result[l][k]["renorm"]:.9g}')
                if not .5 <= value <= 2:
                    print(f'WARNING layer={l} k={k} c*={value:.9g} outside [0.5, 2.0]')
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
        print('PASS k=10 identity; wrote five 48-line LF fold files and summary.json')
        return 0
    except (OSError, ValueError, OverflowError) as error:
        print(f'INVALID: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
