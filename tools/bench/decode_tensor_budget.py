"""Header-only decode inventory and a bandwidth-floor loss ledger.

No HIP imports or tensor payload reads. --model LABEL=FIRST_SHARD may be
repeated. --dense-log LABEL=LOG consumes cold --budget-real output.
--routing consumes the eager diagnostic export, --expert-log its real replay.
Missing cells stay missing, including norms, gathers and state operations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def classify(name, rank):
    if '_exps.weight' in name:
        return 'routed_' + name.split('.ffn_')[1].split('_exps')[0]
    if name == 'output.weight':
        return 'target_head'
    if 'token_embd' in name:
        return 'ple_gather' if name.startswith('per_layer_') else 'embedding_gather'
    if 'conv1d' in name:
        return 'ple_convolution' if '.ple_' in name else 'ssm_convolution'
    if rank != 2:
        return 'norm_scale_bias'
    if 'hc_' in name:
        return 'hc_inject' if 'inject' in name else 'hc_projection'
    if '_shexp.' in name:
        return 'shared_expert'
    if '.ple_' in name:
        return 'ple_projection'
    if 'ffn_gate_inp' in name:
        return 'router'
    if 'ssm_alpha' in name or 'ssm_beta' in name:
        return 'ssm_alpha_beta'
    if name.endswith('.q_proj.weight') or name.endswith('.k_proj.weight'):
        return 'indexer'
    return 'dense_attention'


def shards(first):
    from gufo.gguf import parse_gguf

    first = Path(first)
    header = parse_gguf(first)
    count = header['user_kv'].get('split.count', 1)
    if count == 1:
        return [(first, header)]
    match = re.fullmatch(r'(.*)-(\d+)-of-(\d+)\.gguf', first.name)
    if not match or int(match[2]) != 1 or int(match[3]) != count:
        raise ValueError('expected first shard: ' + str(first))
    result = []
    for i in range(1, count + 1):
        path = first.with_name(f'{match[1]}-{i:0{len(match[2])}d}-of-{match[3]}.gguf')
        h = header if i == 1 else parse_gguf(path)
        if h['user_kv'].get('split.no') != i - 1 or h['user_kv'].get('split.count') != count:
            raise ValueError('inconsistent shard ' + str(path))
        result.append((path, h))
    return result


def inventory(label, first):
    # Import the shared parser only for actual GGUF inventory. CPU ledger
    # policy tests need no optional NumPy/dequantization dependency.
    from gufo.gguf import GGML_TYPE_BLOCK, GGML_TYPE_NAME

    blocks = {**GGML_TYPE_BLOCK, 17: (256, 74), 18: (256, 98),
              22: (256, 82), 29: (256, 56), 42: (64, 18)}
    records, seen = [], set()
    parts = shards(first)
    ple_layers = next((v for k, v in parts[0][1]['user_kv'].items()
                       if k.endswith('.ple.layers')), [])
    prefix_layers = ple_layers[0] if ple_layers else 0
    for path, header in parts:
        for t in header['tensors']:
            name, shape, type_id = t['name'], tuple(reversed(t['shape'])), t['type']
            if name in seen:
                raise ValueError('duplicate tensor ' + name)
            seen.add(name)
            block, size = blocks[type_id]
            if shape[0] % block:
                raise ValueError('unaligned tensor ' + name)
            byte_count = math.prod(shape) // block * size
            if header['data_offset'] + t['offset'] + byte_count > header['file_size']:
                raise ValueError('tensor past EOF ' + name)
            layer = re.match(r'blk\.(\d+)\.', name)
            scope = ('verify-prefix' if (layer and int(layer[1]) < prefix_layers) or
                     name == 'token_embd.weight' else 'verify-body')
            if name == 'per_layer_token_embd.weight':
                scope = 'ple_staging'
            records.append(dict(model=label, name=name, tensor_class=classify(name, len(shape)),
                                type='Q2_0' if type_id == 42 else GGML_TYPE_NAME[type_id],
                                type_id=type_id, shape='x'.join(map(str, shape)),
                                K=shape[0], M=shape[1] if len(shape) > 1 else 1,
                                E=shape[2] if len(shape) > 2 else 1,
                                layer=int(layer[1]) if layer else -1,
                                bytes=byte_count, shard=str(path), scope=scope))
    return records


def assignment(value):
    label, sep, path = value.partition('=')
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError('expected LABEL=PATH')
    return label, Path(path)


def metadata_assignment(value):
    label, sep, data = value.partition('=')
    if not sep or not label or not data:
        raise argparse.ArgumentTypeError('expected LABEL=VALUE')
    return label, data


SAMPLE_STATS = defaultdict(int)  # (kind, label, 'kept'|'dropped:<reason>') -> count


def fields(line):
    return dict(re.findall(r'([^\s=]+)=([^\s]+)', line))


def timing_logs(paths, real, min_sets=1):
    result = defaultdict(list)
    protocols = defaultdict(set)
    for label, path in paths:
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if real and not line.startswith('BUDGET '):
                continue
            if not real and not line.startswith('fmt='):
                continue
            f = fields(line)
            if real:
                sets = int(f.get('sets', 1))
                if sets < min_sets:
                    raise ValueError('expected distinct-matrix interval, got single-call: ' + str(path))
                protocols[label].add(sets)
                if len(protocols[label]) > 1:
                    raise ValueError('do not pool different sets protocols for ' + label)
                if f.get('valid') != '1':
                    SAMPLE_STATS['dense', label, 'dropped:invalid (bench flagged, e.g. above the bandwidth peak)'] += 1
                    continue
                if f.get('weights') != 'real' or int(f.get('evict_bytes', 0)) < 256 << 20:
                    SAMPLE_STATS['dense', label, 'dropped:not real weights or no 256 MiB eviction'] += 1
                    continue
                key = (label, f['name'], int(f['N']))
            else:
                if f.get('variant') != 'existing' or 'us' not in f:
                    continue
                key = (label, f['fmt'], int(f['K']), int(f['M']), int(f['N']))
            us = float(f['us'])
            if not math.isfinite(us) or us <= 0:
                SAMPLE_STATS['dense', label, 'dropped:non-finite time'] += 1
                continue
            SAMPLE_STATS['dense', label, 'kept'] += 1
            result[key].append(us)
    return {k: statistics.median(v) for k, v in result.items()}


def expert_logs(paths):
    cells = defaultdict(list)
    for label, path in paths:
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.startswith('REAL_CELL '):
                continue
            f = fields(line)
            if f.get('valid') == '1' and f.get('source') == 'real_replay' and f.get('kind') == 'chain':
                us = float(f['us'])
                packed = float(f['unique_bytes'])
                if (math.isfinite(us) and us > 0 and math.isfinite(packed) and
                        packed > 0 and packed / us / 1000 <= 250):
                    SAMPLE_STATS['expert', label, 'kept'] += 1
                    cells[label, int(f['layer']), int(f['N'])].append(
                        (us, packed))
                else:
                    SAMPLE_STATS['expert', label, 'dropped:above 250 GB/s or non-finite'] += 1
            elif f.get('kind') == 'chain':
                SAMPLE_STATS['expert', label, 'dropped:invalid or not real replay'] += 1
    return {k: (statistics.median(v[0] for v in samples),
                statistics.median(v[1] for v in samples)) for k, samples in cells.items()}


def routing_logs(paths):
    """Distinct *valid* expert IDs, preserving layer/type correlation."""
    cells = defaultdict(list)
    for label, path in paths:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line or line.startswith('#'):
                continue
            values = list(map(int, line.split()))
            layer, n, *ids = values
            if not 0 <= layer < 48 or not 1 <= n <= 8 or len(ids) != 10 * n:
                raise ValueError(f'{path}:{number}: bad routing shape')
            if any(i < -1 or i >= 512 for i in ids):
                raise ValueError(f'{path}:{number}: bad expert ID')
            cells[label, layer, n].append(len({i for i in ids if i >= 0}))
    return {k: statistics.mean(v) for k, v in cells.items()}


def timelines(paths):
    """Keep the supplied calls/cycle, including rounded rare-width zeros."""
    result = {}
    for label, path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            m = re.match(r'verify-body w([1-8])\s+([\d.]+)\s+[\d.]+%\s+([\d.]+)\s*$', line)
            if m:
                n, busy, calls = int(m[1]), float(m[2]), float(m[3])
                result[label, n] = (calls, busy / calls if calls else None)
    return result


def control_bandwidth(paths):
    """Calibrate only cold real Q4_K N1 samples, never synthetic proxies."""
    values = []
    for _, path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('BUDGET '):
                continue
            f = fields(line)
            if (f.get('fmt') != 'Q4_K' or f.get('N') != '1' or
                    f.get('valid') != '1' or f.get('weights') != 'real' or
                    int(f.get('evict_bytes', 0)) < 256 << 20):
                continue
            byte_count = int(f['K']) * int(f['M']) / 256 * 144
            us = float(f['us'])
            if byte_count >= 4 << 20 and math.isfinite(us) and us > 0:
                rate = byte_count / us / 1000
                if 0 < rate <= 256:
                    values.append(rate)
    if not values:
        raise ValueError('no valid cold real Q4_K N1 control samples >=4 MiB')
    SAMPLE_STATS['control', 'Q4_K', 'min/median/max GB/s'] = (min(values), statistics.median(values), max(values))
    return statistics.median(values), len(values)


PROJECTIONS = {'dense_attention', 'hc_projection', 'hc_inject', 'shared_expert', 'router',
               'ssm_alpha_beta', 'indexer', 'ple_projection', 'target_head'}
MISSING = ('attention_kv_and_index_state', 'gdn_recurrence_and_snapshots',
           'activation_quantization_and_reductions', 'routing_grouping_and_combine',
           'refusal_and_candidate_selection', 'control_and_other_body_work')


def ledger(records, real, experts, routing, bandwidth):
    groups = defaultdict(list)
    for r in records:
        groups[r['model'], r['tensor_class'], r['type'], r['shape'],
               r.get('scope', 'verify-body')].append(r)
    rows = []
    for (label, cls, fmt, shape, scope), tensors in sorted(groups.items()):
        lc = Counter(t['layer'] for t in tensors if t['layer'] >= 0)
        per_layer = f'{min(lc.values())}..{max(lc.values())}/{len(lc)} layers' if lc else 'global'
        for n in range(1, 9):
            # Projection/inject bytes are a minimum packed read per invocation.
            # F32 inject is fused (its time remains unknown); native BF16 inject
            # uses Dense/SmallGemm. Gathers/state do not read full stored tensors.
            byte_count, ms, evidence = None, None, 'UNMEASURED'
            if cls in PROJECTIONS:
                byte_count = sum(t['bytes'] for t in tensors)
                values = [real.get((label, t['name'], n)) for t in tensors]
                if all(v is not None for v in values):
                    ms = sum(values) / 1000
                    evidence = 'real isolated projection; producer quantization excluded'
            elif cls.startswith('routed_'):
                ds = [routing.get((label, t['layer'], n)) for t in tensors]
                if all(d is not None for d in ds):
                    byte_count = sum(t['bytes'] / t['E'] * d for t, d in zip(tensors, ds))
                    evidence = 'real routing unique packed bytes; time belongs to routed_chain'
            floor = byte_count / bandwidth / 1e6 if byte_count is not None else None
            rows.append(dict(model=label, tensor_class=cls, type=fmt, shape=shape,
                             count=len(tensors), per_layer=per_layer, N=n,
                             stored_bytes=sum(t['bytes'] for t in tensors),
                             pass_bytes=byte_count, floor_ms=floor, measured_ms=ms,
                             lost_ms=ms-floor if ms is not None and floor is not None else None,
                             evidence=evidence, scope=scope,
                             additive_time=not cls.startswith('routed_')))
    for label, scope in sorted({(r['model'], r.get('scope', 'verify-body')) for r in records
                               if r.get('scope', 'verify-body') != 'ple_staging'}):
        for n in range(1, 9):
            active_layers = sorted({r['layer'] for r in records
                                    if r['model'] == label and r['tensor_class'] == 'routed_gate' and
                                    r.get('scope', 'verify-body') == scope})
            cells = [experts.get((label, layer, n)) for layer in active_layers]
            complete = bool(cells) and all(c is not None for c in cells)
            ms = sum(c[0] for c in cells) / 1000 if complete else None
            byte_count = sum(c[1] for c in cells) if complete else None
            floor = byte_count / bandwidth / 1e6 if complete else None
            for cls in (('routed_chain', *MISSING) if active_layers else MISSING):
                chain = cls == 'routed_chain'
                rows.append(dict(model=label, tensor_class=cls, type='mixed' if chain else '-',
                                 shape='layer chain' if chain else 'context/state dependent',
                                 count=len(active_layers) if chain else '', per_layer='', N=n,
                                 stored_bytes=None, pass_bytes=byte_count if chain else None,
                                 floor_ms=floor if chain else None,
                                 measured_ms=ms if chain else None,
                                 lost_ms=ms-floor if chain and complete else None,
                                 evidence='real replay chain' if chain and complete else 'UNMEASURED',
                                 scope=scope, additive_time=True))
    return rows


def cell(value):
    return 'NA' if value is None else f'{value:.3f}'


Q8_REFERENCE = (229, 224, 229, 220, 227, 227, 225, 226)


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def fit_call_constants(records, times, bandwidth):
    """Lower-quartile fixed-slope fit on small, non-serial projections.

    t = a + packed_bytes/BW + residual. The residual includes real kernel
    latency, so a is an apparent overhead estimate, not an identified driver
    constant. Report the distribution and validate with distinct-matrix runs.
    Exclude long-K four-row inject: its serial kernel cost is not an intercept.
    """
    fits = {}
    for label in sorted({r['model'] for r in records}):
        eligible = [r for r in records if r['model'] == label and
                    r['tensor_class'] in PROJECTIONS and
                    r['tensor_class'] != 'hc_inject' and
                    0 < int(r['bytes']) <= 2 << 20 and
                    int(r['K']) <= 2560 and int(r['M']) >= 48]
        for n in range(1, 9):
            residuals = [times[label, r['name'], n] - int(r['bytes']) / bandwidth / 1000
                         for r in eligible if (label, r['name'], n) in times]
            if len(residuals) < 8:
                continue
            fits[label, n] = dict(us=quantile(residuals, .25),
                                  p10_us=quantile(residuals, .10),
                                  p50_us=statistics.median(residuals), cells=len(residuals))
    return fits


def corrected_ledger(rows, fits):
    for row in rows:
        raw, byte_count = row['measured_ms'], row['pass_bytes']
        fit = fits.get((row['model'], row['N']))
        # Each expert-chain interval contains several launches but incurs
        # the per-interval intercept once, not once per internal launch.
        correction = (row['count'] * fit['us'] / 1000 if fit and raw is not None
                      and row['additive_time'] else None)
        net = raw - correction if correction is not None else None
        q8_floor = byte_count / Q8_REFERENCE[row['N'] - 1] / 1e6 if byte_count is not None else None
        row.update(single_call_ms=raw, overhead_ms=correction, net_ms=net,
                   q8_reference_gbps=Q8_REFERENCE[row['N'] - 1], q8_floor_ms=q8_floor,
                   net_lost_ms=net-row['floor_ms'] if net is not None and row['floor_ms'] is not None else None,
                   single_call_q8_lost_ms=raw-q8_floor if raw is not None and q8_floor is not None else None,
                   net_q8_lost_ms=net-q8_floor if net is not None and q8_floor is not None else None)
    return rows


def attach_interval_rows(rows, intervals):
    by_key = {(r['model'], r['tensor_class'], r['type'], r['shape'], r['scope'], r['N']): r
              for r in intervals}
    for row in rows:
        current = by_key.get(tuple(row[k] for k in ('model','tensor_class','type','shape','scope','N')))
        ms = current['measured_ms'] if current and row['tensor_class'] in PROJECTIONS else None
        row.update(interval_ms=ms,
                   interval_lost_ms=ms-row['floor_ms'] if ms is not None and row['floor_ms'] is not None else None,
                   interval_q8_lost_ms=ms-row['q8_floor_ms'] if ms is not None and row['q8_floor_ms'] is not None else None)


def net_report(rows, fits, timeline, bandwidth, revisions):
    lines = ['PARTIAL: raw and net-of-call-constant ranking; distinct-matrix protocol confirmation pending.']
    labels = sorted({r['model'] for r in rows},
                    key=lambda x: (x not in {'production','IQ3_S'}, x != 'production', x))
    for label in labels:
        trace = ', '.join(f'N{n}={cell(timeline.get((label,n), (None,None))[1])}' for n in (3,4,5))
        lines.append(f'{label}: whole-pass floor/loss UNKNOWN (state/traffic incomplete); traced body ms {trace}.')
    lines += ['', f'Cold single-call calibrated reference {bandwidth:.3f} GB/s; it includes the call effect.',
              'Second reference: operator FU3 Q8_0 line = ' + '/'.join(map(str, Q8_REFERENCE)) +
              ' GB/s at N1..8, a DIFFERENT protocol. Neither line is a universal physical floor.',
              'Fit: t=a+packed_bytes/BW+residual, a=25th percentile of small-tensor residuals;',
              'bytes<=2 MiB, K<=2560, M>=48, excluding long-K HC inject. Kernel latency is confounded',
              'with a. P10/P50 are empirical sensitivity bounds, not confidence intervals.',
              'Signed times/losses remain visible; the fit is provisional until sets>=16 timing.', '',
              '| Model | Measured routing revision | N | Cells | a us | P10 us | P50 us |',
              '|---|---|---:|---:|---:|---:|---:|']
    for (label, n), fit in sorted(fits.items()):
        lines.append(f'| {label} | {revisions.get(label,"UNSPECIFIED")} | {n} | {fit["cells"]} | '
                     f'{fit["us"]:.3f} | {fit["p10_us"]:.3f} | {fit["p50_us"]:.3f} |')
    sums = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r['scope'] != 'verify-body' or not r['additive_time'] or r['net_ms'] is None:
            continue
        s = sums[r['model'], r['tensor_class'], r['type'], r['N']]
        for key in ('single_call_ms', 'overhead_ms', 'net_ms', 'floor_ms', 'q8_floor_ms',
                    'lost_ms', 'net_lost_ms', 'single_call_q8_lost_ms', 'net_q8_lost_ms'):
            s[key] += r[key] or 0
        s['intervals'] += r['count']
    ranks = defaultdict(lambda: defaultdict(float))
    for (label, cls, fmt, n), values in sums.items():
        calls = timeline.get((label, n), (0,))[0]
        for key in ('lost_ms', 'net_lost_ms', 'single_call_q8_lost_ms', 'net_q8_lost_ms'):
            ranks[label, cls, fmt][key] += values[key] * calls
    lines += ['', 'Ranked measured classes, ms lost/cycle over each actual width mix (body only):', '',
              '| Rank | Model | Class | Type | Raw loss cold | Net loss cold | Raw loss Q8 line | Net loss Q8 line |',
              '|---:|---|---|---|---:|---:|---:|---:|']
    for rank, (key, values) in enumerate(sorted(ranks.items(), key=lambda x: x[1]['net_lost_ms'], reverse=True), 1):
        lines.append('| ' + ' | '.join(map(str, (rank, *key, *[cell(values[k]) for k in
                     ('lost_ms', 'net_lost_ms', 'single_call_q8_lost_ms', 'net_q8_lost_ms')]))) + ' |')
    lines += ['', 'Covered body reconciliation; unmeasured operations are still in the residual:', '',
              '| Model | N | Intervals | Raw sum ms | Removed ms | Net sum ms | Trace ms | Raw residual | Net residual |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        for n in range(1,9):
            ss = [v for (l, _, _, width), v in sums.items() if l == label and width == n]
            raw = sum(v['single_call_ms'] for v in ss)
            removed = sum(v['overhead_ms'] for v in ss)
            trace = timeline.get((label,n), (None,None))[1]
            lines.append(f'| {label} | {n} | {sum(v["intervals"] for v in ss):.0f} | {raw:.3f} | '
                         f'{removed:.3f} | {raw-removed:.3f} | {cell(trace)} | '
                         f'{cell(trace-raw if trace is not None else None)} | '
                         f'{cell(trace-raw+removed if trace is not None else None)} |')
    lines += ['', 'Class detail: single-call raw and net time, with both reference losses (all widths):', '',
              '| Model | Class | Type | N | Raw ms | Removed ms | Net ms | Cold floor | Q8 floor | Raw loss cold | Net loss cold | Raw loss Q8 | Net loss Q8 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, values in sorted(sums.items()):
        lines.append('| ' + ' | '.join(map(str, (*key, *[cell(values[k]) for k in
                     ('single_call_ms', 'overhead_ms', 'net_ms', 'floor_ms', 'q8_floor_ms',
                      'lost_ms', 'net_lost_ms', 'single_call_q8_lost_ms', 'net_q8_lost_ms')]))) + ' |')
    if any(r.get('interval_ms') is not None for r in rows):
        lines += ['', 'Matched distinct-matrix confirmation (same class/shape, current dispatch; no fitted subtraction):', '',
                  '| Model | Class | Type | Shape | N | Historical single-call ms | Fitted net ms | Distinct interval ms | Distinct loss cold | Distinct loss Q8 line |',
                  '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['scope'] == 'verify-body' and r.get('interval_ms') is not None:
                lines.append('| ' + ' | '.join(map(str, (r['model'],r['tensor_class'],r['type'],r['shape'],r['N'],
                             *[cell(r[k]) for k in ('single_call_ms','net_ms','interval_ms','interval_lost_ms','interval_q8_lost_ms')]))) + ' |')
    return '\n'.join(lines) + '\n\n'


def floor_report(records, real, experts, routing, timeline, bandwidth, bw_evidence):
    rows = ledger(records, real, experts, routing, bandwidth)
    labels = sorted({r['model'] for r in records},
                    key=lambda label: (label not in {'production', 'IQ3_S'}, label != 'production', label))
    lines = ['PARTIAL: Phase 0 needs operator measurements before any kernel target can be selected.']
    for label in labels:
        times = ', '.join(f'N{n}={cell(timeline.get((label,n), (None,None))[1])}' for n in (3, 4, 5))
        lines.append(f'{label}: whole-pass BW floor = NA (routing/state traffic incomplete); '
                     f'traced verify-body ms/pass {times}; whole-pass ms LOST = NA.')
    lines += ['', f'BW reference: {bandwidth:.3f} GB/s; {bw_evidence}. Decimal GB and MB.',
              'No GPU/server was run by this agent. Whole-pass floor is deliberately unknown.',
              'The ledger CSV contains each N=1..8, packed pass bytes, floor=bytes/BW, measured ms and ms LOST.',
              'Real isolated projections exclude producer quantization and fusion; their sum is a diagnostic,',
              'not a measured pass. hipMalloc payload placement differs from mapped production weights.',
              'Unique expert bytes are a compulsory-read lower bound, NOT measured DRAM traffic.',
              'Routed per-role byte rows and the measured routed chain overlap; add chain time only once.',
              'Historical synthetic proxies never enter this floor/loss ledger.', '',
              'Each width cell below is **packed MB / floor ms / measured ms / lost ms**. NA is never zero.', '',
              'Scope follows ForwardBody: block 0/embedding are verify-prefix; blocks 1..47/head/PLE compute',
              'are verify-body for these files (ple.layers=[1]). PLE table staging is outside both GPU phases.',
              'Counts are tensors in the indicated scope; full model counts/names are in the inventory CSV.', '',
              '| Model | Class | Type | K x M x E | Scope | Count/scope (per layer range/layers) | Stored MB | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 |',
              '|---|---|---|---|---|---:|---:|' + '---|' * 8]
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['tensor_class'], row['type'], row['shape'], row['scope']].append(row)
    for key, group in sorted(groups.items()):
        first = group[0]
        cells = [' / '.join((cell(r['pass_bytes']/1e6 if r['pass_bytes'] is not None else None),
                             cell(r['floor_ms']), cell(r['measured_ms']), cell(r['lost_ms']))) for r in group]
        stored = first['stored_bytes']
        lines.append('| ' + ' | '.join(map(str, (*key, f'{first["count"]} ({first["per_layer"]})',
                                                cell(stored/1e6 if stored is not None else None), *cells))) + ' |')
    lines += ['', 'Known projection floor and trace reconciliation (no expert-role/chain double counting):', '',
              '| Model | N | Known projection packed MB | Projection floor ms | Covered isolated + chain ms | Trace body ms | Uncovered residual ms | Measured additive classes / total | Calls/cycle |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        for n in range(1, 9):
            rs = [r for r in rows if r['model'] == label and r['N'] == n and
                  r['scope'] == 'verify-body' and r['additive_time']]
            projections = [r for r in rs if r['tensor_class'] in PROJECTIONS]
            byte_count = sum(r['pass_bytes'] for r in projections)
            measured = [r for r in rs if r['measured_ms'] is not None]
            subtotal = sum(r['measured_ms'] for r in measured) if measured else None
            calls, trace = timeline.get((label, n), (None, None))
            # With no coverage the whole measured body is the unallocated residual.
            residual = trace - (subtotal or 0) if trace is not None else None
            lines.append(f'| {label} | {n} | {byte_count/1e6:.3f} | {byte_count/bandwidth/1e6:.3f} | '
                         f'{cell(subtotal)} | {cell(trace)} | {cell(residual)} | {len(measured)}/{len(rs)} | {cell(calls)} |')
    lines += ['', 'Rank by ms lost per traced cycle (sum width loss * observed calls/cycle; signed losses retained):', '',
              '| Rank | Model | Class/type/shape | Covered loss ms/cycle | Width coverage |',
              '|---:|---|---|---:|---|']
    ranked, missing = [], []
    for key, group in groups.items():
        if not group[0]['additive_time'] or group[0]['scope'] != 'verify-body':
            continue
        relevant = [(r, timeline.get((key[0], r['N']))) for r in group
                    if timeline.get((key[0], r['N']), (0,))[0] > 0]
        if not relevant:
            missing.append(key)
            continue
        valid = [(r, mix) for r, mix in relevant if r['lost_ms'] is not None]
        if valid:
            ranked.append((sum(r['lost_ms']*mix[0] for r, mix in valid), key, len(valid), len(relevant)))
        else:
            missing.append(key)
    for rank, (loss, key, covered, total) in enumerate(sorted(ranked, reverse=True), 1):
        lines.append(f'| {rank} | {key[0]} | {" / ".join(key[1:])} | {loss:.3f} | {covered}/{total} |')
    if not ranked:
        lines.append('| NA | both | No eligible measured cells | NA | 0 |')
    lines += ['', f'{len(missing)} additive classes remain UNRANKED; see NA cells. '
              'A partial ranking does not choose a winner over unmeasured classes.',
              'Timeline calls/cycle are rounded to 0.01; rare widths rounded to zero have no ranking weight.',
              'The input mixes differ between files. This ranks losses within each actual workload,',
              'not a counterfactual native-minus-production attribution. Draft head/MTP weights are outside',
              'verify-body and are not added to its floor; production target decode mirrors are overlaid.',
              'Missing gathers, convolution/norm/inject work, PLE staging and all state/activation classes',
              'remain in the residual. An isolated subtotal above trace time signals scope/cache/fusion mismatch;',
              'do not clamp it or invent a negative-time class. Whole-pass bytes need actual routing and state traffic.', '']
    return '\n'.join(lines), rows


def write_csv(path, rows):
    if not rows:
        raise ValueError('no records for ' + str(path))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', action='append', type=assignment, default=[])
    p.add_argument('--inventory-csv', type=Path)
    p.add_argument('--fit-single-call', action='store_true')
    p.add_argument('--routing-revision', action='append', type=metadata_assignment, default=[])
    p.add_argument('--twin', action='append', type=assignment, default=[],
                   help='LABEL=TWIN_GGUF overlays only Q8 trunk mirrors for decode widths')
    p.add_argument('--dense-log', action='append', type=assignment, default=[])
    p.add_argument('--single-call-log', action='append', type=assignment, default=[],
                   help='Historical single-call column; --dense-log must then use >=16 distinct matrices')
    p.add_argument('--expert-log', action='append', type=assignment, default=[])
    p.add_argument('--routing', action='append', type=assignment, default=[])
    p.add_argument('--timeline', action='append', type=assignment, default=[])
    p.add_argument('--control-log', action='append', type=assignment, default=[])
    p.add_argument('--bandwidth-gbps', type=float, default=210.0)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    bandwidth = args.bandwidth_gbps
    bw_evidence = 'ASSUMED from operator supplied historical 206--212 GB/s range; not measured here'
    if args.control_log:
        bandwidth, samples = control_bandwidth(args.control_log)
        bw_evidence = f'MEASURED median of {samples} supplied cold real Q4_K N1 samples >=4 MiB'
    if not math.isfinite(bandwidth) or not 0 < bandwidth <= 256:
        p.error('bandwidth must be finite and in (0,256] GB/s')
    if not args.model and not args.inventory_csv:
        p.error('--model or --inventory-csv is required')
    models = {label: inventory(label, path) for label, path in args.model}
    if args.inventory_csv:
        if args.model or args.twin:
            p.error('--inventory-csv already includes mirrors; do not also pass --model/--twin')
        with args.inventory_csv.open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                for key in ('type_id', 'K', 'M', 'E', 'layer', 'bytes'):
                    row[key] = int(row[key])
                models.setdefault(row['model'], []).append(row)
    for label, path in args.twin:
        if label not in models:
            raise ValueError('twin has no matching model ' + label)
        by_name = {r['name']: r for r in models[label]}
        for r in inventory(label, path):
            old = by_name.get(r['name'])
            if not old or old['type'] != 'Q8_0' or old['shape'] != r['shape']:
                raise ValueError('invalid Q8 twin mirror ' + r['name'])
            r['scope'] = old['scope']
            by_name[r['name']] = r
        models[label] = list(by_name.values())
    records = [r for rs in models.values() for r in rs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out.with_suffix('.csv'), records)
    times = timing_logs(args.dense_log, True, 16 if args.single_call_log else 1)
    mix = timelines(args.timeline)
    report, rows = floor_report(records, times,
                                expert_logs(args.expert_log), routing_logs(args.routing),
                                mix, bandwidth, bw_evidence)
    if args.fit_single_call:
        legacy_times = timing_logs(args.single_call_log, True) if args.single_call_log else times
        fits = fit_call_constants(records, legacy_times, bandwidth)
        interval_rows = rows
        if args.single_call_log:
            rows = ledger(records, legacy_times, expert_logs(args.expert_log), routing_logs(args.routing), bandwidth)
        corrected_ledger(rows, fits)
        if args.single_call_log:
            attach_interval_rows(rows, interval_rows)
        report = net_report(rows, fits, mix, bandwidth, dict(args.routing_revision)) + report
    lines = ['', '## Sample bookkeeping (nothing is clamped; dropped samples would bias the median if they are the fast ones)', '']
    for key in sorted(SAMPLE_STATS, key=str):
        lines.append(f'- {key[0]} {key[1]}: {key[2]} = {SAMPLE_STATS[key]}')
    lines.append('- Traced body times include trace inflation (about 5-20 %); isolated untraced sums can look smaller.')
    lines.append('- The Q4_K bandwidth control is circular for Q4_K cells by construction (they show about 0 ms lost).')
    report = report + '\n'.join(lines) + '\n'
    print('\n'.join(lines))
    args.out.with_suffix('.md').write_text(report, encoding='utf-8')
    write_csv(args.out.with_name(args.out.stem + '-ledger.csv'), rows)
    print(f'Inventoried {len(records)} tensors, headers only; budget {args.out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
