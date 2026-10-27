"""Manual HTTP gate; no model files, device APIs or internal dumps are read."""
import argparse
import json
import math
from pathlib import Path
import sys
import urllib.request
import urllib.error

SEED = 1234567
PROMPT = 'List the integers from one to fifty, separated by spaces. Start now.'


def check(ok, label):
    print(('PASS ' if ok else 'FAIL ') + label, flush=True)
    if not ok:
        raise AssertionError(label)


def http(url, path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def post(url, path, body):
    status, raw = http(url, path, body)
    check(status == 200, f'{path}: HTTP {status}, {raw[:120]!r}')
    return raw


def frames(raw):
    return [json.loads(line[6:]) for line in raw.decode('utf-8').splitlines()
            if line.startswith('data: ') and line[6:] != '[DONE]']


def chat_result(raw, stream, scored):
    if not stream:
        response = json.loads(raw)
        choice = response['choices'][0]
        return (choice['message']['content'], response['gufo']['token_ids'],
                (choice.get('logprobs') or {}).get('content', []), response['usage'])
    values = frames(raw)
    check(not any('error' in value for value in values), 'no stream error event')
    text, records, ids, usage = '', [], None, None
    scored_chunks = 0
    for value in values:
        if 'gufo' in value and 'token_ids' in value['gufo']:
            ids = value['gufo']['token_ids']
        if 'usage' in value:
            usage = value['usage']
        for choice in value.get('choices', []):
            text += choice.get('delta', {}).get('content', '')
            scores = (choice.get('logprobs') or {}).get('content', [])
            if scores:
                check(len(scores) == 1 and choice['finish_reason'] is None,
                      'one live token score per chat chunk')
                scored_chunks += 1
                records.extend(scores)
    check(ids is not None and usage is not None, 'terminal actual IDs and requested usage')
    if scored:
        check(scored_chunks == len(ids), 'one scored chunk per published token')
    check(raw.endswith(b'data: [DONE]\n\n'), 'chat DONE terminator')
    return text, ids, records, usage


def score_gate(args):
    params = dict(model='debug-gate', messages=[dict(role='user', content=PROMPT)],
                  seed=SEED, max_tokens=24, top_k=40, top_p=1.0, min_p=0.0,
                  repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                  cache_prompt=False, debug=dict(timeline=True))
    status, settings = http(args.url, '/debug/settings')
    check(status == 200, 'loopback settings snapshot available')
    (args.out / 'score-settings.json').write_bytes(settings)
    # top_k=40 exercises the eligible GPU candidate shortcut OFF versus full rows ON.
    prepared = json.loads(post(args.url, '/debug/prompt', params))
    prefix, prefix_ids = prepared['text'], prepared['token_ids']
    check(json.loads(post(args.url, '/debug/prompt', dict(text=prefix)))['token_ids'] == prefix_ids,
          'prepared text round-trips to the exact model prompt IDs')
    for temperature in (1.0, 0.0):
        for stream in (False, True):
            label = f't{temperature}-stream{int(stream)}'
            request = dict(params, temperature=temperature, stream=stream)
            if stream:
                request['stream_options'] = dict(include_usage=True)
            plain_raw = post(args.url, '/v1/chat/completions', request)
            scored_raw = post(args.url, '/v1/chat/completions',
                              dict(request, logprobs=True, top_logprobs=0))
            (args.out / f'{label}.off.bytes').write_bytes(plain_raw)
            (args.out / f'{label}.on.bytes').write_bytes(scored_raw)
            plain = chat_result(plain_raw, stream, False)
            scored = chat_result(scored_raw, stream, True)
            check(plain[:2] == scored[:2], f'{label}: exact text and published IDs OFF == ON')
            check(plain[3]['draft_tokens'] > 0 and scored[3]['draft_tokens'] > 0,
                  f'{label}: MTP actually proposed drafts on both requests')
            check(len(scored[2]) == len(scored[1]), f'{label}: complete score alignment')
            full_text = prefix + scored[0]
            full = json.loads(post(args.url, '/debug/prompt', dict(text=full_text)))
            check(full['token_ids'] == prefix_ids + scored[1],
                  f'{label}: echo text has exact prefix + output IDs')
            echo = json.loads(post(args.url, '/v1/completions',
                dict(model='debug-gate', prompt=full_text, echo=True, logprobs=0, max_tokens=0,
                     temperature=1.0, repeat_penalty=1.0,
                     frequency_penalty=0.0, presence_penalty=0.0)))
            reference = echo['choices'][0]['logprobs']['token_logprobs'][len(prefix_ids):]
            check(len(reference) == len(scored[2]), f'{label}: echo reference length')
            differences = [abs(record['logprob'] - expected)
                           for record, expected in zip(scored[2], reference)]
            check(all(math.isfinite(diff) and diff <= 1e-3 for diff in differences),
                  f'{label}: teacher-force recomputation <= 1e-3; max={max(differences, default=0)}')


def fixtures():
    yield 'health', '/health', None
    base = dict(model='debug-gate', messages=[dict(role='user', content='Say hello.')],
                seed=SEED, temperature=0.0, max_tokens=8, cache_prompt=False)
    yield 'chat', '/v1/chat/completions', base
    yield 'chat-stream', '/v1/chat/completions', dict(base, stream=True)
    yield 'legacy', '/v1/completions', dict(model='debug-gate', prompt='Hello', seed=SEED,
                                         temperature=0.0, max_tokens=8)
    yield 'responses', '/v1/responses', dict(model='debug-gate', input='Hello', seed=SEED,
                                          temperature=0.0, max_output_tokens=8)
    for i, debug in enumerate((None, False, 17, [], {}, dict(unknown=True),
                               dict(timeline=True), dict(cache_probe=True))):
        # Deterministic error bodies have no randomized ID or measured duration.
        yield f'debug-{i}', '/v1/chat/completions', dict(debug=debug)
    yield 'chat-invalid-logprobs', '/v1/chat/completions', dict(base, logprobs='bad')
    yield 'legacy-top-rejected', '/v1/completions', dict(prompt='Hello', top_logprobs=2)
    yield 'responses-logprobs-rejected', '/v1/responses', dict(input='Hello', logprobs=True)


def capture(args):
    index = []
    for name, path, body in fixtures():
        status, raw = http(args.url, path, body)
        (args.out / f'{args.phase}-{name}.bytes').write_bytes(raw)
        index.append(dict(name=name, status=status, path=path, request=body))
        print(f'RECORDED {args.phase} {name}: HTTP {status}, {len(raw)} body bytes', flush=True)
    (args.out / f'{args.phase}-index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')


# Raw body bytes always remain saved and compared separately. Only existing
# nondeterministic IDs and measured durations are normalized in the second gate.
DYNAMIC = {'id', 'created', 'created_at', 'timings', 'prompt_tokens_per_second',
           'completion_tokens_per_second', 'queue_ms', 'ttft_ms', 'prefill_ms', 'decode_ms',
           'cache_restore_ms', 'cache_snapshot_ms', 'cache_disk_enqueue_ms', 'cache_shared_prefix_ms',
           'shared_prefix_wait_ms', 'mean_inter_token_ms', 'max_inter_token_ms'}


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in DYNAMIC}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def compare(args):
    a = json.loads((args.out / 'candidate-index.json').read_text())
    b = json.loads((args.out / 'base-index.json').read_text())
    check(a == b, 'base/candidate fixtures and HTTP statuses')
    for item in a:
        name = item['name']
        left = (args.out / f'candidate-{name}.bytes').read_bytes()
        right = (args.out / f'base-{name}.bytes').read_bytes()
        print(('PASS ' if left == right else 'FAIL ') + f'{name}: literal response-body bytes', flush=True)
        if item['status'] != 200 or name == 'health':
            check(left == right, f'{name}: deterministic golden response/error bytes')
        else:
            # A live HTTP API cannot freeze random IDs or measured durations.
            # This separate semantic gate never claims that raw bytes match.
            l = frames(left) if name.endswith('stream') else json.loads(left)
            r = frames(right) if name.endswith('stream') else json.loads(right)
            check(stable(l) == stable(r), f'{name}: exact deterministic fields after explicit normalization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'candidate', 'base', 'compare'))
    parser.add_argument('--url', default='http://127.0.0.1:18101')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.phase == 'score':
        score_gate(args)
    elif args.phase == 'compare':
        compare(args)
    else:
        capture(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'FAIL {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
