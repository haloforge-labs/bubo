"""Reject drift between the public event/error inventory and the wire schema."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]

def check(document, schema):
    events = set(schema['properties']['type']['enum'])
    errors = set(schema['$defs']['error']['properties']['code']['enum'])
    documented_events = re.findall(r'^### Event `([a-z_]+)`$', document, re.M)
    documented_errors = re.findall(r'^\| `([a-z_]+)` \|',
                                  document.split('## Error codes\n', 1)[1].split('\n## ', 1)[0], re.M)
    failures = []
    for label, specified, documented in [('event', events, documented_events),
                                         ('error', errors, documented_errors)]:
        missing = specified-set(documented)
        extra = set(documented)-specified
        duplicates = len(documented)-len(set(documented))
        if missing or extra or duplicates:
            failures.append(f'{label}: missing={sorted(missing)} extra={sorted(extra)} duplicates={duplicates}')
    examples = []
    for block in re.findall(r'```jsonl\n(.*?)\n```', document, re.S):
        for line in block.splitlines():
            row = json.loads(line)
            if len(line.encode('utf-8'))+1 > 1024*1024:
                failures.append('example exceeds 1 MiB')
            if not set(schema['required']) <= row.keys() or row.get('type') not in events:
                failures.append('example has no valid event envelope')
            examples.append(row)
    example_events = {row['type'] for row in examples}
    example_errors = {row['payload'].get('code') for row in examples if row['type']=='error'}
    if events-example_events or errors-example_errors:
        failures.append(f'examples missing events={sorted(events-example_events)} errors={sorted(errors-example_errors)}')
    if example_errors-errors:
        failures.append(f'examples have unknown error codes={sorted(example_errors-errors)}')
    # All defined field names must appear in the public text. This also covers
    # nested source/licence/status fields and reserved instance identity fields.
    def fields(value):
        result = set()
        if isinstance(value, dict):
            result.update(value.get('properties', {}))
            for item in value.values():
                result.update(fields(item))
        elif isinstance(value, list):
            for item in value:
                result.update(fields(item))
        return result
    undocumented = {name for name in fields(schema) if f'`{name}`' not in document}
    if undocumented:
        failures.append(f'undocumented fields: {sorted(undocumented)}')
    if failures:
        raise ValueError('\n'.join(failures))
    return len(events), len(errors), len(examples)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', type=Path, default=ROOT/'docs/CLI-CONTRACT.md')
    parser.add_argument('--schema', type=Path, default=Path(__file__).with_name('event.schema.json'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    document = args.docs.read_text(encoding='utf-8')
    schema = json.loads(args.schema.read_text(encoding='utf-8'))
    events, errors, examples = check(document, schema)
    print(f'Contract docs: {events} event types, {errors} error codes, {examples} example lines; 0 missing, 0 extra')
    if args.self_test:
        import copy
        passed = 0
        for kind in ('missing_event', 'extra_event', 'missing_error', 'extra_error'):
            changed = copy.deepcopy(schema)
            if kind.endswith('event'):
                enum = changed['properties']['type']['enum']
            else:
                enum = changed['$defs']['error']['properties']['code']['enum']
            if kind.startswith('missing'):
                enum.append('planted_missing')
            else:
                enum.pop()
            try:
                check(document, changed)
            except ValueError:
                passed += 1
            else:
                raise AssertionError(f'drift was accepted: {kind}')
        print(f'Docs drift fault injection: {passed} passed, 0 failed')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, IndexError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
