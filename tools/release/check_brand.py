#!/usr/bin/env python3
"""Reject unreviewed legacy program names; retain explicit compatibility sites."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import re

LEGACY = re.compile(r'\bgufo\b', re.IGNORECASE)
LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
SOURCE = {'.cpp', '.hpp', '.h', '.cc', '.in', '.inc', '.md'}
DOCS = {'.md', '.txt', '.json', '.csv', '.tsv'}


def sites(root):
    found = Counter()
    locations = {}
    for directory, extensions in [('src/cli', SOURCE), ('docs', DOCS)]:
        for path in sorted((root / directory).rglob('*')):
            if not path.is_file() or path.suffix not in extensions:
                continue
            for number, line in enumerate(path.read_bytes().splitlines(), 1):
                text = line.decode('utf-8', errors='surrogateescape')
                visible = text if directory == 'docs' or path.suffix == '.md' else ' '.join(LITERAL.findall(text))
                if LEGACY.search(visible):
                    key = (path.relative_to(root).as_posix(), line.hex())
                    found[key] += 1
                    locations[key] = number
    # README is the public entry point, so include it in the same policy.
    path = root / 'README.md'
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if LEGACY.search(line.decode('utf-8', errors='surrogateescape')):
            key = ('README.md', line.hex())
            found[key] += 1
            locations[key] = number
    return found, locations


def check(root):
    entries = json.loads((root / 'tools/release/program_name_allowlist.json').read_text(encoding='utf-8'))
    allowed = Counter()
    fixtures = set()
    errors = []
    for entry in entries:
        if not entry.get('reason', '').strip():
            raise ValueError('allow-list entries require a reason')
        if 'sha256' in entry:
            path = root / entry['file']
            digest = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            if digest != entry['sha256']:
                errors.append(entry['file'] + ': historic fixture changed; review legacy names')
            fixtures.add(entry['file'])
            continue
        if entry['count'] < 1:
            raise ValueError('allow-list entries require a positive count')
        key = (entry['file'], entry['line'].encode('utf-8').hex())
        if key in allowed:
            raise ValueError('duplicate allow-list entry: ' + entry['file'])
        allowed[key] = entry['count']
    found, locations = sites(root)
    found = Counter({key: count for key, count in found.items() if key[0] not in fixtures})
    for key in sorted(found.keys() | allowed.keys()):
        if found[key] != allowed[key]:
            errors.append(f'{key[0]}:{locations.get(key, 0)}: legacy program literal count={found[key]}, allowed={allowed[key]}: '
                          + bytes.fromhex(key[1]).decode('utf-8', errors='replace'))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = check(args.root)
    for error in errors:
        print('FAIL: ' + error)
    if not errors:
        print('PASS: Bubo program-name literal scan (src/cli, docs, README; reasoned exact-line allow-list)')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
