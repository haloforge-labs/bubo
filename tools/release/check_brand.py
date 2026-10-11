#!/usr/bin/env python3
"""Reject unreviewed legacy program names; retain explicit compatibility sites."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import re
import subprocess

LEGACY = re.compile(r'\bgufo\b', re.IGNORECASE)
LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
SOURCE = {'.cpp', '.hpp', '.h', '.cc', '.in', '.inc', '.md'}
DOCS = {'.md', '.txt', '.json', '.csv', '.tsv'}
CATEGORIES = {'api', 'api-contract', 'schema', 'resource-path', 'cache-config',
              'engine', 'build-target', 'developer-tool', 'upstream-url',
              'historic-fixture', 'logging', 'compatibility'}


def scanned_paths(root):
    for directory, extensions in [('src/cli', SOURCE), ('docs', DOCS)]:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix in extensions:
                yield path, directory == 'docs' or path.suffix == '.md'
    for name in ['README.md', 'start.cmd', 'flake.nix', 'CMakeLists.txt']:
        path = root / name
        if path.is_file():
            yield path, True
    # Scan complete script/build lines, including unquoted commands and comments.
    for path in sorted((root / 'tools/windows').glob('*.ps1')):
        yield path, True
    for directory in ['.devops', 'cmake']:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file():
                yield path, True


def is_api_key(entry):
    if not entry['file'].startswith('src/cli/'):
        return False
    line = entry.get('line', '')
    return bool(re.search(r'\["gufo"\]|\.find\("gufo"\)|kApiKeys\[\]|'
                          r'field_name == "gufo"|"seconds", "gufo", "bubo"', line))


def sites(root):
    found = Counter()
    locations = {}
    for path, complete in scanned_paths(root):
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            text = line.decode('utf-8', errors='surrogateescape')
            visible = text if complete else ' '.join(LITERAL.findall(text))
            if LEGACY.search(visible):
                key = (path.relative_to(root).as_posix(), line.hex())
                found[key] += 1
                locations[key] = number
    return found, locations


def check(root):
    entries = json.loads((root / 'tools/release/program_name_allowlist.json').read_text(encoding='utf-8'))
    allowed = Counter()
    fixtures = set()
    errors = []
    for entry in entries:
        if not isinstance(entry.get('category'), str) or entry['category'] not in CATEGORIES:
            errors.append(entry['file'] + ': invalid allow-list category')
        if not isinstance(entry.get('reason'), str) or not entry['reason'].strip():
            errors.append(entry['file'] + ': allow-list entries require a reason')
        if entry.get('category') == 'api' and not is_api_key(entry):
            errors.append(entry['file'] + ': api category must parse/emit an API key')
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
    parser.add_argument('--package-folder', type=Path,
                        help='Also scan every shipped Windows EXE/DLL import')
    args = parser.parse_args()
    errors = check(args.root)
    for error in errors:
        print('FAIL: ' + error)
    if not errors:
        print('PASS: Bubo program-name literal scan (src/cli, docs, README, tools/windows/*.ps1, start.cmd, flake.nix, .devops, CMakeLists.txt, cmake; categorized exact-line allow-list)')
    if args.package_folder:
        result = subprocess.run(['pwsh', '-NoProfile', '-File',
                                 str(args.root / 'tools/release/check-package.ps1'),
                                 '-PackageFolder', str(args.package_folder)], check=False)
        return bool(errors) or result.returncode != 0
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
