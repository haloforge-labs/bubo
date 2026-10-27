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

# This is separate from the exact-line brand allow-list: security findings
# cannot be waived by adding a historic name exception.
SECOND_STACK = re.compile(
    r'\b(?:WinHttp\w*|WINHTTP\w*|BCrypt\w*|BCRYPT\w*|'
    r'tinfl\w*|puff|stbi_zlib\w*|Huffman\w*|InflateBits|'
    r'SHA256_Transform|sha256_compress)\b', re.IGNORECASE)
FIXTURE_OVERRIDE = re.compile(
    r'\b(?:(?:RunModel|RunRuntime|RunAll|\w*(?:Manifest|Catalogue|Package))ForTesting|'
    r'(?:Model|Runtime|Package|Manifest)TestEnvironment|test_loopback|origin_override|synthetic_disclosure|'
    r'PIN_OVERRIDE|CATALOGUE_OVERRIDE|BUBO_FIXTURE\w*)\b')
BINARY_HOOKS = (b'RunModelForTesting', b'RunRuntimeForTesting',
                b'RunAllForTesting', b'origin_override', b'test_loopback',
                b'RuntimeReadyErrorForTesting', b'LoadModelPackageForTesting',
                b'synthetic_disclosure', b'http,https')


def production_lines(text):
    """Keep every non-test branch; remove only explicitly guarded test code."""
    stack = []
    for number, line in enumerate(text.splitlines(), 1):
        directive = re.match(r'\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)', line)
        if directive:
            kind, condition = directive.groups()
            if kind in ('ifdef', 'ifndef', 'if'):
                test = condition.strip() == 'BUBO_PREPARE_TESTING'
                test = test or condition.strip() == 'defined(BUBO_PREPARE_TESTING)'
                stack.append(False if test and kind != 'ifndef' else
                             True if test else None)
            elif kind in ('else', 'elif') and stack:
                if stack[-1] is not None:
                    stack[-1] = not stack[-1] if kind == 'else' else True
            elif kind == 'endif' and stack:
                stack.pop()
            continue
        if False not in stack:
            yield number, line


def prepare_security(root, prepare_library=None):
    errors = []
    for path in sorted((root / 'src').rglob('*')):
        if not path.is_file() or path.suffix not in SOURCE | {'.hip', '.c'}:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8', errors='surrogateescape')
        for number, line in enumerate(text.splitlines(), 1):
            if SECOND_STACK.search(line):
                errors.append(f'{relative}:{number}: forbidden second HTTP/hash/inflate stack: {line.strip()}')
            # An inflater implementation cannot define the linked zlib API.
            if re.search(r'\b(?:inflate|inflateInit2_?|inflateEnd)\s*\([^;]*\)\s*\{', line):
                errors.append(f'{relative}:{number}: own inflater implementation')
        for number, line in production_lines(text):
            package_source = '/prepare/' in relative or path.name.startswith('model_manifest.')
            package_hook = package_source and re.search(r'\b\w*ForTesting\b', line)
            if FIXTURE_OVERRIDE.search(line) or package_hook:
                errors.append(f'{relative}:{number}: fixture override in production code: {line.strip()}')
    if prepare_library:
        data = prepare_library.read_bytes()
        for hook in BINARY_HOOKS:
            if hook in data:
                errors.append(f'{prepare_library}: release archive contains test hook {hook.decode()}')
    return errors


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


def gui_urls(root):
    """Production GUI links cannot use the old organization, even if allow-listed."""
    errors = []
    legacy_url = re.compile(r'https?://[^\s\x27\x22<>]*gufo-org', re.IGNORECASE)
    for path in sorted((root / 'gui/wails/frontend/src').rglob('*')):
        if not path.is_file() or path.suffix not in {'.ts', '.tsx', '.css'}:
            continue
        if '.test.' in path.name or '.shots.' in path.name:
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if legacy_url.search(line):
                errors.append(f'{path.relative_to(root).as_posix()}:{number}: legacy GUI organization URL')
    return errors


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
    parser.add_argument('--prepare-library', type=Path,
                        help='Reject fixture hook symbols in the production prepare archive')
    parser.add_argument('--release-gate', type=Path,
                        help='Also run the offline production pin/evidence gate executable')
    args = parser.parse_args()
    errors = check(args.root) + gui_urls(args.root) + prepare_security(args.root, args.prepare_library)
    if args.release_gate:
        result = subprocess.run([str(args.release_gate), '--release',
                                 str(args.root / 'tools/release/prepare-evidence')], check=False)
        if result.returncode:
            errors.append('prepare production PIN SLOT/source evidence gate failed')
    for error in errors:
        print('FAIL: ' + error)
    if not errors:
        print('PASS: Bubo program-name literal scan (src/cli, docs, README, tools/windows/*.ps1, start.cmd, flake.nix, .devops, CMakeLists.txt, cmake; categorized exact-line allow-list)')
        print('PASS: production prepare source/archive fixture guard and forbidden second HTTP/hash/inflater stack scan')
        print('PASS: production GUI links reject legacy organization URLs')
    if args.package_folder:
        result = subprocess.run(['pwsh', '-NoProfile', '-File',
                                 str(args.root / 'tools/release/check-package.ps1'),
                                 '-PackageFolder', str(args.package_folder)], check=False)
        return bool(errors) or result.returncode != 0
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
