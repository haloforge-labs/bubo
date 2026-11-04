#!/usr/bin/env python3
"""Bring program_name_allowlist.json in line with the tree after an upstream merge.

Run from the repository root: python tools/release/regen_program_name_allowlist.py
Then run python tools/release/check_brand.py: it must print PASS.

It changes only what check_brand.py reports, and nothing is renamed:
  * a fixture (whole-file sha256) entry gets its hash refreshed;
  * a line entry whose line is gone is dropped, and a changed count is corrected;
  * a documentation JSON file with unlisted lines becomes one historic fixture entry;
  * every other unlisted line gets one entry with the category and reason below.
Review the diff: a reason is the only record of why a legacy name stays.
"""
import hashlib
import json
from pathlib import Path

import check_brand as brand

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / 'tools/release/program_name_allowlist.json'
MERGE_NOTE = 'Post-0.1.1 upstream merge'

# Shipped wire and cache contracts in src/cli (exact line text -> category, reason).
SHIPPED = {
    '"X-Gufo-Schema"': ('api-contract', 'HTTP response header of the /v1/embeddings API; the header name is a wire contract kept for client compatibility.'),
    '"X-Gufo-Embedding-Backend"': ('api-contract', 'HTTP response header of the /v1/embeddings API; the header name is a wire contract kept for client compatibility.'),
    '"gufo.embeddings-api.v1"': ('schema', 'Schema identifier returned by the /v1/embeddings API; a wire contract kept for client compatibility.'),
    'schema=gufo-process-': ('cache-config', 'Cache identity text hashed into cache keys; renaming it would invalidate every existing cache.'),
}


def classify(path, line):
    if path.startswith('src/cli/'):
        for needle, (category, reason) in SHIPPED.items():
            if needle in line:
                return category, reason
        raise SystemExit(f'{path}: unlisted legacy name in shipped source, decide by hand: {line.strip()}')
    if path.startswith('docs/'):
        return 'historic-fixture', f'{MERGE_NOTE}: documentation or measurement record, does not ship.'
    return 'developer-tool', f'{MERGE_NOTE}: build, Nix or CMake text, does not ship.'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    raw = ALLOWLIST.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    entries = json.loads(raw.decode('utf-8'))
    found, _ = brand.sites(ROOT)
    fixtures = {e['file'] for e in entries if 'sha256' in e}
    stats = {'fixture_refreshed': 0, 'dropped': 0, 'recounted': 0, 'fixture_added': 0, 'added': 0}

    kept = []
    for entry in entries:
        if 'sha256' in entry:
            if not (ROOT / entry['file']).is_file():
                stats['dropped'] += 1
                continue
            digest = sha(entry['file'])
            if digest != entry['sha256']:
                entry['sha256'] = digest
                stats['fixture_refreshed'] += 1
            kept.append(entry)
            continue
        key = (entry['file'], entry['line'].encode('utf-8').hex())
        count = found.get(key, 0)
        if count == 0:
            stats['dropped'] += 1
            continue
        if count != entry['count']:
            entry['count'] = count
            stats['recounted'] += 1
        kept.append(entry)

    listed = {(e['file'], e['line'].encode('utf-8').hex()) for e in kept if 'line' in e}
    new_fixture_files = set()
    new_lines = []
    for (path, hexline), count in sorted(found.items()):
        if path in fixtures or (path, hexline) in listed:
            continue
        if path.startswith('docs/') and path.endswith('.json'):
            new_fixture_files.add(path)
            continue
        line = bytes.fromhex(hexline).decode('utf-8')
        category, reason = classify(path, line)
        new_lines.append({'file': path, 'line': line, 'count': count, 'category': category, 'reason': reason})
        stats['added'] += 1
    if new_fixture_files:
        # A JSON record becomes one fixture, so its older line entries are redundant.
        kept = [e for e in kept if e['file'] not in new_fixture_files]
    for path in sorted(new_fixture_files):
        kept.append({'file': path, 'sha256': sha(path),
                     'reason': f'{MERGE_NOTE}: measurement or review record, does not ship.',
                     'category': 'historic-fixture'})
        stats['fixture_added'] += 1
    kept.extend(new_lines)

    text = json.dumps(kept, indent=2, ensure_ascii=False) + '\n'
    ALLOWLIST.write_bytes(text.replace('\n', newline).encode('utf-8'))
    print(stats, f'entries={len(kept)}')


if __name__ == '__main__':
    main()
