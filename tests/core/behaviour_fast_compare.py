"""Run one single-threaded process at a time; compare raw transcript bytes."""
import hashlib
from pathlib import Path
import subprocess
import sys

old, new, out = sys.argv[1:]
frozen = Path(__file__).resolve().parents[1] / 'fixtures/behaviour_fast_baseline/src/core'
# Frozen at c847e6df (B6.2) on 2026-10-11; previously a40d37f. Provenance: tests/fixtures/behaviour_fast_baseline/README.md.
expected = {
    'sampling.cpp': '232ac3b2a47ff48d7ea2ebaee3b1b478b5b6d84ee70f75f61cc632e471395816',
    'sampling.hpp': '00ebc5cd7559fd95529c84af164f07f8b869666a3e975d06e3440028c51b0e64',
    'behaviour.hpp': 'f5370a56190de6e897a5a9b52757e607017dd9bb771f81c26d6d31ea74a44b0e',
}
for name, digest in expected.items():
    if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'Frozen baseline was changed: {name}')
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
for name, exe in [('baseline', old), ('candidate', new)]:
    result = subprocess.run([exe, '--dump', str(out / (name + '.bin'))],
                            capture_output=True, text=True)
    (out / (name + '.txt')).write_text(result.stdout + result.stderr)
    print(name, result.stdout.strip(), result.stderr.strip(), flush=True)
    if name == 'baseline' and ('ORACLE c847e6df sampling.cpp sha256=' +
            '232ac3b2a47ff48d7ea2ebaee3b1b478b5b6d84ee70f75f61cc632e471395816') not in result.stdout:
        raise SystemExit('Frozen baseline source fingerprint absent or wrong')
    if result.returncode:
        raise SystemExit(result.returncode)
    if 'DUMP cases=200 steps=5760 PASS' not in result.stdout:
        raise SystemExit(f'{name}: incomplete fixture matrix')
a, b = [(out / (name + '.bin')).read_bytes() for name in ('baseline', 'candidate')]
if a != b:
    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    index = [tuple(map(int, line.split())) for line in (out / 'baseline.bin.index.tsv').read_text().splitlines()]
    marker = next((r for r in reversed(index) if r[0] <= at), None)
    raise SystemExit(f'MISMATCH at byte {at}, case/step marker {marker}, sizes {len(a)} / {len(b)}; revert this step')
print(f'EQUAL bytes={len(a)} sha256={hashlib.sha256(a).hexdigest()}')
