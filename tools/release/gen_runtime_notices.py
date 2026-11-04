"""Generate src/core/prepare/runtime_notices.inc from tools/release/runtime-notices.

Usage: python -I tools/release/gen_runtime_notices.py [--check] [NOTICE_DIR [OUTPUT.inc]]
Every file under NOTICE_DIR (default tools/release/runtime-notices) is embedded with its exact bytes (hex escapes), its path
relative to NOTICE_DIR in forward slashes, and the SHA-256 of those bytes. Entries are sorted by that path (byte order).
--check writes nothing. It exits 1 when OUTPUT.inc differs from the regenerated text.
"""
import hashlib
import pathlib
import sys

HEADER = ('// Exact release notice text bytes; generated with linked OpenSSL SHA-256.\n'
          'const std::vector<RuntimeNotice> kRuntimeNotices{\n')
MAX_BYTES = 1 << 20


def literal(data: bytes) -> str:
    # The committed text has a closing "" line when the length is a multiple of 20 (and for an empty file).
    lines = []
    for i in range(0, len(data) + 1, 20):
        lines.append('"' + ''.join('\\x%02x' % b for b in data[i:i + 20]) + '"')
    return '\n'.join(lines)


def render(folder: pathlib.Path) -> str:
    files = sorted((p for p in folder.rglob('*') if p.is_file()),
                   key=lambda p: p.relative_to(folder).as_posix().encode('utf-8'))
    if not files:
        raise SystemExit('no notice files under ' + str(folder))
    out = [HEADER]
    for path in files:
        data = path.read_bytes()
        if len(data) > MAX_BYTES:
            raise SystemExit(str(path) + ': larger than ' + str(MAX_BYTES) + ' bytes')
        name = path.relative_to(folder).as_posix()
        if '"' in name or '\\' in name:
            raise SystemExit(name + ': unsupported character in the path')
        out.append('{"%s","%s",\n%s},\n' % (name, hashlib.sha256(data).hexdigest(), literal(data)))
    out.append('};\n')
    return ''.join(out)


def main() -> int:
    args = sys.argv[1:]
    check = '--check' in args
    args = [a for a in args if a != '--check']
    root = pathlib.Path(__file__).resolve().parents[2]
    folder = pathlib.Path(args[0]) if args else root / 'tools' / 'release' / 'runtime-notices'
    target = pathlib.Path(args[1]) if len(args) > 1 else root / 'src' / 'core' / 'prepare' / 'runtime_notices.inc'
    text = render(folder)
    if check:
        current = target.read_bytes().decode('utf-8').replace('\r\n', '\n') if target.exists() else ''
        if current != text:
            print('FAIL: %s differs from %s' % (target, folder))
            return 1
        print('PASS: %s equals %s' % (target, folder))
        return 0
    target.write_bytes(text.encode('utf-8'))
    print('wrote', target)
    return 0


if __name__ == '__main__':
    sys.exit(main())
