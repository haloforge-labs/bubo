#!/usr/bin/env python3
"""Offline regression proofs for the prepare source/archive security scan."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'check_brand', ROOT / 'tools/release/check_brand.py')
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def main():
    with tempfile.TemporaryDirectory(prefix='prepare-ci-', dir=Path.cwd()) as folder:
        root = Path(folder)
        source = root / 'src/core/prepare/fixture.cpp'
        source.parent.mkdir(parents=True)
        def check(text):
            source.write_text(text, encoding='utf-8')
            return scan.prepare_security(root)
        assert not check('#include <zlib.h>\nvoid f() { inflate(&z, 0); }\n')
        for forbidden in ('WinHttpOpen()', 'BCryptCreateHash()',
                          'tinfl_decompress()', 'HuffmanTree tree;',
                          'SHA256_Transform()', 'int inflate(int x) { return x; }'):
            assert check(forbidden), forbidden
        for hook in ('RunModelForTesting()', 'RunRuntimeForTesting()',
                     'RuntimeReadyErrorForTesting()', 'LoadModelPackageForTesting()',
                     'test_loopback = true;', 'origin_override = "http://local";'):
            assert check(hook), hook
            assert not check('#ifdef BUBO_PREPARE_TESTING\n' + hook + '\n#endif\n')
            assert check('#ifdef BUBO_PREPARE_TESTING\nint test;\n#else\n' + hook + '\n#endif\n')
            assert check('#ifndef BUBO_PREPARE_TESTING\n' + hook + '\n#endif\n')
            assert check('#if UNRELATED\n' + hook + '\n#endif\n')
        source.write_text('int production;\n', encoding='utf-8')
        library = root / 'production.lib'
        library.write_bytes(b'linked EVP and zlib symbols')
        assert not scan.prepare_security(root, library)
        for hook in scan.BINARY_HOOKS:
            library.write_bytes(b'archive\0' + hook + b'\0')
            assert scan.prepare_security(root, library)
    print('PASS: prepare CI rejects WinHTTP/BCrypt/own inflater/hasher; '
          'unguarded fixture overrides and planted production archive hooks; '
          'guarded test code and linked zlib accepted')
    if sys.platform == 'win32':
        subprocess.run(['pwsh', '-NoProfile', '-File',
                        str(ROOT / 'tests/prepare/package_policy_test.ps1')], check=True)


if __name__ == '__main__':
    main()
