"""CPU-only startup resolver contract; no engine, GPU, server or model weights."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolver', type=Path, required=True,
                        help='release_profile_test executable (CPU echo harness)')
    args = parser.parse_args()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('GUFO_', 'BUBO_'))}
    with tempfile.TemporaryDirectory(prefix='bubo-refusal-default-') as temporary:
        root = Path(temporary)
        env['LOCALAPPDATA'] = str(root / 'local')
        model = root / 'model.gguf'
        def text(value):
            value = value.encode()
            return struct.pack('<Q', len(value)) + value
        header = (b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
                  + text('general.architecture') + struct.pack('<I', 8)
                  + text('qwen4exp'))
        model.write_bytes(header + bytes(-len(header) % 32))
        default = root / 'refusal' / 'refusal-r-d180ec15-fitted.gguf'
        # Without a direction beside the model, serve uses the one installed
        # beside the executable; the error names that place.
        installed = (args.resolver.resolve().parent / 'refusal'
                     / 'refusal-r-d180ec15-fitted.gguf')
        base = [str(args.resolver.resolve()), '--print-effective-settings',
                'llm', '--profile', 'vanilla', '--model', str(model)]
        def run(*flags, environment=None):
            return subprocess.run(base + list(flags), env=environment or env,
                                  cwd=root, capture_output=True, text=True,
                                  timeout=30)
        def flags(result):
            assert result.returncode == 0, result.stderr
            return {row['name']: row['value'] for row in
                    json.loads(result.stdout.splitlines()[-1])['settings']}
        assert flags(run())['--refusal-direction'] == 'off'
        missing = run('--refusal-direction', 'on')
        assert missing.returncode != 0
        assert 'Refusal direction: packaged default missing:' in missing.stderr
        assert (str(installed).replace('\\', '/')
                in missing.stderr.replace('\\', '/')), missing.stderr
        print('PASS missing default:', missing.stderr.strip())
        assert flags(run('--refusal-direction', 'off'))['--refusal-direction'] == 'off'
        explicit = flags(run('--refusal-direction', 'on',
                             '--refusal-direction-file', 'developer.gguf'))
        assert explicit['--refusal-direction-file'] == 'developer.gguf'
        default.parent.mkdir()
        # Startup resolves the path; the independent strict-loader CPU test
        # validates GGUF contents and fitted strength before engine execution.
        default.write_bytes(b'path-resolution fixture')
        enabled = flags(run('--refusal-direction', 'on'))
        assert Path(enabled['--refusal-direction-file']) == default
        assert enabled['--refusal-direction-mode'] == 'sites'
        assert (enabled['--refusal-direction-strength']
                == 'Fitted value in file (about 1.96)')
        print('PASS ON resolves packaged default:', enabled['--refusal-direction-file'])
        print('PASS sites mode; strength omitted for strict loader to read from file')
        override = flags(run('--refusal-direction', 'on',
                             '--refusal-direction-file', 'developer.gguf',
                             '--refusal-direction-mode', 'stream',
                             '--refusal-direction-strength', '1'))
        assert override['--refusal-direction-file'] == 'developer.gguf'
        assert override['--refusal-direction-mode'] == 'stream'
        assert override['--refusal-direction-strength'] == '1'
        legacy_env = dict(env, GUFO_REFUSAL_DIRECTION='legacy.gguf')
        legacy = flags(run('--refusal-direction', 'on', environment=legacy_env))
        assert (legacy['--refusal-direction-file']
                == 'Packaged refusal/refusal-r-d180ec15-fitted.gguf')
        print('PASS explicit controls and legacy file precedence retained; OFF needs no file')

        # Every activation route meets the same rule: a 256-expert model is
        # refused before load whether the switch, the environment or a legacy
        # file turns the direction on.
        coder = root / 'coder.gguf'
        coder_header = (b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
                        + text('general.architecture') + struct.pack('<I', 8)
                        + text('qwen4exp')
                        + text('qwen4exp.expert_count') + struct.pack('<II', 4, 256))
        coder.write_bytes(coder_header + bytes(-len(coder_header) % 32))
        coder_base = [str(args.resolver.resolve()), '--print-effective-settings',
                      'llm', '--profile', 'vanilla', '--model', str(coder)]
        def run_coder(*flags, environment):
            return subprocess.run(coder_base + list(flags), env=environment,
                                  cwd=root, capture_output=True, text=True,
                                  timeout=30)
        for route, extra_env, flags_used in (
                ('switch', {}, ('--refusal-direction', 'on')),
                ('environment', {'GUFO_REFUSAL_DIRECTION_ENABLED': 'on'}, ()),
                ('legacy file', {'GUFO_REFUSAL_DIRECTION': str(default)}, ())):
            refused = run_coder(*flags_used, environment=dict(env, **extra_env))
            assert refused.returncode != 0, (route, refused.stdout)
            assert 'not tested for this model' in refused.stderr, (route, refused.stderr)
        off = run_coder('--refusal-direction', 'off',
                        environment=dict(env, GUFO_REFUSAL_DIRECTION=str(default)))
        assert off.returncode == 0, off.stderr
        print('PASS a 256-expert model is refused on every activation route; off wins over a legacy file')
        # A file flag only chooses the vector; alone it leaves the direction
        # off, so an unsupported model starts. With the switch on it is refused.
        file_only = run_coder('--refusal-direction-file', str(default), environment=env)
        assert flags(file_only)['--refusal-direction'] == 'off'
        refused = run_coder('--refusal-direction', 'on', '--refusal-direction-file',
                            str(default), environment=env)
        assert refused.returncode != 0 and 'not tested for this model' in refused.stderr, refused.stderr
        print('PASS a file flag alone stays off on a 256-expert model; with the switch on it is refused')


if __name__ == '__main__':
    main()
