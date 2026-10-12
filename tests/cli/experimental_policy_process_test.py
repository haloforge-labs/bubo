"""Whole parser/resolver/creation boundary; CPU-only expected exit-83 factory."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

host, mode = sys.argv[1:3]
flags = 0x08004000 if os.name == 'nt' else 0
with tempfile.TemporaryDirectory(prefix='p4-policy-') as directory:
    root = Path(directory)
    header = root / 'metadata-only.gguf'
    def string(value):
        b = value.encode()
        return struct.pack('<Q', len(b)) + b
    header.write_bytes(struct.pack('<IIQQ', 0x46554747, 3, 0, 2) +
                       string('general.architecture') + struct.pack('<I', 8) + string('qwen4exp') +
                       string('qwen4exp.context_length') + struct.pack('<II', 4, 262144) + bytes(32))
    count = 0
    def case(profile, source, request=None, entry='cli', manifest_on=True,
             expected=None, extra=(), path='0'):
        global count
        count += 1
        counter = root / f'counter-{count}.txt'
        manifest = root / f'manifest-{count}.json'
        data = {'--npu-prefill': 'on', '--fabric-peak': '1'} if manifest_on else {}
        if source in ('manifest', 'both'):
            data['GUFO_NPU_DIR'] = path
        manifest.write_text(json.dumps(data))
        env = {k: v for k, v in os.environ.items() if not k.startswith('GUFO_')}
        if source in ('env', 'both'):
            env['GUFO_NPU_DIR'] = path
        argv = [host, entry, str(counter), str(manifest), '--profile', profile,
                '--model', str(header), '--speculative', 'off', *extra]
        if request is not None:
            argv += ['--npu-prefill', request]
        r = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=15, creationflags=flags)
        records = counter.read_text().splitlines() if counter.exists() else []
        rejected = request == 'on' and (mode == 'OFF' or profile in ('exact', 'stock'))
        positive = request == 'on' and not rejected and entry != 'noncli'
        want = expected if expected is not None else 2 if rejected else 83 if positive else 0
        assert r.returncode == want, (argv, r.returncode, want, r.stdout)
        assert records == (['factory=1'] if want == 83 else []), (argv, records)
        if want == 2:
            assert 'phase=release-deny' in r.stdout and 'effective=off' in r.stdout, r.stdout
        elif want == 83:
            assert 'phase=factory counter=1' in r.stdout, r.stdout
        else:
            assert 'phase=model-create effective=off' in r.stdout, r.stdout
            assert 'phase=model-ready factory=0' in r.stdout, r.stdout
        print(f'PASS: {mode} {profile} DIR={source} CLI={request} entry={entry} exit={want} factory={len(records)}', flush=True)
        return r.stdout
    for source in ('env', 'manifest', 'both'):
        for profile in ('exact', 'stock', 'default', 'subagents'):
            case(profile, source)
            case(profile, source, 'off')
            case(profile, source, 'on')
            case(profile, source, entry='noncli')
    if mode == 'ON':
        case('default', 'both', 'on', entry='noncli')
    case('default', 'both', entry='fabric-as-prefill', extra=('--fabric-peak',))
    print('PASS: fabric authorization cannot authorize a prefill factory')
    for value in ('', 'ON', '0', '1', 'onx', ' on', 'off '):
        # Malformed values must fail at the resolver, without a factory call.
        counter = root / 'malformed-counter.txt'
        manifest = root / 'empty.json'
        manifest.write_text('{}')
        r = subprocess.run([host, 'cli', str(counter), str(manifest), '--npu-prefill='+value],
                           text=True, capture_output=True, timeout=15, creationflags=flags)
        assert r.returncode == 2 and 'invalid value' in r.stdout and not counter.exists(), r.stdout
    print('PASS: strict off|on parsing rejects seven malformed variants before Create')
    for profile in ('exact', 'stock'):
        counter = root / 'fabric-counter.txt'
        r = subprocess.run([host, 'cli', str(counter), str(manifest), '--profile', profile, '--fabric-peak'],
                           text=True, capture_output=True, timeout=15, creationflags=flags)
        assert r.returncode == 2 and '--fabric-peak effective=off' in r.stdout and not counter.exists(), r.stdout
    print('PASS: fabric-peak exact/stock rejection; manifest never enables it')
    for profile in ('exact', 'stock', 'default'):
        if profile == 'default' and mode == 'ON':
            continue
        r = subprocess.run([host, 'cli', str(counter), str(manifest), '--profile', profile,
                            '--model', str(root/'missing.gguf'), '--npu-prefill', 'on'],
                           text=True, capture_output=True, timeout=15, creationflags=flags)
        assert r.returncode == 2 and 'effective=off' in r.stdout and 'model header' not in r.stdout, r.stdout
    print('PASS: forbidden ON is rejected before model-header inspection')
    for modality in ('video', 'image', 'tts', 'asr'):
        for option in (['--npu-prefill','on'], ['--fabric-peak']):
            r = subprocess.run([host, 'cli', str(counter), str(manifest), modality,
                                '--profile', 'exact', *option], text=True, capture_output=True,
                               timeout=15, creationflags=flags)
            assert r.returncode == 2 and 'requires serve llm' in r.stdout and not counter.exists(), r.stdout
    print('PASS: non-LLM experimental ON flags rejected before any loader')
    # Failing-variant proof: the same negative exact case hits the flushed
    # factory and reserved abort when either original defect is planted.
    for variant in ('variant-dir-enabler', 'variant-unguarded'):
        case('exact', 'env', entry=variant, expected=83)
        print(f'PASS: failing variant {variant} violates negative-case oracle (factory=1, exit=83)')
print('PASS: process deny-before-Create matrix; GPU/XRT/model/server calls=0')
