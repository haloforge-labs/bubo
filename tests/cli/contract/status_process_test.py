import json
from pathlib import Path
import shutil
import subprocess
import sys

executable = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
import os
env = dict(os.environ, LOCALAPPDATA=str(root / 'empty-local-é-中' /
                                      ('long-'+'x'*220) / ('long-'+'x'*220)))
env.pop('BUBO_MANIFEST', None)
passed = 0
for arguments, expected, kind in [(['status','--json'],0,'status'),
                                  (['status','--jsonl'],0,'status'),
                                  (['status','--json','--watch'],0,'status'),
                                  (['status','--json','--bad'],2,'error'),
                                  (['--bad','--json'],2,'error'),
                                  (['--help','--json'],0,'capabilities'),
                                  (['--version','--jsonl'],0,'capabilities'),
                                  (['help','prepare','--json'],2,'error'),
                                  (['status','--json','--jsonl'],2,'error'),
                                  (['prepare','--jsonl','--help'],0,'capabilities'),
                                  (['prepare','bad-step','--jsonl'],2,'error'),
                                  (['prepare','--output','--jsonl'],2,'error'),
                                  (['prepare','model','--runtime-dir','unused','--jsonl'],2,'error'),
                                  (['prepare','runtime','--mtp-model','unused','--json'],2,'error'),
                                  (['prepare','--json','--jsonl'],2,'error'),
                                  (['serve','--jsonl'],11,'stopped'),
                                  (['stop','--json'],2,'error'),
                                  (['unknown-command','--json'],2,'error'),
                                  (['diagnose','--json'],2,'error')]:
    result = subprocess.run([str(executable), *arguments],cwd=root,env=env,
                            input=b'',capture_output=True,timeout=10,
                            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    assert result.returncode == expected, (arguments,result.returncode,result.stderr)
    assert not result.stdout.startswith(b'\xef\xbb\xbf') and b'\r' not in result.stdout
    rows = [json.loads(line) for line in result.stdout.decode('utf-8').splitlines()]
    assert rows[-1]['type'] == kind and rows[-1]['schema_version'] == 1
    if arguments == ['serve','--jsonl']:
        assert [row['type'] for row in rows] == ['starting','error','stopping','stopped']
        assert rows[1]['payload']['code'] == 'runtime_missing'
    if kind == 'status':
        payload = rows[-1]['payload']
        assert payload['phase'] == 'not_prepared' and not payload['prepared']
        assert payload['instances'] == []
        assert payload.get('endpoints') is None
        assert payload['memory_estimate'] is None
        assert payload['active_model'] is None
        assert payload['client_snippets'] is None
        assert payload['api_key_env'] == 'BUBO_API_KEY'
        assert payload['paths']['local_app_data'] == env['LOCALAPPDATA']
        assert not payload['readiness']['runtime'] and not payload['readiness']['model']
        if arguments[-1]=='--json':
            assert len(rows)==1
            (root/'status.stdout.jsonl').write_bytes(result.stdout)
    passed += 1

def run(arguments, environment):
    result = subprocess.run([str(executable), *arguments],cwd=root,env=environment,
                            input=b'',capture_output=True,timeout=10,
                            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    return result, [json.loads(line) for line in result.stdout.decode('utf-8').splitlines()]

# Real production plans, no downloads or model loads. Both producers must
# supply their own remaining-space facts in the split volume records.
for component in ('model', 'runtime'):
    destination = root / ('plan-' + component)
    result, rows = run(['prepare', component, '--jsonl',
                        *(['--plan'] if component == 'model' else []),
                        '--runtime-dir' if component == 'runtime' else '--output',
                        str(destination)], env)
    # Runtime has no --plan switch: EOF declines the emitted preview before
    # approval, so it must cancel with 3 without downloading or writing.
    assert result.returncode == (3 if component == 'runtime' else 0), (component, result.returncode, result.stderr)
    volumes = [volume for row in rows if row['type'] == 'plan'
               for volume in row['payload']['volumes']]
    assert volumes, component
    for volume in volumes:
        assert volume['safety_margin_bytes'] == 2147483648
        assert volume['bytes_to_write'] > 0
        assert volume['required_bytes'] == volume['bytes_to_write'] + 2147483648
        assert volume['free_after_download_bytes'] == max(0, volume['free_bytes'] - volume['bytes_to_write'])
    assert not destination.exists()
    passed += 1

# Shared AMD runtime folder: %LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151.
local = root / 'shared-runtime-local'
shutil.rmtree(local, ignore_errors=True)
shared = local / 'Bubo' / 'runtime' / 'rocm-10.0.0-gfx1151'
env2 = dict(env, LOCALAPPDATA=str(local))
env2.pop('BUBO_RUNTIME_DIR', None)
result, rows = run(['status', '--json'], env2)
paths = rows[-1]['payload']['paths']
assert Path(paths['runtime']) == shared and paths['runtime_kind'] == 'shared', paths
assert Path(paths['runtime_shared']) == shared, paths
# A damaged shared runtime (junk marker) stops serve with the typed error.
(shared / '.bubo').mkdir(parents=True)
(shared / '.bubo' / 'runtime-ready.json').write_text('{"junk":1}')
result, rows = run(['serve', '--jsonl'], env2)
assert result.returncode == 11 and rows[1]['payload']['code'] == 'runtime_missing', (result.returncode, rows)
def run_text(arguments, environment):
    result = subprocess.run([str(executable), *arguments],cwd=root,env=environment,
                            input=b'',capture_output=True,timeout=10,
                            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    return result, result.stdout.decode('utf-8').replace('\r\n', '\n')
# Help is not held back by a missing or damaged runtime: with no engine here,
# the built-in text prints and the exit code is 0.
result, text = run_text(['serve', 'llm', '--help'], env2)
assert result.returncode == 0 and text.startswith('Usage: bubo serve llm') and not result.stderr, (result.returncode, result.stderr)
passed += 1
# Global help: what the product is, where to start, no stale family list.
result, text = run_text(['--help'], env2)
assert result.returncode == 0, result.returncode
assert 'Local LLM server for AMD Strix Halo (gfx1151).' in text, text
assert 'Start here: bubo prepare, then bubo serve.' in text, text
assert 'Fast LLM' not in text and 'Model families' not in text, text
passed += 1
result, rows = run(['status', '--json'], env2)
assert not rows[-1]['payload']['readiness']['runtime'], rows[-1]
# BUBO_RUNTIME_DIR names another folder: status reports it as explicit.
explicit = root / 'explicit-runtime'
explicit.mkdir(exist_ok=True)
env3 = dict(env2, BUBO_RUNTIME_DIR=str(explicit))
result, rows = run(['status', '--json'], env3)
paths = rows[-1]['payload']['paths']
assert Path(paths['runtime']) == explicit and paths['runtime_kind'] == 'explicit', paths
result, rows = run(['serve', '--jsonl'], env3)
assert result.returncode == 11 and rows[1]['payload']['code'] == 'runtime_missing', (result.returncode, rows)
passed += 3
print(f'Bootstrap status without runtime/model: {passed} passed, 0 failed')
