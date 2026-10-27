"""Regenerate model contract examples from a built CPU models_test executable."""

import argparse
import json
from pathlib import Path, PureWindowsPath
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models_test', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    valid = root / 'tests/cli/contract/valid'
    with tempfile.TemporaryDirectory(prefix='bubo-model-contract-') as scratch:
        subprocess.run([str(args.models_test.resolve()), '--write-fixtures', scratch], check=True)
        generated = {p.name: json.loads(p.read_text(encoding='utf-8'))
                     for p in Path(scratch).glob('*.json')}
    new_base = str(Path(generated['model-estimate.json']['payload']['subject']).parent.parent)
    old_subject = json.loads((valid / 'model-estimate.json').read_text(
        encoding='utf-8'))['payload']['subject']
    old_base = str(PureWindowsPath(old_subject).parent.parent)

    def normalize(value):
        if isinstance(value, str):
            if value.startswith(new_base):
                return old_base + value[len(new_base):].replace('/', '\\')
            return value.replace(new_base, old_base)
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        return value

    # Preserve the recorded envelopes; regenerate every payload and normalize
    # only the temporary fixture root. No memory or model facts live here.
    for name, event in generated.items():
        target = valid / name
        old = json.loads(target.read_text(encoding='utf-8'))
        old['payload'] = normalize(event['payload'])
        generated[name] = old
        target.write_text(json.dumps(old, ensure_ascii=False, separators=(',', ':')) + '\n',
                          encoding='utf-8')
    document = root / 'docs/CLI-CONTRACT.md'
    lines = document.read_text(encoding='utf-8').splitlines()
    count = 0
    for i, line in enumerate(lines):
        if not line.startswith('{"schema_version":'):
            continue
        event = json.loads(line)
        if event['type'] == 'models':
            name = 'models.json'
        elif event['type'] == 'model_estimate':
            name = ('model-estimate.json' if event['payload']['subject_kind'] == 'path'
                    else 'model-estimate-none.json')
        else:
            continue
        lines[i] = json.dumps(generated[name], ensure_ascii=False, separators=(',', ':'))
        count += 1
    document.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Model contract regeneration: {len(generated)} fixtures, {count} documentation examples')


if __name__ == '__main__':
    main()
