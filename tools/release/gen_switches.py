#!/usr/bin/env python3
"""Validate the release catalog and regenerate committed C++ and Markdown."""
import argparse
import json
import math
import re
from pathlib import Path

CLASSES = {'DEFAULT', 'OPTION', 'DEBUG', 'INTERNAL', 'REMOVED', 'OTHER_MODEL'}
FAMILIES = {'any', 'qwen3.8-flash-next', 'qwen3.8-27b', 'deepseek-v4-flash', 'minimax-h3', 'qwen-image-2.1', 'qwen3-tts', 'qwen3-asr'}
REQUIRES = {'think!=off', 'sessions==1 or GUFO_DENSE_TWIN_BATCH==1', 'speculative==mtp', 'artefact:twin', 'artefact:draft_head', 'artefact:mtp', 'artefact:mmproj'}
RULES = {'text', 'bool', 'min_prob', 'prob', 'lambda', 'transition', 'number', 'effort_map', 'lenient', 'forward', 'twin_width', 'uint', 'readers', 'tuning'}
FIELDS = {'name', 'kind', 'class', 'model_family', 'platform', 'requires', 'values', 'exact_off', 'inject_with', 'suppressed_by', 'purpose', 'default_text', 'hardware_scope', 'note', 'built_in', 'value_rule', 'aliases', 'takes_value'}
INTRO = '# Release switches\n\nGenerated from `src/cli/serve/release_switches.json`.\n\nProfile defaults apply to Qwen3.8 Flash Next.\nOther model families keep their existing defaults.\nExplicit flags and real environment values take precedence.\nEmpty environment values count as unset and produce a warning.\n\n'
PROFILES = '## Profiles\n\n| Profile | Effect |\n| --- | --- |\n| default | Use production settings for one session. Use medium reasoning effort. |\n| subagents | Use four sessions with 131072 context tokens each and a 4 GiB RAM cache. |\n| exact | Disable lenient acceptance and the dense twin. Keep behaviour policies and the draft head. |\n| stock | Keep upstream defaults. Inject no environment values or flag defaults. |\n\nSubagents verifies up to 524288 total context tokens.\nUse `--allow-unmeasured-memory` above this measured limit.\nFour concurrent deep sessions can reach the host RAM guard near 50K tokens per session.\nExact pins seed 1, one session, and forward size 4096 when unset.\nA real environment value can override a pin and produces a log.\nExact refuses an explicit dense twin or bounded lenient acceptance.\nExact does not promise text identity with plain autoregressive decoding.\nVerify widths and unaligned warm restores can change floating point results.\nManifest artefacts remain unavailable until the manifest job.\nUse `--print-effective-settings` to inspect settings without loading tensors or starting a server.\n\n'

def validate_value(row, value):
    name, rule = row['name'], row['value_rule']
    enums = {'--speculative': {'mtp', 'off', 'dspark', 'dflash2'},
             '--mtp-policy': {'survival', 'length'},
             '--mtp-draft-vocab': {'auto', 'latin', 'full'},
             '--think': {'auto', 'on', 'off'},
             '--reasoning-effort': {'auto', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}}
    def fail():
        raise ValueError(name + ': invalid value ' + repr(value))
    if name in enums and value not in enums[name]:
        fail()
    if rule in {'text', 'twin_width', 'lenient', 'tuning'}:
        return
    if rule == 'bool':
        if value not in {'0', '1'}:
            fail()
        return
    if rule == 'effort_map':
        labels = set()
        for entry in value.strip().split(','):
            label, sep, number = entry.partition(':')
            label, number = label.strip(), number.strip()
            try:
                bias = float(number)
            except ValueError:
                fail()
            if not sep or not label or label in labels or not math.isfinite(bias) or abs(bias) > 3.402823466e38:
                fail()
            labels.add(label)
        return
    if rule == 'uint':
        if not re.fullmatch(r'[0-9]+', value) or int(value) > 18446744073709551615:
            fail()
        if name in {'--sessions', '--max-request-bytes', '--max-connections'} and int(value) == 0:
            fail()
        limit = {'GUFO_DISK_EVICT_SKIP_TOKENS': 1000000, 'GUFO_DISK_IDLE_MS': 600000, 'GUFO_DISK_DEFER_MS': 600000}.get(name)
        if limit is not None and int(value) > limit:
            fail()
        return
    try:
        number = float(value)
    except ValueError:
        fail()
    if not math.isfinite(number) or value != value.rstrip():
        fail()
    if rule == 'number':
        return
    if rule == 'min_prob' and number != -1 and not 0 <= number <= 1:
        fail()
    if rule == 'prob' and not 0 <= number <= 1:
        fail()
    if rule == 'lambda' and not 0 < number <= 2:
        fail()
    if rule in {'transition', 'forward', 'readers'}:
        lower, upper = {'transition': (0, 2), 'forward': (64, 16384), 'readers': (1, 16)}[rule]
        if number != math.floor(number) or not lower <= number <= upper:
            fail()

def validate(data):
    if set(data) != {'schema', 'switches'} or data['schema'] != 1 or not isinstance(data['switches'], list):
        raise ValueError('invalid catalog schema')
    names = set()
    for row in data['switches']:
        if set(row) != FIELDS:
            raise ValueError('invalid row fields')
        name = row['name']
        if not isinstance(name, str) or not re.fullmatch(r'--[a-z0-9-]+|[A-Z][A-Z0-9_]+', name) or name in names:
            raise ValueError('invalid or duplicate name')
        names.add(name)
        for key, allowed in [('kind', {'env', 'flag'}), ('class', CLASSES), ('model_family', FAMILIES), ('platform', {'any', 'windows'}), ('hardware_scope', {'generic', 'strix-halo-tuned'}), ('value_rule', RULES)]:
            if row[key] not in allowed:
                raise ValueError(name + ': unknown ' + key)
        for key in ['purpose', 'default_text']:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(name + ': empty ' + key)
        if len(row['purpose'].split()) > 20 or re.search(r'\b(I|we|our|ours|us|unfortunately|sadly|mistake|bug)\b', row['purpose'], re.I):
            raise ValueError(name + ': invalid STE-style purpose')
        for key in ['requires', 'suppressed_by', 'aliases']:
            if not isinstance(row[key], list) or not all(isinstance(v, str) for v in row[key]):
                raise ValueError(name + ': invalid ' + key)
        if not set(row['requires']) <= REQUIRES:
            raise ValueError(name + ': unknown requires')
        if not isinstance(row['values'], dict) or set(row['values']) != {'default', 'subagents', 'exact'}:
            raise ValueError(name + ': invalid values')
        if not all(v is None or isinstance(v, str) and v for v in row['values'].values()):
            raise ValueError(name + ': invalid profile value')
        for key in ['built_in', 'inject_with']:
            if row[key] is not None and (not isinstance(row[key], str) or not row[key]):
                raise ValueError(name + ': invalid ' + key)
        for value in row['values'].values():
            if value is not None:
                validate_value(row, value)
        if type(row['exact_off']) is not bool or type(row['takes_value']) is not bool or not isinstance(row['note'], str):
            raise ValueError(name + ': invalid metadata')
        if (row['kind'] == 'flag') != name.startswith('--'):
            raise ValueError(name + ': kind mismatch')
    for row in data['switches']:
        for name in row['suppressed_by'] + ([row['inject_with']] if row['inject_with'] else []):
            if name not in names or not name.startswith('--'):
                raise ValueError('unknown flag reference: ' + name)
    return sorted(data['switches'], key=lambda row: row['name'])

def render_inc(rows):
    out = '// Generated by tools/release/gen_switches.py. Do not edit.\n#if defined(GUFO_RELEASE_ROWS)\n'
    for row in rows:
        out += 'json::parse(R"switch(' + json.dumps(row, ensure_ascii=False, separators=(',', ':')) + ')switch"),\n'
    out += '#elif defined(GUFO_RELEASE_SNAPSHOT)\n'
    for row in rows:
        if row['kind'] == 'env' and row['class'] != 'REMOVED':
            name = json.dumps(row['name'])
            out += 'environment[' + name + '] = CopyEnvironmentValue(std::getenv(' + name + '));\n'
    out += '#elif defined(GUFO_RELEASE_CATALOG)\n'
    for row in rows:
        if row['kind'] == 'env' and row['class'] != 'REMOVED':
            out += json.dumps(row['name']) + ',\n'
    return out + '#endif\n'

def cell(text):
    return text.replace('|', '\\|').replace('\n', ' ').replace('\r', ' ').replace('TODO:', 'Follow-up:') or 'None'

def table(rows, classes):
    out = '| Name | What it does | Default | When to change | Platform and hardware |\n| --- | --- | --- | --- | --- |\n'
    for row in rows:
        if row['class'] in classes:
            scope = row['platform'] + '; ' + row['hardware_scope'] + '; ' + row['model_family']
            out += '| `' + row['name'] + '` | ' + cell(row['purpose']) + ' | ' + cell(row['default_text']) + ' | ' + cell(row['note']) + ' | ' + scope + ' |\n'
    return out + '\n'

def render_docs(rows):
    return (INTRO + '## Options and defaults\n\n' + table(rows, {'DEFAULT', 'OPTION'}) + PROFILES + '## Diagnostics\n\n' + table(rows, {'DEBUG', 'INTERNAL'}) + '## Removed\n\nOther model switches remain available for their model families.\n\n' + table(rows, {'REMOVED', 'OTHER_MODEL'})).rstrip('\n') + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        rows = validate(json.loads((args.input or root / 'src/cli/serve/release_switches.json').read_text(encoding='utf-8')))
        outputs = {'src/cli/serve/release_switches.inc': render_inc(rows), 'docs/FLAGS.md': render_docs(rows)}
        if args.check:
            for name, content in outputs.items():
                if (root / name).read_text(encoding='utf-8') != content:
                    raise ValueError('generated output differs: ' + name)
            if (root / 'tests/fixtures/release_flags_golden.md').read_text(encoding='utf-8') != render_docs(rows):
                raise ValueError('C++ renderer golden differs')
            print('PASS: release catalog, generated header, docs, and C++ renderer golden')
        else:
            for name, content in outputs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                newline = '\r\n' if not path.exists() or b'\r\n' in path.read_bytes() else '\n'
                path.write_bytes(content.replace('\n', newline).encode('utf-8'))
            print('Generated', len(rows), 'switches')
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, str(error) + '\n')

if __name__ == '__main__':
    main()
