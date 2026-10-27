#!/usr/bin/env python3
"""Validate the release catalog and regenerate committed C++ and Markdown."""
import argparse
import copy
import json
import math
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_brand import check as check_brand

CLASSES = {'DEFAULT', 'OPTION', 'DEBUG', 'INTERNAL', 'REMOVED', 'OTHER_MODEL'}
FAMILIES = {'any', 'qwen3.8-flash-next', 'qwen3.8-27b', 'qwen3.6-a3b', 'deepseek-v4-flash', 'minimax-h3', 'qwen-image-2.1', 'qwen3-tts', 'qwen3-asr'}
REQUIRES = {'think!=off', 'sessions==1 or GUFO_DENSE_TWIN_BATCH==1', 'speculative==mtp', 'artefact:twin', 'artefact:draft_head', 'artefact:mtp', 'artefact:mmproj'}
RULES = {'text', 'bool', 'min_prob', 'prob', 'lambda', 'transition', 'number', 'effort_map', 'lenient', 'forward', 'twin_width', 'uint', 'int31', 'readers', 'tuning', 'on_off'}
AUDIENCES = {'user', 'advanced', 'debug'}
FIELDS = {'name', 'kind', 'class', 'model_family', 'platform', 'requires', 'values', 'exact_off', 'inject_with', 'suppressed_by', 'purpose', 'default_text', 'hardware_scope', 'note', 'built_in', 'value_rule', 'aliases', 'takes_value', 'echo', 'secret', 'group', 'audience', 'experimental', 'restart_required', 'allowed', 'gui_control'}
OPTIONAL_FIELDS = {'label'}

# Group assignment and accepted values live on each catalog row. Only the
# documented group order and optional presentation defaults are shared here.
GROUP_ORDER = ['Hardware and memory', 'Memory', 'Speed', 'Behaviour', 'Server', 'Other']
GROUP_NAMES = set(GROUP_ORDER)
NORMALIZATION = {
    'defaults': {'echo': False, 'secret': False, 'audience': 'debug', 'gui_control': False,
                 'experimental': False, 'restart_required': True},
    'removed_restart_required': False,
    'groups': GROUP_ORDER,
}


def normalize_row(row):
    if not isinstance(row, dict):
        raise ValueError('invalid switch row')
    result = copy.deepcopy(row)
    for key, value in NORMALIZATION['defaults'].items():
        result.setdefault(key, copy.deepcopy(value))
    if 'restart_required' not in row and row.get('class') == 'REMOVED':
        result['restart_required'] = NORMALIZATION['removed_restart_required']
    return result
INTRO = '# Release switches\n\nGenerated from `src/cli/serve/release_switches.json`.\n\nProfile defaults apply to Qwen3.8 Flash Next. A Gemma 4 package with an MTP file shares the MTP rows (`--speculative`, `--mtp-model`).\nOther model families keep their existing defaults except for the exact seed and NPU profile policy.\nExplicit flags and real environment values take precedence.\nEmpty environment values count as unset and produce a warning.\n\n'
PROFILES = '## Profiles\n\n| Profile | Effect |\n| --- | --- |\n| default | Use fast upstream practice for one session and medium reasoning effort. Output is not byte-exact. |\n| subagents | Use four sessions with 131072 context tokens each and a 4 GiB RAM cache. |\n| exact | Use reproducible plain autoregressive decoding with one session and no prompt cache. |\n| vanilla | Keep upstream defaults. Inject no environment values or flag defaults. |\n\nSubagents verifies up to 524288 total context tokens.\nUse `--allow-unmeasured-memory` above this measured limit.\nFour concurrent deep sessions can reach the host RAM guard near 50K tokens per session.\nThe default profile follows upstream gufo practice: fast output that is not byte-exact.\nExact is a reproducibility preset.\nOutput is byte-identical across runs of the same build and seed.\nOutput is identical to plain autoregressive decoding of that build with the same settings.\nOutput is not identical across builds or to other engines.\nExact disables speculation, behaviour policies, lenient acceptance, the dense twin, the draft head, and RAM and disk prompt caches.\nExact uses one session, forward size 4096, and seed 1 unless a seed is supplied.\nExact refuses values that enable disabled settings or change its session count or prefill geometry.\nIt runs at about half the decode speed, and every request prefills its full prompt.\nSidecar files (MTP, vision, dense twin, draft head) are set with their own options.\nReference: `bubo serve --profile vanilla --speculative off --no-cache --sessions 1 --seed 1 --think on --reasoning-effort medium`.\nUse `--print-effective-settings` to inspect settings without loading tensors or starting a server.\n\n'

def validate_allowed(row):
    allowed = row['allowed']
    if not isinstance(allowed, dict) or allowed.get('type') not in {'boolean', 'enum', 'integer', 'number', 'string'}:
        raise ValueError(row['name'] + ': invalid allowed type')
    if set(allowed) - {'type', 'values', 'minimum', 'exclusive_minimum', 'maximum', 'also', 'format', 'suggested'}:
        raise ValueError(row['name'] + ': unknown allowed field')
    for key in ('minimum', 'exclusive_minimum', 'maximum'):
        if key in allowed and (type(allowed[key]) not in (int, float) or not math.isfinite(allowed[key])):
            raise ValueError(row['name'] + ': invalid allowed bound')
    if 'minimum' in allowed and 'exclusive_minimum' in allowed:
        raise ValueError(row['name'] + ': conflicting lower bounds')
    low = allowed.get('minimum', allowed.get('exclusive_minimum'))
    if low is not None and 'maximum' in allowed and (low > allowed['maximum'] or
            low == allowed['maximum'] and 'exclusive_minimum' in allowed):
        raise ValueError(row['name'] + ': empty allowed range')
    if allowed['type'] in {'boolean', 'enum'} and not allowed.get('values'):
        raise ValueError(row['name'] + ': missing allowed values')
    if 'values' in allowed and (not isinstance(allowed['values'], list) or
            not all(type(value) in (str, bool, int, float) for value in allowed['values'])):
        raise ValueError(row['name'] + ': invalid allowed values')
    if 'also' in allowed and (not isinstance(allowed['also'], list) or
            not all(type(value) in (int, float) and math.isfinite(value) for value in allowed['also'])):
        raise ValueError(row['name'] + ': invalid allowed alternatives')
    if 'format' in allowed and not isinstance(allowed['format'], str):
        raise ValueError(row['name'] + ': invalid allowed format')
    if 'suggested' in allowed and not isinstance(allowed['suggested'], dict):
        raise ValueError(row['name'] + ': invalid suggested values')


def validate_value(row, value):
    name, rule, allowed = row['name'], row['value_rule'], row['allowed']
    def fail():
        shown = '<redacted>' if row.get('secret') else value
        raise ValueError(name + ': invalid value ' + repr(shown))
    def text(candidate):
        return str(int(candidate)) if type(candidate) is bool else str(candidate)
    if 'values' in allowed:
        if value not in [text(candidate) for candidate in allowed['values']]:
            fail()
        return
    # These readers have grammars that numeric bounds/enum lists cannot express.
    # Keep their grammar here; the row still owns any numeric range.
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
    if allowed['type'] == 'string':
        # Platform tuning names are owned and checked by the C++ tuning reader.
        return
    if rule == 'forward':
        # PrefillChunkTokens intentionally accepts a signed decimal prefix.
        prefix = re.match(r'[ \t\r\n\v\f]*[+-]?[0-9]+', value)
        if prefix is None:
            fail()
        number = int(prefix.group())
    elif rule in {'uint', 'int31'}:
        if not re.fullmatch(r'[0-9]+', value):
            fail()
        number = int(value)
    else:
        try:
            number = float(value)
        except ValueError:
            fail()
        if not math.isfinite(number) or value != value.rstrip():
            fail()
        if allowed['type'] == 'integer' and number != math.floor(number):
            fail()
    if number in allowed.get('also', []):
        return
    if ('minimum' in allowed and number < allowed['minimum'] or
            'exclusive_minimum' in allowed and number <= allowed['exclusive_minimum'] or
            'maximum' in allowed and number > allowed['maximum']):
        fail()

def validate(data):
    if set(data) != {'schema', 'switches'} or data['schema'] != 1 or not isinstance(data['switches'], list):
        raise ValueError('invalid catalog schema')
    names = set()
    rows = [normalize_row(row) for row in data['switches']]
    for row in rows:
        if row['secret'] and row['echo']:
            raise ValueError(row['name'] + ': never echo-safe')
        if row['secret'] and row['gui_control']:
            raise ValueError(row['name'] + ': secret cannot be GUI-writable')
        if not FIELDS <= set(row) or set(row) - FIELDS - OPTIONAL_FIELDS:
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
        if 'label' in row and (not isinstance(row['label'], str) or not row['label'].strip()):
            raise ValueError(name + ': invalid label')
        # A GUI shows a boolean flag as Off/On, and On sets the flag. So the
        # label of a --no-* flag names what it turns off, never the feature.
        if 'label' in row and name.startswith('--no-') and not row['takes_value'] and not re.match(r'(Disable|Skip) ', row['label']):
            raise ValueError(name + ': the label of a --no-* flag must start with "Disable " or "Skip "')
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
        validate_allowed(row)
        for value in row['values'].values():
            if value is not None:
                validate_value(row, value)
        if type(row['secret']) is not bool or type(row['echo']) is not bool or type(row['exact_off']) is not bool or type(row['takes_value']) is not bool or not isinstance(row['note'], str):
            raise ValueError(name + ': invalid metadata')
        if type(row['gui_control']) is not bool:
            raise ValueError(name + ': invalid gui_control')
        if row['class'] in {'DEFAULT', 'OPTION'}:
            note = row['note']
            if not 1 <= len(note.split()) <= 20 or not note.endswith('.') or note.count('.') != 1:
                raise ValueError(name + ': note must be one sentence of at most 20 words')
        if row['group'] not in GROUP_NAMES:
            raise ValueError(name + ': invalid group')
        if row['audience'] not in AUDIENCES:
            raise ValueError(name + ': invalid audience')
        if type(row['experimental']) is not bool or type(row['restart_required']) is not bool:
            raise ValueError(name + ': invalid feature metadata')
        if row['class'] == 'REMOVED' and row['restart_required']:
            raise ValueError(name + ': removed switch cannot require restart')
        if row['experimental'] and name != 'GUFO_EXPERT_V2K':
            raise ValueError(name + ': experimental switch not approved')
        if (row['kind'] == 'flag') != name.startswith('--'):
            raise ValueError(name + ': kind mismatch')
    for row in rows:
        for name in row['suppressed_by'] + ([row['inject_with']] if row['inject_with'] else []):
            if name not in names or not name.startswith('--'):
                raise ValueError('unknown flag reference: ' + name)
    return sorted(rows, key=lambda row: row['name'])

def render_inc(rows):
    rows = validate({'schema': 1, 'switches': rows})
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
        if row['kind'] == 'env' and row['class'] != 'REMOVED' and row.get('echo', False):
            out += json.dumps(row['name']) + ',\n'
    out += '#elif defined(GUFO_RELEASE_NORMALIZATION)\n'
    out += 'json::parse(R"switch(' + json.dumps(NORMALIZATION, ensure_ascii=False, separators=(',', ':')) + ')switch")\n'
    return out + '#endif\n'

def cell(text):
    return text.replace('|', '\\|').replace('\n', ' ').replace('\r', ' ').replace('TODO:', 'Follow-up:') or 'None'

def allowed_text(allowed):
    if 'values' in allowed:
        return ', '.join(str(value).lower() if isinstance(value, bool) else str(value)
                         for value in allowed['values'])
    if 'minimum' in allowed or 'exclusive_minimum' in allowed:
        low = allowed.get('minimum', allowed.get('exclusive_minimum'))
        bracket = '(' if 'exclusive_minimum' in allowed else '['
        high = allowed.get('maximum', '∞')
        extra = ' or ' + ', '.join(map(str, allowed['also'])) if 'also' in allowed else ''
        return f'{allowed["type"]} {bracket}{low}, {high}]{extra}'
    return allowed['type'] + (': ' + allowed['format'] if 'format' in allowed else '')


def profile_text(row):
    values = row['values']
    exact = values['exact'] if values['exact'] is not None else values['default']
    return '. '.join(f'{label}={value if value is not None else row["default_text"] + " (built-in)"}'
                     for label, value in [('default', values['default']),
                                          ('subagents', values['subagents']),
                                          ('exact', exact)])


def table(rows, classes):
    out = ('| Name | Group | Audience | What it does | Profile defaults | Built-in | Allowed | Restart | Experimental | Platform and hardware | Requires | Suppressed by | When to change |\n'
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n')
    for row in rows:
        if row['class'] in classes:
            scope = row['platform'] + '; ' + row['hardware_scope'] + '; ' + row['model_family']
            out += ('| `' + row['name'] + '` | ' + row['group'] + ' | ' + row['audience'] +
                    ' | ' + cell(row['purpose']) + ' | ' + cell(profile_text(row)) +
                    ' | ' + cell(row['built_in'] if row['built_in'] is not None else row['default_text']) + ' | ' +
                    cell(allowed_text(row['allowed'])) + ' | ' +
                    ('yes' if row['restart_required'] else 'no') + ' | ' +
                    ('yes' if row['experimental'] else 'no') + ' | ' + scope +
                    ' | ' + cell(', '.join(row['requires'])) + ' | ' +
                    cell(', '.join(row['suppressed_by'])) + ' | ' +
                    cell(row['note']) + ' |\n')
    return out + '\n'

def render_docs(rows):
    rows = validate({'schema': 1, 'switches': rows})
    return (INTRO + 'The optional `label` field supplies short display wording.\n'
            'The `group`, `audience`, `experimental`, `restart_required`, and `allowed` fields are typed registry metadata.\n'
            'Each row owns its group and allowed values; suggested values document reader recommendations, not enforced limits.\n'
            'The default audience is debug, with no GUI control.\n'
            'Switches default to non-experimental and require restart, except removed switches.\n'
            'A null profile value means no value is injected: the engine built-in in default_text applies. The exact profile inherits a non-null default profile value where shown.\n'
            'Changing a setting marked Restart requires a new server start.\n'
            'The `debug` audience appears in the collapsed Debug section of the Advanced panel.\n\n'
            '## Options and defaults\n\n' + table(rows, {'DEFAULT', 'OPTION'}) +
            PROFILES + '## Diagnostics\n\n' + table(rows, {'DEBUG', 'INTERNAL'}) +
            '## Removed and other models\n\nOther model switches remain available for their model families.\n\n' +
            table(rows, {'REMOVED', 'OTHER_MODEL'}) +
            '## Engine-managed queue cap\n\n'
            '| Name | What it does | Default | When to change | Platform and hardware |\n'
            '| --- | --- | --- | --- | --- |\n'
            '| `GPU_MAX_HW_QUEUES` | Set the hardware queue cap. | engine sets | The engine manages this queue cap; a real environment value wins. | any; generic; any |\n').rstrip('\n') + '\n'


def render_listing(rows):
    return json.dumps({'schema_version': 1, 'switches': rows}, indent=2, ensure_ascii=False) + '\n'


def render_fixture(rows, root):
    # Preserve the independently recorded event envelope; the listing payload is generated.
    fixture = json.loads((root / 'tests/cli/contract/valid/switches.json').read_text(encoding='utf-8'))
    fixture['payload'] = json.loads(render_listing(rows))
    return json.dumps(fixture, indent=2, ensure_ascii=False) + '\n'


def render_panels(rows):
    forbidden = {row['name'] for row in rows if not row['gui_control']}
    panels = []
    for audience, panel_id in [('user', 'main'), ('advanced', 'advanced'),
                               ('debug', 'debug')]:
        sections = []
        for group in GROUP_ORDER:
            names = [row['name'] for row in rows
                     if row['audience'] == audience and row['group'] == group and row['name'] not in forbidden]
            if audience == 'user' and group == 'Hardware and memory':
                priority = ['--no-twin', '--npu-prefill', '--twin-model']
                names.sort(key=lambda name: (priority.index(name) if name in priority else len(priority), name))
            if names:
                sections.append({'group': group, 'switches': names})
        panel = {'id': panel_id, 'audience': audience,
                 'collapsed': audience == 'debug', 'sections': sections}
        if audience == 'debug':
            panel['warning'] = 'These switches are for troubleshooting.'
        panels.append(panel)
    return json.dumps({'schema_version': 1, 'panels': panels,
                       'omitted_controls': sorted(forbidden),
                       'omission_reason': 'Rows with gui_control false are excluded from GUI controls.'}, indent=2, ensure_ascii=False) + '\n'


def render_audience_doc(rows):
    counts = {audience: sum(row['audience'] == audience for row in rows)
              for audience in ('user', 'advanced', 'debug')}
    out = ('# Switch audiences\n\nGenerated from `src/cli/serve/release_switches.json`.\n\n'
           'The main panel shows `user` switches. The Advanced panel shows `advanced` switches. '
           'Its collapsed Debug section shows permitted `debug` switches with their purpose and default for troubleshooting.\n'
           'GUFO_REASONING_BUDGET, GUFO_THINK_GUARD_ESCALATE, GUFO_LOOP_BREAKER_EOG_CLOSE and GUFO_LOOP_BREAKER_TRANSITION remain in the complete CLI diagnostics registry. '
           'The policy against thinking-token budgets, caps and force-close controls excludes them from GUI controls.\n'
           'Changes to a running server take effect after restart when `restart_required` is true.\n\n'
           'The decode twin head makes writing faster. Hardware controls take effect after restart.\n\n'
           '## Add a switch\n\n'
           'Add its complete row to `src/cli/serve/release_switches.json`, including `group` and `allowed`. '
           'Choose a group from Hardware and memory, Memory, Speed, Behaviour, Server, or Other. '
           'Add the switch name with a null value to `docs/switch-evidence.json` for the name-coverage contract. '
           'Run `python tools/release/gen_switches.py`, then `python tools/release/gen_switches.py --check`. '
           'The generator writes the C++ include, family include, flags docs, flags golden, listing golden, listing schema, '
           'GUI panel fixture, audience docs, and the payload of `tests/cli/contract/valid/switches.json`; '
           'the recorded event envelope stays intact. No Go or TypeScript edit is needed.\n\n'
           '`allowed` owns enum values and numeric bounds. `allowed.suggested` records reader recommendations '
           'without restricting accepted values. Unsigned decimal, prefill prefix, effort-map, and platform-tuning '
           'grammars remain in the CLI reader; the recipe test checks every listed value and numeric boundary against C++. '
           'Secret rows must set `echo: false` and `gui_control: false`.\n\n'
           '## CLI registry contract\n\n'
           '`bubo serve --list-switches --json` emits a `switches` event whose payload is the object in '
           '`tests/cli/contract/switches.golden.json`: `schema_version: 1` and one '
           'complete catalog row per switch. The JSON schema is '
           '`tests/cli/contract/switches.schema.json`. The listing is safe before '
           'model load and server start. The renderer must preserve the catalog order.\n\n'
           '`RenderReleaseDocs` must match `docs/FLAGS.md` and '
           '`tests/fixtures/release_flags_golden.md`, including the final '
           'engine-managed queue cap table retained for the brand gate.\n\n'
           '`--loop-breaker on|off` maps to `GUFO_LOOP_BREAKER` values `1|0`. '
           '`--think-marker-bias` maps its finite number to '
           '`GUFO_THINK_MARKER_BIAS`. Each environment row lists the CLI flag in '
           '`suppressed_by`; an explicit CLI value takes precedence over the '
           'environment and profile default. Exact keeps both controls off or zero.\n\n'
           '`GUFO_MAX_IMAGE_TOKENS` accepts positive integers. The disk cache '
           'byte caps use unsigned integers. The C++ value validator and '
           'the JSON schema must agree with these catalog rules.\n\n'
           f'Counts: user {counts["user"]}, advanced {counts["advanced"]}, debug {counts["debug"]}.\n\n'
           '| Switch | Group | Audience | Purpose |\n| --- | --- | --- | --- |\n')
    for row in rows:
        out += f'| `{row["name"]}` | {row["group"]} | {row["audience"]} | {cell(row["purpose"])} |\n'
    return out


def render_schema():
    string = {'type': 'string'}
    optional_string = {'type': ['string', 'null']}
    string_list = {'type': 'array', 'items': string}
    row_properties = {
        'name': {'type': 'string', 'pattern': r'^(--[a-z0-9-]+|[A-Z][A-Z0-9_]+)$'},
        'kind': {'enum': ['flag', 'env']}, 'class': {'enum': sorted(CLASSES)},
        'model_family': {'enum': sorted(FAMILIES)},
        'platform': {'enum': ['any', 'windows']}, 'requires': string_list,
        'values': {'type': 'object', 'required': ['default', 'subagents', 'exact'],
                   'additionalProperties': True,
                   'properties': {key: optional_string for key in ('default', 'subagents', 'exact')}},
        'exact_off': {'type': 'boolean'}, 'inject_with': optional_string,
        'suppressed_by': string_list, 'purpose': string, 'default_text': string,
        'label': {'type': 'string', 'minLength': 1, 'pattern': r'.*\S.*'},
        'hardware_scope': {'enum': ['generic', 'strix-halo-tuned']},
        'note': string, 'built_in': optional_string,
        'value_rule': {'enum': sorted(RULES)}, 'aliases': string_list,
        'takes_value': {'type': 'boolean'}, 'echo': {'type': 'boolean'}, 'secret': {'type': 'boolean'},
        'group': {'enum': sorted(GROUP_NAMES)},
        'audience': {'enum': sorted(AUDIENCES)},
        'experimental': {'type': 'boolean'},
        'restart_required': {'type': 'boolean'},
        'gui_control': {'type': 'boolean'},
        'allowed': {'type': 'object', 'required': ['type'],
                    'additionalProperties': True,
                    'properties': {'type': {'enum': ['boolean', 'enum', 'integer', 'number', 'string']},
                                   'values': {'type': 'array', 'items': {'type': ['boolean', 'number', 'string']}},
                                   'minimum': {'type': 'number'}, 'maximum': {'type': 'number'},
                                   'exclusive_minimum': {'type': 'number'},
                                   'also': {'type': 'array', 'items': {'type': 'number'}},
                                   'format': string}},
    }
    return json.dumps({'$schema': 'https://json-schema.org/draft/2020-12/schema',
                       'title': 'Bubo serve switch listing', 'type': 'object',
                       'required': ['schema_version', 'switches'],
                       'additionalProperties': True,
                       '$defs': {'switch_row': {
                           'title': 'Switch row', 'type': 'object', 'required': sorted(FIELDS),
                           'additionalProperties': True, 'properties': row_properties}},
                       'properties': {'schema_version': {'const': 1},
                                      'switches': {'type': 'array', 'items': {'$ref': '#/$defs/switch_row'}}}},
                      indent=2, ensure_ascii=False) + '\n'


def validate_evidence(rows, path):
    evidence = json.loads(path.read_text(encoding='utf-8'))
    names = {row['name'] for row in rows}
    if set(evidence) != names:
        raise ValueError('switch evidence names differ from catalog')
    for name, record in evidence.items():
        if record is None:
            continue
        if not isinstance(record, dict) or set(record) != {'text', 'conditions', 'source_path'}:
            raise ValueError(name + ': invalid evidence fields')
        if not all(isinstance(value, str) and value.strip() for value in record.values()):
            raise ValueError(name + ': empty evidence field')
        if not record['source_path'].replace('\\', '/').startswith('release-measurements/'):
            raise ValueError(name + ': evidence source outside measured release records')

def generated_outputs(rows, root):
    families = '// Generated from the catalog FAMILIES set. Do not edit.\n'
    families += ''.join(json.dumps(family) + ',\n' for family in sorted(FAMILIES - {'any'}))
    return {'src/cli/serve/release_switches.inc': render_inc(rows),
               'src/cli/serve/release_families.inc': families,
               'docs/FLAGS.md': render_docs(rows),
               'tests/fixtures/release_flags_golden.md': render_docs(rows),
               'tests/cli/contract/switches.golden.json': render_listing(rows),
               'tests/cli/contract/valid/switches.json': render_fixture(rows, root),
               'tests/cli/contract/switches.schema.json': render_schema(),
               'tests/cli/contract/engine-panels.json': render_panels(rows),
               'docs/SWITCH-AUDIENCE.md': render_audience_doc(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        rows = validate(json.loads((args.input or root / 'src/cli/serve/release_switches.json').read_text(encoding='utf-8')))
        outputs = generated_outputs(rows, root)
        validate_evidence(rows, root / 'docs/switch-evidence.json')
        if args.check:
            for name, content in outputs.items():
                if (root / name).read_text(encoding='utf-8') != content:
                    raise ValueError('generated output differs: ' + name)
            errors = check_brand(root)
            if errors:
                raise ValueError('\n'.join(errors))
            print(f'PASS: {len(rows)} switches, {len(outputs)} generated files, evidence names, and brand gate')
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
