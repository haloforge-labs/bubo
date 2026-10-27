"""Prove row-only extensions through generation and the production C++ resolver."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/release'))
import gen_switches  # noqa: E402


def row(name, allowed, value, *, secret=False, rule='text'):
    # Complete row, with no name registration, group rule, or value table edit.
    return {
        'name': name, 'kind': 'flag' if name.startswith('--') else 'env',
        'class': 'OPTION', 'model_family': 'any', 'platform': 'any',
        'requires': [], 'values': {'default': value, 'subagents': None, 'exact': None},
        'exact_off': False, 'inject_with': None, 'suppressed_by': [],
        'purpose': 'Exercise a future setting.', 'default_text': value,
        'hardware_scope': 'generic', 'note': 'Use for contract tests.',
        'built_in': value, 'value_rule': rule, 'aliases': [], 'takes_value': True,
        'echo': not secret, 'secret': secret, 'group': 'Behaviour',
        'audience': 'debug', 'experimental': False, 'restart_required': True,
        'allowed': allowed, 'gui_control': not secret,
    }


def case(name, value, accepted=True, **extra):
    return {'name': name, 'value': str(value), 'accepted': accepted, **extra}


def text(value):
    return str(int(value)) if type(value) is bool else str(value)


def value_cases(rows):
    """Check every listed value/bound; retain grammar cases the row cannot encode."""
    cases = []
    for item in rows:
        name, allowed = item['name'], item['allowed']
        values = [text(value) for value in allowed.get('values', [])]
        values += [str(allowed[key]) for key in ('minimum', 'maximum') if key in allowed]
        values += [str(value) for value in allowed.get('also', [])]
        if 'exclusive_minimum' in allowed:
            low = allowed['exclusive_minimum']
            cases.append(case(name, low, False, value_only=True))
            values.append(str((low + allowed.get('maximum', low + 2)) / 2))
        for value in values:
            gen_switches.validate_value(item, value)
            cases.append(case(name, value, value_only=True))
        if 'values' in allowed:
            cases.append(case(name, 'not-listed', False, value_only=True))
        if 'minimum' in allowed:
            value = allowed['minimum'] - 1
            if value not in allowed.get('also', []):
                cases.append(case(name, value, False, value_only=True))
        if 'maximum' in allowed:
            cases.append(case(name, allowed['maximum'] + 1, False, value_only=True))
        if allowed['type'] in {'integer', 'number'}:
            cases += [case(name, value, False, value_only=True) for value in ('nan', 'inf')]
    grammar = {
        'forward': [(' +64suffix', True), ('16384tail', True), ('63tail', False),
                    ('16385tail', False), ('no-prefix', False)],
        'effort_map': [('low:-2,medium:-1,xhigh:0', True), ('custom:0.5', True),
                       ('low:1,low:2', False), ('medium:nan', False)],
        'tuning': [('all', True), ('none', True), ('+all', False),
                   ('unknown-tuning-field', False)],
        # Reader compatibility: these recommendations have never been limits.
        'lenient': [('arbitrary-disabled-text', True)],
        'twin_width': [('99', True), ('reader-fallback', True)],
    }
    for item in rows:
        for value, accepted in grammar.get(item['value_rule'], []):
            cases.append(case(item['name'], value, accepted, value_only=True))
    for value in ('-100', '0.25', '100'):
        cases.append(case('--think-marker-bias', value, value_only=True))
    # Pin Python/C++ agreement for every expressible constraint, not just defaults.
    by_name = {item['name']: item for item in rows}
    for test in cases:
        if by_name[test['name']]['value_rule'] == 'tuning':
            continue  # C++ owns platform tuning field names, as documented.
        try:
            gen_switches.validate_value(by_name[test['name']], test['value'])
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == test['accepted'], test
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validator', type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'src/cli/serve/release_switches.json').read_text(encoding='utf-8'))
    evidence = json.loads((ROOT / 'docs/switch-evidence.json').read_text(encoding='utf-8'))
    variants = [
        (row('GUFO_RECIPE_LIMIT', {'type': 'integer', 'minimum': 3, 'maximum': 11}, '7', rule='uint'),
         [('3', True), ('11', True), ('2', False), ('12', False), ('1.5', False)]),
        (row('--recipe-choice', {'type': 'enum', 'values': ['first', 'second']}, 'first'),
         [('first', True), ('second', True), ('third', False)]),
        (row('--recipe-credential', {'type': 'string'}, 'unset', secret=True),
         [('private-flag-credential', True)]),
        (row('GUFO_RECIPE_CREDENTIAL', {'type': 'string'}, 'unset', secret=True),
         [('private-env-credential', True)]),
    ]
    checks = value_cases(gen_switches.validate(catalog))
    count = 0
    with tempfile.TemporaryDirectory(prefix='bubo-switch-recipe-') as directory:
        root = Path(directory)
        fixture = root / 'tests/cli/contract/valid/switches.json'
        fixture.parent.mkdir(parents=True)
        fixture.write_text((ROOT / 'tests/cli/contract/valid/switches.json').read_text(encoding='utf-8'), encoding='utf-8')
        evidence_path = root / 'docs/switch-evidence.json'
        evidence_path.parent.mkdir(parents=True)
        for index, (new_row, values) in enumerate(variants):
            changed = copy.deepcopy(catalog)
            changed['switches'].append(new_row)
            rows = gen_switches.validate(changed)
            new_evidence = {**evidence, new_row['name']: None}
            evidence_path.write_text(json.dumps(new_evidence), encoding='utf-8')
            gen_switches.validate_evidence(rows, evidence_path)
            outputs = gen_switches.generated_outputs(rows, root)
            for name, content in outputs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            listing = json.loads(outputs['tests/cli/contract/switches.golden.json'])
            assert json.loads(outputs['tests/cli/contract/valid/switches.json'])['payload'] == listing
            assert new_row['name'] in outputs['src/cli/serve/release_switches.inc']
            assert gen_switches.generated_outputs(rows, root) == outputs
            cases = [case(new_row['name'], value, accepted, secret=new_row['secret'])
                     for value, accepted in values]
            if index == 0:
                cases += checks
            listing['cases'] = cases
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps(listing), encoding='utf-8')
            subprocess.run([str(args.validator.resolve()), '--registry-recipe', str(recipe)], check=True)
            count += len(cases)
    print(f'PASS: {len(variants)} row-only recipes, {count} value cases, 9 generated outputs per recipe')


if __name__ == '__main__':
    main()
