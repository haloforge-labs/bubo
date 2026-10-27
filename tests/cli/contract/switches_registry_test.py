"""Check the generated switch listing, schema, GUI panels, and evidence."""

import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/release'))
import gen_switches  # noqa: E402


def check_schema(value, schema, place='$', root=None):
    """Validate the JSON Schema keywords used by switches.schema.json."""
    root = schema if root is None else root
    if '$ref' in schema:
        schema = root['$defs'][schema['$ref'].removeprefix('#/$defs/')]
    types = schema.get('type')
    if types is not None:
        accepted = types if isinstance(types, list) else [types]
        matching = {
            'object': lambda v: isinstance(v, dict),
            'array': lambda v: isinstance(v, list),
            'string': lambda v: isinstance(v, str),
            'boolean': lambda v: type(v) is bool,
            'number': lambda v: (type(v) is int or type(v) is float),
            'null': lambda v: v is None,
        }
        if not any(matching[kind](value) for kind in accepted):
            raise AssertionError(f'{place}: expected {accepted}, got {type(value).__name__}')
    if 'const' in schema and value != schema['const']:
        raise AssertionError(f'{place}: wrong constant')
    if 'enum' in schema and value not in schema['enum']:
        raise AssertionError(f'{place}: value outside enum')
    if 'pattern' in schema and not re.fullmatch(schema['pattern'], value):
        raise AssertionError(f'{place}: pattern mismatch')
    if isinstance(value, dict):
        required = set(schema.get('required', []))
        if not required <= value.keys():
            raise AssertionError(f'{place}: missing {sorted(required - value.keys())}')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and not value.keys() <= properties.keys():
            raise AssertionError(f'{place}: extra {sorted(value.keys() - properties.keys())}')
        for name, child in value.items():
            if name in properties:
                check_schema(child, properties[name], f'{place}.{name}', root)
    if isinstance(value, list) and 'items' in schema:
        for number, child in enumerate(value):
            check_schema(child, schema['items'], f'{place}[{number}]', root)


def check_exact_integers(value):
    if isinstance(value, dict):
        for child in value.values():
            check_exact_integers(child)
    elif isinstance(value, list):
        for child in value:
            check_exact_integers(child)
    elif type(value) is int or (type(value) is float and value.is_integer()):
        if abs(value) > 2**53 - 1:
            raise AssertionError(f'Integer exceeds the exact JSON range: {value}')
    elif isinstance(value, str) and re.fullmatch(
            r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?:[eE][+-]?\d+)?', value):
        number = Decimal(value.replace(',', ''))
        if number == number.to_integral_value():
            check_exact_integers(int(number))


class SwitchRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / 'src/cli/serve/release_switches.json').read_text(encoding='utf-8'))
        cls.rows = gen_switches.validate(copy.deepcopy(cls.catalog))
        cls.listing = json.loads((ROOT / 'tests/cli/contract/switches.golden.json').read_text(encoding='utf-8'))
        cls.schema = json.loads((ROOT / 'tests/cli/contract/switches.schema.json').read_text(encoding='utf-8'))
        cls.panels = json.loads((ROOT / 'tests/cli/contract/engine-panels.json').read_text(encoding='utf-8'))
        cls.evidence = json.loads((ROOT / 'docs/switch-evidence.json').read_text(encoding='utf-8'))

    def test_catalog_and_listing(self):
        names = [row['name'] for row in self.rows]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.listing, {'schema_version': 1, 'switches': self.rows})
        fixture = json.loads((ROOT / 'tests/cli/contract/valid/switches.json').read_text(encoding='utf-8'))
        self.assertEqual(fixture['payload'], self.listing)
        self.assertEqual({row['group'] for row in self.rows}, gen_switches.GROUP_NAMES)
        self.assertEqual({row['audience'] for row in self.rows}, gen_switches.AUDIENCES)

    def test_secret_is_independent_of_echo(self):
        self.assertEqual({row['name'] for row in self.rows if row['secret']},
                         {'--api-key', 'BUBO_API_KEY', 'GUFO_EVAL_API_KEY', 'PATH'})
        self.assertTrue(all(not row['echo'] and not row['gui_control']
                            for row in self.rows if row['secret']))
        changed = copy.deepcopy(self.catalog)
        row = next(row for row in changed['switches'] if row['name'] == 'BUBO_API_KEY')
        row['name'] = 'FUTURE_CREDENTIAL'
        row['echo'] = True
        with self.assertRaisesRegex(ValueError, 'never echo-safe'):
            gen_switches.validate(changed)
        row['echo'] = False
        row['gui_control'] = True
        with self.assertRaisesRegex(ValueError, 'GUI-writable'):
            gen_switches.validate(changed)
        row['gui_control'] = False
        row['secret'] = 'yes'
        with self.assertRaises(ValueError):
            gen_switches.validate(changed)
        row.pop('secret')
        self.assertFalse(next(row for row in gen_switches.validate(changed)
                              if row['name'] == 'FUTURE_CREDENTIAL')['secret'])

    def test_listing_schema(self):
        check_schema(self.listing, self.schema)
        changed = copy.deepcopy(self.listing)
        del changed['switches'][0]['audience']
        with self.assertRaises(AssertionError):
            check_schema(changed, self.schema)

    def test_named_row_schema_and_optional_label(self):
        self.assertEqual(self.schema['properties']['switches']['items'], {'$ref': '#/$defs/switch_row'})
        self.assertNotIn('label', self.schema['$defs']['switch_row']['required'])
        changed = copy.deepcopy(self.listing)
        for row in changed['switches']:
            row.pop('label', None)
        check_schema(changed, self.schema)
        changed['switches'][0]['label'] = 42
        with self.assertRaises(AssertionError):
            check_schema(changed, self.schema)

    def test_labels_cover_user_and_advanced_rows(self):
        labeled = [row for row in self.rows if row['audience'] in ('user', 'advanced')]
        self.assertEqual(len(labeled), 48)
        for row in labeled:
            with self.subTest(name=row['name']):
                self.assertTrue(row['label'].strip())
        self.assertEqual(next(row['label'] for row in self.rows if row['name'] == 'GUFO_EXPERT_V2K'),
                         'Grouped expert kernel')

    def test_published_integers_are_exact(self):
        for row in self.rows:
            with self.subTest(name=row['name']):
                for field in ('values', 'built_in', 'default_text', 'allowed'):
                    check_exact_integers(row[field])
        check_exact_integers(self.listing)

    def test_integer_guard_rejects_unsafe_metadata(self):
        for field in ('default', 'minimum', 'maximum', 'suggested'):
            for value in (2**53, str(2**53), -(2**53), float(2**53),
                          '9,007,199,254,740,992', '9007199254740992.0'):
                with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                    check_exact_integers({field: {'value': value}})
        check_exact_integers({'minimum': -(2**53 - 1), 'maximum': str(2**53 - 1)})

    def test_boolean_defaults_match_controls(self):
        for name in ('GUFO_LOGIT_BIAS', 'GUFO_MODEL_SUPPRESS_TOKENS',
                     'GUFO_PRODUCTION_SAMPLING'):
            row = next(row for row in self.rows if row['name'] == name)
            with self.subTest(name=name):
                self.assertEqual(row['default_text'], '0')
                gen_switches.validate_value(row, row['default_text'])
                self.assertIsNone(row['values']['default'])
                self.assertIsNone(row['values']['subagents'])
                self.assertEqual(row['values']['exact'], '0')
                self.assertIsNone(row['built_in'])

    def test_effort_bias_is_a_user_control(self):
        row = next(row for row in self.rows if row['name'] == 'GUFO_THINK_MARKER_BIAS_BY_EFFORT')
        self.assertEqual(row['audience'], 'user')
        self.assertEqual(row['allowed']['suggested'], {'minimum': -2, 'maximum': 0})
        self.assertEqual(row['values']['default'], 'low:-2,medium:-1,xhigh:0')
        self.assertEqual(row['values']['subagents'], row['values']['default'])
        self.assertIsNone(row['values']['exact'])

    def test_engine_rows_without_gui_metadata(self):
        fields = {'echo', 'audience', 'gui_control', 'experimental', 'restart_required'}
        stripped = copy.deepcopy(self.catalog)
        for row in stripped['switches']:
            for field in fields:
                row.pop(field, None)
        normalized = gen_switches.validate(stripped)
        for row in normalized:
            with self.subTest(name=row['name']):
                source = next(item for item in self.catalog['switches'] if item['name'] == row['name'])
                self.assertEqual(row['allowed'], source['allowed'])
                self.assertEqual(row['group'], source['group'])
                self.assertEqual(row['audience'], 'debug')
                self.assertFalse(row['gui_control'])
                self.assertFalse(row['experimental'])
                self.assertFalse(row['echo'])
                self.assertEqual(row['restart_required'], row['class'] != 'REMOVED')
        self.assertEqual(gen_switches.render_inc(stripped['switches']), gen_switches.render_inc(normalized))
        self.assertFalse(any('audience' in row for row in stripped['switches']))

    def test_new_engine_row_uses_defaults_without_gui_edits(self):
        row = copy.deepcopy(next(row for row in self.rows if row['name'] == 'BUBO_RUNTIME_DIR'))
        for field in ('echo', 'audience', 'gui_control', 'experimental', 'restart_required'):
            row.pop(field, None)
        row['name'] = 'GUFO_FUTURE_ENGINE_OPTION'
        rows = gen_switches.validate({'schema': 1, 'switches': [row]})
        self.assertEqual(rows[0]['allowed'], {'type': 'string'})
        self.assertEqual(rows[0]['audience'], 'debug')
        self.assertIn('GUFO_FUTURE_ENGINE_OPTION', gen_switches.render_inc([row]))

    def test_runtime_override_is_a_complete_debug_row(self):
        row = next(row for row in self.rows if row['name'] == 'BUBO_RUNTIME_DIR')
        self.assertEqual(set(row), gen_switches.FIELDS | {'label'})
        self.assertEqual((row['audience'], row['platform'], row['allowed']),
                         ('debug', 'windows', {'type': 'string'}))
        self.assertTrue(row['echo'])
        self.assertFalse(row['gui_control'])

    def test_debug_labels_and_builtin_default_display(self):
        debug = [row for row in self.rows if row['audience'] == 'debug']
        self.assertTrue(debug)
        self.assertTrue(all(row.get('label', '').strip() for row in debug))
        floor = next(row for row in self.rows if row['name'] == 'GUFO_CACHE_HOST_FLOOR_MB')
        self.assertIsNone(floor['values']['default'])
        self.assertEqual(floor['default_text'], '4096')
        self.assertEqual(gen_switches.profile_text(floor),
                         'default=4096 (built-in). subagents=4096 (built-in). exact=4096 (built-in)')

    def test_invalid_optional_metadata_is_not_defaulted(self):
        for field, value in [('label', ''), ('label', ' \t'), ('label', 42), ('label', None),
                             ('audience', None), ('allowed', None), ('gui_control', None)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                changed = copy.deepcopy(self.catalog)
                changed['switches'][0][field] = value
                gen_switches.validate(changed)

    def test_gui_panels_cover_each_switch_once(self):
        self.assertEqual(self.panels['schema_version'], 1)
        panels = self.panels['panels']
        self.assertEqual([(panel['id'], panel['audience'], panel['collapsed']) for panel in panels],
                         [('main', 'user', False), ('advanced', 'advanced', False),
                          ('debug', 'debug', True)])
        self.assertEqual(panels[2]['warning'], 'These switches are for troubleshooting.')
        by_name = {row['name']: row for row in self.rows}
        seen = []
        for panel in panels:
            for section in panel['sections']:
                self.assertIn(section['group'], gen_switches.GROUP_NAMES)
                for name in section['switches']:
                    self.assertIn(name, by_name)
                    self.assertEqual(by_name[name]['audience'], panel['audience'])
                    self.assertEqual(by_name[name]['group'], section['group'])
                    seen.append(name)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen) | set(self.panels['omitted_controls']), set(by_name))
        self.assertFalse(set(seen) & set(self.panels['omitted_controls']))
        self.assertEqual(set(self.panels['omitted_controls']),
                         {name for name, row in by_name.items() if not row['gui_control']})
        hardware = next(section['switches'] for section in panels[0]['sections']
                        if section['group'] == 'Hardware and memory')
        self.assertEqual(hardware[:2], ['--no-twin', '--npu-prefill'])
        self.assertEqual(by_name['GUFO_EXPERT_V2K']['audience'], 'debug')
        self.assertTrue(by_name['GUFO_EXPERT_V2K']['experimental'])
        self.assertEqual(by_name['GUFO_EXPERT_V2K']['built_in'], '0')
        self.assertNotIn('GUFO_REASONING_BUDGET', [name for panel in panels[:2]
                                                  for section in panel['sections']
                                                  for name in section['switches']])

    def test_flag_env_links_and_allowed_ranges(self):
        by_name = {row['name']: row for row in self.rows}
        self.assertIn('--loop-breaker', by_name['GUFO_LOOP_BREAKER']['suppressed_by'])
        self.assertIn('--think-marker-bias', by_name['GUFO_THINK_MARKER_BIAS']['suppressed_by'])
        self.assertEqual(by_name['--loop-breaker']['allowed']['values'], ['on', 'off'])
        self.assertEqual(by_name['--think-marker-bias']['allowed']['type'], 'number')
        bias = by_name['--think-marker-bias']['allowed']
        self.assertEqual((bias['suggested']['minimum'], bias['suggested']['maximum']), (-2, 0))
        self.assertNotIn('step', bias)
        # GUI limits must not tighten the finite-number engine parser.
        for value in ('-100', '0.25', '100'):
            gen_switches.validate_value(by_name['--think-marker-bias'], value)
        self.assertEqual(by_name['GUFO_MAX_IMAGE_TOKENS']['allowed']['minimum'], 1)
        for value in ('on', 'off'):
            gen_switches.validate_value(by_name['--loop-breaker'], value)
        with self.assertRaises(ValueError):
            gen_switches.validate_value(by_name['--loop-breaker'], 'maybe')

    def test_groups_are_row_data(self):
        changed = copy.deepcopy(self.catalog)
        row = next(row for row in changed['switches'] if row['name'] == '--context')
        for group in gen_switches.GROUP_ORDER:
            row['group'] = group
            actual = gen_switches.validate(changed)
            self.assertEqual(next(item['group'] for item in actual if item['name'] == row['name']), group)

    def test_process_and_host_controls_are_not_gui_settings(self):
        excluded = {'--help', '--print-effective-settings', 'HOME', 'LOCALAPPDATA', 'XDG_CACHE_HOME'}
        for row in self.rows:
            if row['name'] in excluded or row['class'] == 'REMOVED':
                self.assertFalse(row['gui_control'], row['name'])

    def test_missing_row_facts_are_rejected(self):
        for field in ('allowed', 'group'):
            changed = copy.deepcopy(self.catalog)
            del changed['switches'][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                gen_switches.validate(changed)

    def test_bad_metadata_is_rejected(self):
        for field, value in [('audience', 'hidden'), ('group', 'Hardware'),
                             ('allowed', {'type': 'integer', 'minimum': 8, 'maximum': 2}), ('experimental', 'yes')]:
            changed = copy.deepcopy(self.catalog)
            changed['switches'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                gen_switches.validate(changed)
        changed = copy.deepcopy(self.catalog)
        changed['switches'].append(copy.deepcopy(changed['switches'][0]))
        with self.assertRaises(ValueError):
            gen_switches.validate(changed)


if __name__ == '__main__':
    unittest.main()
