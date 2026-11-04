import { expect, test } from 'vitest'
import { scenarioByName } from '@/../tests/scenarios'
import {
  aboutParagraphs,
  defaultWords,
  settingLabel,
  unavailableHint,
  unitOf,
  valueWord,
} from './presentation'
import { canonicalSwitches } from './registry-view'

const view = scenarioByName('ready-idle').view
const row = view.switches[0]!

test('one value map names every enum value the same way everywhere', () => {
  expect(valueWord('--reasoning-effort', 'xhigh')).toBe('Extra high')
  expect(valueWord('--anything', 'auto')).toBe('Automatic')
  expect(valueWord('--npu-prefill', 'on')).toBe('On')
  expect(valueWord('--kv-cache-type', 'q8_0')).toBe('Q8')
  expect(valueWord('--kv-cache-type', 'f16')).toBe('f16')
  expect(valueWord('--speculative', 'dflash2')).toBe('DFlash 2')
  // A value with no words is shown as it is, never invented.
  expect(valueWord('--anything', 'whatever')).toBe('whatever')
})

test('units come from the name of the setting, in one place', () => {
  expect(unitOf('--request-timeout-ms')).toBe('ms')
  expect(unitOf('GUFO_DISK_IDLE_MS')).toBe('ms')
  expect(unitOf('GUFO_CACHE_HOST_FLOOR_MB')).toBe('MiB')
  expect(unitOf('--context')).toBe('tokens')
  expect(unitOf('--temperature')).toBeUndefined()
})

test('a default the CLI does not report is said plainly and a byte budget is read in GiB', () => {
  expect(defaultWords({ ...row, default_text: 'as shown' })).toBe(
    'Default not reported',
  )
  expect(defaultWords({ ...row, default_text: '' })).toBe(
    'Default not reported',
  )
  expect(
    defaultWords({
      ...row,
      name: '--cache-disk-bytes',
      default_text: '8589934592',
    }),
  ).toBe('8 GiB')
  expect(defaultWords({ ...row, default_text: 'on' })).toBe('on')
})

test('generated placeholders are no explanation', () => {
  expect(
    aboutParagraphs({
      ...row,
      purpose: 'Control debug hooks.',
      note: 'Inventory classification.',
    }),
  ).toEqual([])
  expect(
    aboutParagraphs({
      ...row,
      purpose: 'Set the maximum draft length.',
      note: 'unclassified',
    }),
  ).toEqual(['Set the maximum draft length.'])
  expect(
    aboutParagraphs({ ...row, purpose: 'Same text.', note: 'Same text.' }),
  ).toEqual(['Same text.'])
})

test('the GUI label wins where the GUI needs other words, else the registry label stands', () => {
  expect(
    settingLabel({
      ...row,
      name: '--refusal-direction',
      label: 'In-engine abliteration (Qwen3.8 Flash Next)',
    }),
  ).toBe('In-engine abliteration (Qwen3.8 Flash Next)')
  expect(
    settingLabel({ ...row, name: '--no-twin', label: 'Disable decode twin' }),
  ).toBe('Decode twin')
  expect(settingLabel({ ...row, name: '--top-p', label: 'Top-p' })).toBe(
    'Top-p',
  )
})

test('the short phrase for an unavailable setting keeps the reason for About', () => {
  expect(unavailableHint('This model is already abliterated.')).toBe(
    'Already abliterated',
  )
  expect(
    unavailableHint('In-engine abliteration is not tested for this model.'),
  ).toBe('Untested model')
  expect(
    unavailableHint(
      'Q8 KV cache is not supported for this model yet; it uses f16.',
    ),
  ).toBe('Not supported yet')
  expect(unavailableHint('Fast mode is for 512 experts only.')).toBe(
    'Unsupported model',
  )
})

test('a setting spelled as a flag and as an environment name is one editor', () => {
  const flag = { ...row, name: '--loop-breaker', label: 'Loop breaker' }
  const env = { ...row, name: 'GUFO_LOOP_BREAKER', label: 'Loop breaker' }
  const other = { ...row, name: 'GUFO_TRACE', label: 'Trace' }
  expect(
    canonicalSwitches([flag, env, other]).map((item) => item.name),
  ).toEqual(['--loop-breaker', 'GUFO_TRACE'])
})
