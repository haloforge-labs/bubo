import { expect, test } from 'vitest'
import type { View } from '@/lib/bindings'
import { fieldFromRow } from '@/lib/registry'
import { decodeValue, effectiveValue, groupRows, registryRow } from './registry'
import { scenarioByName } from '@/../tests/scenarios'

const view = scenarioByName('ready-idle').view
const row = view.switches[0]!

// Ports the applicable registry and live-data rules from b2-settings.test.js
// and live.test.js. Removed per-switch branches stay removed: CLI gui_control
// and audience are the sole visibility contract.
test('audience groups preserve all visible rows and hide gui_control false', () => {
  const visible = view.switches.filter((item) => item.gui_control)
  const groups = ['user', 'advanced', 'debug'].flatMap((audience) =>
    Object.values(groupRows(view.switches, audience)).flat(),
  )
  expect(groups).toHaveLength(visible.length)
  expect(groups.every((item) => item.gui_control)).toBe(true)
  let unsupported = 0
  const unsupportedRows: string[] = []
  for (const item of visible) {
    try {
      if (fieldFromRow(registryRow(item)).control === 'unsupported') {
        unsupported++
        unsupportedRows.push(
          `${item.name}: ${item.allowed.properties?.format?.text}`,
        )
      }
    } catch {
      unsupported++
    }
  }
  console.log(
    `Recorded registry: ${visible.length} visible; ${visible.filter((item) => item.label === undefined).length} label fallbacks; ${unsupported} unsupported constraints.`,
  )
  if (unsupportedRows.length)
    console.log(`Unsupported constraints: ${unsupportedRows.join('; ')}`)
  expect(
    Object.values(groupRows([{ ...row, gui_control: false }], row.audience)),
  ).toHaveLength(0)
})
test('candidate registry shows the 8 GiB RAM cache default and keeps expert overlays hidden', () => {
  const ramCache = view.switches.find(
    (item) => item.name === '--cache-ram-bytes',
  )
  expect(ramCache?.values.default?.text).toBe('8589934592')
  expect(ramCache?.default_text).toBe('8589934592')
  expect(ramCache?.gui_control).toBe(true)
  const visible = ['user', 'advanced', 'debug'].flatMap((audience) =>
    Object.values(groupRows(view.switches, audience)).flat(),
  )
  expect(visible.some((item) => item.name === '--cache-ram-bytes')).toBe(true)
  for (const name of [
    'GUFO_EXPERT_OVERLAY_MODEL',
    'GUFO_EXPERT_OVERLAY_KEEP',
  ]) {
    expect(view.switches.find((item) => item.name === name)?.gui_control).toBe(
      false,
    )
    expect(visible.some((item) => item.name === name)).toBe(false)
  }
})
test('candidate 4 engine switches preserve CLI facts without editable controls', () => {
  const visible = ['user', 'advanced', 'debug'].flatMap((audience) =>
    Object.values(groupRows(view.switches, audience)).flat(),
  )
  for (const name of [
    'GUFO_INCREMENTAL_TOKENIZE',
    'GUFO_ROUTED_WMMA_IQ',
    'GUFO_IQ_ROUTE_MIN_ROWS',
  ]) {
    const item = view.switches.find((entry) => entry.name === name)
    expect(item, name).toBeDefined()
    expect(item?.gui_control).toBe(false)
    expect(item?.label).toBeTruthy()
    expect(item?.purpose).toBeTruthy()
    expect(item?.note).toBeTruthy()
    expect(visible.some((entry) => entry.name === name)).toBe(false)
  }
})

test('search matches names labels and purposes without duplicating rows', () => {
  const rows = [
    {
      ...row,
      name: '--synthetic',
      purpose: 'Searchable fixture',
      label: 'Fixture label',
      gui_control: true,
    },
  ]
  for (const search of ['SYNTHETIC', 'searchable', 'fixture label'])
    expect(
      Object.values(groupRows(rows, row.audience, search)).flat(),
    ).toHaveLength(1)
  expect(groupRows(rows, row.audience, 'absent text')).toEqual({})
})
test('wire allowed constraints drive the shared registry helper', () => {
  const converted = registryRow({
    ...row,
    allowed: {
      kind: 'object',
      properties: {
        type: { kind: 'string', text: 'number' },
        minimum: { kind: 'number', text: '0.25' },
      },
    },
  })
  const field = fieldFromRow(converted)
  expect(field.control).toBe('number')
  if (field.control === 'unsupported') throw field.error
  expect(field.schema.safeParse(0.5).success).toBe(true)
  expect(field.schema.safeParse(0.1).success).toBe(false)
})
test('unknown effective values remain unknown and decimal values are never rounded', () => {
  expect(
    effectiveValue(
      { ...view, switch_overrides: [], settings: [], effective_switches: [] },
      row.name,
    ),
  ).toBeUndefined()
  const supplied: View = {
    ...view,
    settings: [],
    switch_overrides: [],
    effective_switches: [
      {
        name: row.name,
        effective: { kind: 'number', text: '0.5' },
        source: 'fixture',
        audience: row.audience,
        group: row.group,
      },
    ],
  }
  expect(effectiveValue(supplied, row.name)).toBe(0.5)
})
test('unsafe integers stay exact recursively and unknown wire kinds remain unsupported', () => {
  expect(decodeValue({ kind: 'number', text: '18446744073709551615' })).toBe(
    '18446744073709551615',
  )
  expect(
    decodeValue({
      kind: 'object',
      properties: {
        suggested: {
          kind: 'object',
          properties: {
            maximum: { kind: 'number', text: '18446744073709551615' },
          },
        },
      },
    }),
  ).toEqual({ suggested: { maximum: '18446744073709551615' } })
  expect(() => decodeValue({ kind: 'future' })).toThrow(/Unsupported/)
})

test('default and nested suggested integer bounds stay exact on a converted registry row', () => {
  const exact = '18446744073709551615'
  const converted = registryRow({
    ...row,
    values: { default: { kind: 'number', text: exact } },
    allowed: {
      kind: 'object',
      properties: {
        type: { kind: 'string', text: 'integer' },
        maximum: { kind: 'number', text: exact },
        suggested: {
          kind: 'object',
          properties: { maximum: { kind: 'number', text: exact } },
        },
      },
    },
  })
  expect(converted.defaultValue).toBe(exact)
  expect(converted.allowed.maximum).toBe(exact)
  expect(converted.allowed.suggested).toEqual({ maximum: exact })
})

test('actual integer controls retain the CLI lower bounds without invented maxima', () => {
  for (const name of ['--context', '--sessions']) {
    const row = registryRow(view.switches.find((row) => row.name === name)!)
    expect(row.allowed).toEqual({
      type: 'integer',
      minimum: name === '--context' ? 0 : 1,
    })
    const field = fieldFromRow(row)
    if (field.control === 'unsupported') throw field.error
    expect(field.schema.safeParse(7).success).toBe(true)
    expect(field.schema.safeParse(Number.MAX_SAFE_INTEGER + 1).success).toBe(
      false,
    )
  }
})
test('labels fall back only when missing and debug rows retain CLI descriptions', () => {
  expect(
    fieldFromRow(registryRow({ ...row, label: 'Supplied label' })).label,
  ).toBe('Supplied label')
  expect(fieldFromRow(registryRow({ ...row, label: undefined })).label).toBe(
    row.name,
  )
  const debug = Object.values(groupRows(view.switches, 'debug')).flat()
  expect(debug.every((item) => item.name && item.purpose)).toBe(true)
})

// The release exe reports an unset text option's effective value as its
// default text ("Off" for --cache-disk, source "default"). Ways this could fail:
//   R1 "Off" is shown as the value of a folder field, so the user edits around it;
//   R2 a text value the user really set is hidden because it equals the default text;
//   R3 an enum or number whose default text is a real value ("Off" as an enum
//      entry, a built-in byte count) turns into "Not set".
test('a text option whose effective value is only its default text is unset', () => {
  const real = scenarioByName('ready-idle').view
  const disk = real.switches.find((item) => item.name === '--cache-disk')!
  const reported = (real.effective_switches ?? []).find(
    (item) => item.name === '--cache-disk',
  )
  expect(disk.default_text).toBe('Off')
  expect(reported && decodeValue(reported.effective)).toBe('Off')
  expect(effectiveValue(real, '--cache-disk')).toBeUndefined() // R1
  const bytes = (real.effective_switches ?? []).find(
    (item) => item.name === '--cache-disk-bytes',
  )
  expect(bytes && decodeValue(bytes.effective)).not.toBeUndefined()
  expect(effectiveValue(real, '--cache-disk-bytes')).not.toBeUndefined() // R3
  const set = structuredClone(real)
  const setting = (set.settings ?? []).find(
    (item) => item.name === '--cache-disk',
  )
  if (setting) setting.requested = { kind: 'string', text: 'Off' } as never
  else
    set.switch_overrides = [
      ...(set.switch_overrides ?? []),
      { name: '--cache-disk', saved: 'Off' } as never,
    ]
  expect(effectiveValue(set, '--cache-disk')).toBe('Off') // R2
})

test('a CLI numeric scalar in a text control stays editable when it equals the default text', () => {
  const real = scenarioByName('ready-idle').view
  const draft = real.switches.find((row) => row.name === '--draft-tokens')!
  expect(effectiveValue(real, draft.name)).toBe(draft.default_text)
})
