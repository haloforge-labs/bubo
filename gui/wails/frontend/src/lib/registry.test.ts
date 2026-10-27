import { expect, test } from 'vitest'
import { dummyRow } from '../../tests/dummy-registry'
import {
  fieldFromRow,
  fieldsFromRegistry,
  schemaFromAllowed,
  schemaFromRegistry,
  UnsupportedRegistryError,
} from '@/lib/registry'

test('exact integer text validates without converting large CLI bounds or values to floats', () => {
  const schema = schemaFromAllowed({
    type: 'integer',
    minimum: '9007199254740993',
    maximum: '18446744073709551615',
  })
  expect(schema.parse('9007199254740995')).toBe('9007199254740995')
  expect(schema.safeParse('9007199254740992').success).toBe(false)
  expect(schema.safeParse('18446744073709551616').success).toBe(false)
  expect(schema.safeParse('not-an-integer').success).toBe(false)
})

test('effort-map validation accepts only unique levels with finite biases', () => {
  const schema = schemaFromAllowed({
    type: 'string',
    format: 'effort:bias comma list',
  })
  expect(schema.safeParse('quick:-3,careful:0').success).toBe(true)
  for (const value of [
    'quick:1e999',
    'quick:-3,quick:0',
    'quick:',
    'quick:none',
  ])
    expect(schema.safeParse(value).success).toBe(false)
})

test('field labels use the CLI label when present and retain the name when absent', () => {
  expect(fieldFromRow({ ...dummyRow, label: 'Fixture label' }).label).toBe(
    'Fixture label',
  )
  expect(fieldFromRow(dummyRow).label).toBe(dummyRow.name)
})

test.each([
  [{ type: 'boolean' as const }, true, 'true'],
  [{ type: 'enum' as const, values: ['small', 'large'] }, 'small', 'medium'],
  [{ type: 'integer' as const, minimum: 3, maximum: 17 }, 9, 9.5],
  [{ type: 'number' as const, minimum: 0.25, maximum: 0.75 }, 0.5, 1],
  [{ type: 'string' as const }, 'fixture', false],
])('schema from allowed validates %j', (allowed, valid, invalid) => {
  const schema = schemaFromAllowed(allowed)
  expect(schema.safeParse(valid).success).toBe(true)
  expect(schema.safeParse(invalid).success).toBe(false)
})

test('numeric bounds, exclusive minimum, special allowed numbers and finite format follow the registry', () => {
  const schema = schemaFromAllowed({
    type: 'number',
    minimum: 2,
    maximum: 7,
    exclusive_minimum: 2,
    also: [0],
    format: 'finite',
  })
  expect([0, 3, 7].every((value) => schema.safeParse(value).success)).toBe(true)
  expect(
    [1, 2, 8, NaN, Infinity].every((value) => !schema.safeParse(value).success),
  ).toBe(true)
})

test.each([
  ['email', 'person@example.test', 'invalid'],
  ['uri', 'https://example.test/path', 'invalid'],
  ['uuid', '937234a1-0b01-40b1-8714-3f663eb307f2', 'invalid'],
  ['date-time', '2026-01-01T00:00:00Z', 'invalid'],
])(
  'string format %s uses established Zod validation',
  (format, valid, invalid) => {
    const schema = schemaFromAllowed({ type: 'string', format })
    expect(schema.safeParse(valid).success).toBe(true)
    expect(schema.safeParse(invalid).success).toBe(false)
  },
)

test('unknown formats and empty enums produce typed unsupported results', () => {
  expect(() =>
    schemaFromAllowed({ type: 'string', format: 'fixture syntax' }),
  ).toThrow(UnsupportedRegistryError)
  expect(() => schemaFromAllowed({ type: 'enum', values: [] })).toThrow(
    UnsupportedRegistryError,
  )
  expect(
    fieldFromRow({
      ...dummyRow,
      allowed: { type: 'string', format: 'fixture syntax' },
    }).control,
  ).toBe('unsupported')
})

test('an added registry row determines control, limits and schema without implementation changes', () => {
  const fields = fieldsFromRegistry([dummyRow])
  expect(fields).toHaveLength(1)
  const field = fields[0]
  expect(field?.control).toBe('number')
  if (!field || field.control === 'unsupported')
    throw new Error('Fixture should have a supported field')
  expect(field.allowed).toMatchObject({ minimum: 3, maximum: 17 })
  expect(
    schemaFromRegistry([dummyRow]).safeParse({ [dummyRow.name]: 12 }).success,
  ).toBe(true)
  expect(
    schemaFromRegistry([dummyRow]).safeParse({ [dummyRow.name]: 21 }).success,
  ).toBe(false)
  expect(fieldsFromRegistry([{ ...dummyRow, gui_control: false }])).toEqual([])
})

test('mixed enum values retain their original types', () => {
  const schema = schemaFromAllowed({
    type: 'enum',
    values: [true, 3, 'choice'],
  })
  expect(
    [true, 3, 'choice'].every((value) => schema.safeParse(value).success),
  ).toBe(true)
  expect(schema.safeParse('3').success).toBe(false)
})

test('suggestions never narrow validation or numeric input limits', () => {
  for (const allowed of [
    {
      type: 'number' as const,
      format: 'finite',
      suggested: { minimum: -2, maximum: 0 },
    },
    {
      type: 'string' as const,
      suggested: { type: 'integer', minimum: 1, maximum: 8 },
    },
    {
      type: 'string' as const,
      suggested: { type: 'enum', values: ['bounded', 'off'] },
    },
  ]) {
    const field = fieldFromRow({ ...dummyRow, allowed })
    expect(field.suggestion).toBeTruthy()
    if (field.control === 'unsupported') throw field.error
    expect(
      field.schema.safeParse(
        allowed.type === 'number' ? 5 : 'outside-suggestion',
      ).success,
    ).toBe(true)
    expect(field.allowed.maximum).toBeUndefined()
  }
})
