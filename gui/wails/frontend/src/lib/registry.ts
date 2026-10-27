import { z } from 'zod'
import type { SwitchRow } from '@/generated/switches'

export type { SwitchRow }
export type RegistryRow = Pick<
  SwitchRow,
  'name' | 'label' | 'purpose' | 'gui_control' | 'audience' | 'group'
> & {
  allowed: Allowed
  valueLabels?: Record<string, string>
  defaultValue?: RegistryValue
}
export type RegistryValue = string | number | boolean
export type RegistryValues = Record<string, RegistryValue>
// Keep exact integer bounds through the Go tagged-value transport. JSON numbers
// above JavaScript's safe integer range must never be rounded into another limit.
type Allowed = Pick<
  SwitchRow['allowed'],
  'type' | 'values' | 'also' | 'format'
> & {
  suggested?: SwitchRow['allowed']['suggested']
  minimum?: number | string
  maximum?: number | string
  exclusive_minimum?: number | string
}
const boundSchema = z.union([z.number(), z.string().regex(/^-?\d+$/)])
const allowedSchema = z.object({
  type: z.enum(['boolean', 'enum', 'integer', 'number', 'string']),
  values: z.array(z.union([z.string(), z.number(), z.boolean()])).optional(),
  also: z.array(z.number()).optional(),
  format: z.string().optional(),
  suggested: z.unknown().optional(),
  minimum: boundSchema.optional(),
  maximum: boundSchema.optional(),
  exclusive_minimum: boundSchema.optional(),
})

export function parseAllowed(value: unknown): Allowed {
  const parsed = allowedSchema.safeParse(value)
  if (!parsed.success)
    throw new UnsupportedRegistryError('invalid allowed contract')
  return parsed.data
}

function compareBound(value: number, bound: string) {
  if (Number.isInteger(value)) {
    if (!Number.isSafeInteger(value)) return undefined
    const current = BigInt(value),
      limit = BigInt(bound)
    return current < limit ? -1 : current > limit ? 1 : 0
  }
  // Fractional representable values have smaller magnitude than an unsafe
  // integer bound; numeric comparison cannot cross that exact bound.
  return value < Number(bound) ? -1 : value > Number(bound) ? 1 : 0
}
const suggestionSchema = z.object({
  type: z.string().optional(),
  values: z.array(z.union([z.string(), z.number(), z.boolean()])).optional(),
  minimum: boundSchema.optional(),
  maximum: boundSchema.optional(),
})

function suggestion(allowed: Allowed) {
  const parsed = suggestionSchema.safeParse(allowed.suggested)
  if (!parsed.success) return undefined
  const { values, minimum, maximum } = parsed.data
  if (values?.length) return values.map(String).join(', ')
  return [
    minimum === undefined ? '' : `from ${minimum}`,
    maximum === undefined ? '' : `up to ${maximum}`,
  ]
    .filter(Boolean)
    .join(' ')
}

export class UnsupportedRegistryError extends Error {
  readonly code = 'unsupported'

  constructor(detail: string) {
    super(`Unsupported registry constraint: ${detail}`)
    this.name = 'UnsupportedRegistryError'
  }
}

// Every validation rule comes from allowed; no switch names, defaults or limits live here.
export function schemaFromAllowed(allowed: Allowed): z.ZodType<RegistryValue> {
  let schema: z.ZodType<RegistryValue>
  switch (allowed.type) {
    case 'boolean':
      if (allowed.format) throw new UnsupportedRegistryError(allowed.format)
      schema = allowed.values?.length
        ? z.union(allowed.values.map((value) => z.literal(value)))
        : z.boolean()
      break
    case 'enum': {
      if (allowed.format) throw new UnsupportedRegistryError(allowed.format)
      const values = allowed.values
      if (!values?.length) throw new UnsupportedRegistryError('empty enum')
      schema = z.union(values.map((value) => z.literal(value)))
      break
    }
    case 'integer':
    case 'number': {
      if (allowed.format && allowed.format !== 'finite')
        throw new UnsupportedRegistryError(allowed.format)
      let number = z.number()
      if (allowed.type === 'integer') number = number.int()
      if (allowed.minimum !== undefined) {
        const limit = allowed.minimum
        number =
          typeof limit === 'string'
            ? number.refine((value) => {
                const compared = compareBound(value, limit)
                return compared !== undefined && compared >= 0
              }, `Must be at least ${limit}`)
            : number.min(limit)
      }
      if (allowed.maximum !== undefined) {
        const limit = allowed.maximum
        number =
          typeof limit === 'string'
            ? number.refine((value) => {
                const compared = compareBound(value, limit)
                return compared !== undefined && compared <= 0
              }, `Must be at most ${limit}`)
            : number.max(limit)
      }
      if (allowed.exclusive_minimum !== undefined) {
        const limit = allowed.exclusive_minimum
        number =
          typeof limit === 'string'
            ? number.refine(
                (value) => compareBound(value, limit) === 1,
                `Must be greater than ${limit}`,
              )
            : number.gt(limit)
      }
      schema =
        allowed.type === 'integer'
          ? z.union([
              number,
              z
                .string()
                .regex(/^-?\d+$/)
                .refine((value) => {
                  if (!/^-?\d+$/.test(value)) return false
                  const current = BigInt(value)
                  const minimum = allowed.minimum
                  const maximum = allowed.maximum
                  const exclusive = allowed.exclusive_minimum
                  return (
                    (minimum === undefined ||
                      (Number.isInteger(Number(minimum)) &&
                        current >= BigInt(minimum))) &&
                    (maximum === undefined ||
                      (Number.isInteger(Number(maximum)) &&
                        current <= BigInt(maximum))) &&
                    (exclusive === undefined ||
                      (Number.isInteger(Number(exclusive)) &&
                        current > BigInt(exclusive)))
                  )
                }, 'Outside the supported integer range.'),
            ])
          : number
      break
    }
    case 'string':
      switch (allowed.format) {
        case undefined:
          schema = z.string()
          break
        case 'email':
          schema = z.email()
          break
        case 'uri':
          schema = z.url()
          break
        case 'uuid':
          schema = z.uuid()
          break
        case 'date-time':
          schema = z.iso.datetime()
          break
        case 'effort:bias comma list':
          schema = z
            .string()
            .refine(
              (value) =>
                value === '' ||
                (new Set(
                  value.split(',').map((part) => part.trim().split(':')[0]),
                ).size === value.split(',').length &&
                  value
                    .split(',')
                    .every(
                      (part) =>
                        /^[^:,\s]+\s*:\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(
                          part.trim(),
                        ) && Number.isFinite(Number(part.split(':')[1])),
                    )),
              'Enter one finite bias per effort level.',
            )
          break
        default:
          throw new UnsupportedRegistryError(allowed.format)
      }
      break
  }
  if (allowed.also?.length) {
    schema = z.union([schema, ...allowed.also.map((value) => z.literal(value))])
  }
  return schema
}

export function fieldFromRow(row: RegistryRow) {
  const suggested = suggestionSchema.safeParse(row.allowed.suggested)
  const minimum =
    typeof row.allowed.minimum === 'number'
      ? row.allowed.minimum
      : typeof suggested.data?.minimum === 'number'
        ? suggested.data.minimum
        : undefined
  const maximum =
    typeof row.allowed.maximum === 'number'
      ? row.allowed.maximum
      : typeof suggested.data?.maximum === 'number'
        ? suggested.data.maximum
        : undefined
  const common = {
    name: row.name,
    label: row.label ?? row.name,
    description: row.purpose,
    audience: row.audience,
    group: row.group,
    suggestion: suggestion(row.allowed),
    range:
      minimum != null && maximum != null && minimum < maximum
        ? { minimum, maximum }
        : undefined,
    valueLabels: row.valueLabels,
    defaultValue: row.defaultValue,
    suggestedMinimum:
      typeof suggested.data?.minimum === 'number'
        ? suggested.data.minimum
        : undefined,
    suggestedMaximum:
      typeof suggested.data?.maximum === 'number'
        ? suggested.data.maximum
        : undefined,
    // CLI boolean rows may declare string wire values for environment switches.
    // Only a declared boolean type gets a switch; ordinary enums retain a select.
    booleanValues:
      row.allowed.type === 'boolean'
        ? row.allowed.values?.length
          ? [
              row.allowed.values.find(
                (value) => value === false || value === '0' || value === 0,
              ),
              row.allowed.values.find(
                (value) => value === true || value === '1' || value === 1,
              ),
            ]
          : [false, true]
        : undefined,
  }
  try {
    const schema = schemaFromAllowed(row.allowed)
    const control = common.booleanValues?.every((value) => value !== undefined)
      ? 'switch'
      : row.allowed.type === 'enum' ||
          (row.allowed.type === 'boolean' && !!row.allowed.values?.length)
        ? 'select'
        : row.allowed.type === 'integer' || row.allowed.type === 'number'
          ? 'number'
          : 'text'
    return { ...common, control, schema, allowed: row.allowed } as const
  } catch (error) {
    if (!(error instanceof UnsupportedRegistryError)) throw error
    return { ...common, control: 'unsupported', error } as const
  }
}

export function fieldsFromRegistry(rows: readonly RegistryRow[]) {
  return rows.filter((row) => row.gui_control).map(fieldFromRow)
}

export function schemaFromRegistry(rows: readonly RegistryRow[]) {
  return z.object(
    Object.fromEntries(
      rows
        .filter((row) => row.gui_control)
        .map((row) => [row.name, schemaFromAllowed(row.allowed)]),
    ),
  )
}
