import type { View } from '@/lib/bindings'
import type { RegistryRow, RegistryValue } from '@/lib/registry'
import { parseAllowed, UnsupportedRegistryError } from '@/lib/registry'
import { markerListUnset } from '@/lib/server-values'

// Tab wording only: the CLI still owns row membership and write routing.
export function groupLabel(group: string) {
  return group === 'Behaviour' ? 'Inference' : group
}

type Value = View['switches'][number]['allowed']
type JSONValue =
  string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue }

// Decode the supervisor's tagged JSON transport; constraints still come only
// from the CLI and are interpreted by the shared registry helper.
export function decodeValue(value: Value): JSONValue {
  switch (value.kind) {
    case 'null':
      return null
    case 'string':
      return value.text ?? ''
    case 'boolean':
      return value.boolean ?? false
    case 'number': {
      const number = Number(value.text)
      if (
        value.text &&
        /^-?\d+$/.test(value.text) &&
        !Number.isSafeInteger(number)
      )
        return value.text
      if (
        !Number.isFinite(number) ||
        (Number.isInteger(number) && !Number.isSafeInteger(number))
      )
        throw new UnsupportedRegistryError(
          'number cannot be represented exactly by this control',
        )
      return number
    }
    case 'array':
      return (value.items ?? []).map(decodeValue)
    case 'object':
      return Object.fromEntries(
        Object.entries(value.properties ?? {}).map(([key, item]) => [
          key,
          decodeValue(item),
        ]),
      )
    default:
      throw new UnsupportedRegistryError(value.kind)
  }
}

export function registryRow(row: View['switches'][number]): RegistryRow {
  const properties = row.allowed.properties ?? {}
  const bounds = new Set(['minimum', 'maximum', 'exclusive_minimum'])
  const allowed = Object.fromEntries(
    Object.entries(properties).map(([key, value]) => {
      if (
        bounds.has(key) &&
        value.kind === 'number' &&
        value.text &&
        /^-?\d+$/.test(value.text) &&
        !Number.isSafeInteger(Number(value.text))
      )
        return [key, value.text]
      return [key, decodeValue(value)]
    }),
  )
  // The supervisor validates the CLI schema before publishing this View.
  return {
    ...row,
    defaultValue: registryDefault(row),
    audience: row.audience as RegistryRow['audience'],
    group: row.group as RegistryRow['group'],
    allowed: parseAllowed(allowed),
    // Use only meanings explicitly attached to a scalar by the CLI row.
    valueLabels: Object.fromEntries(
      [...row.default_text.matchAll(/(-?\d+(?:\.\d+)?)\s*\(([^)]+)\)/g)]
        .filter((match) =>
          /^(?:random|off|no limit|unlimited|automatic)$/i.test(match[2]!),
        )
        .map((match) => [String(Number(match[1])), match[2]!]),
    ),
  }
}

// The engine reports some values as 32-bit floats (0.949999988079071 for
// 0.95). Seven significant digits is the precision of that type.
function tidy(value: number) {
  return Number.isInteger(value) ? value : Number(value.toPrecision(7))
}

export function effectiveValue(
  view: View,
  name: string,
): RegistryValue | undefined {
  const override = view.switch_overrides?.find((row) => row.name === name)
  const row = view.switches.find((row) => row.name === name)
  function controlValue(value: RegistryValue): RegistryValue {
    if (typeof value === 'number') value = tidy(value)
    if (!row) return value
    const allowed = registryRow(row).allowed
    // CLI effective scalars may use a different JSON type than the registry's
    // input representation (for example a numeric effective value in a text control).
    if (allowed.type === 'string') return String(value)
    if (
      (allowed.type === 'number' || allowed.type === 'integer') &&
      typeof value === 'string'
    ) {
      const decoded = decodeValue({ kind: 'number', text: value })
      return typeof decoded === 'number' ? tidy(decoded) : String(decoded)
    }
    if (allowed.type === 'boolean' && typeof value === 'string') {
      if (allowed.values?.length)
        return allowed.values.find((entry) => String(entry) === value) ?? value
      if (value === 'true') return true
      if (value === 'false') return false
    }
    if (allowed.type === 'enum')
      return (
        allowed.values?.find((entry) => String(entry) === String(value)) ??
        value
      )
    return value
  }
  if (override?.saved != null) {
    if (markerListUnset(name, override.saved)) return undefined
    return controlValue(override.saved)
  }
  const setting = (view.settings ?? []).find((row) => row.name === name)
  const effective = (view.effective_switches ?? []).find(
    (row) => row.name === name,
  )
  const requested =
    setting?.requested.kind !== 'null' && setting?.requested
      ? setting.requested
      : undefined
  const value =
    override?.effective.kind !== 'null' && override?.effective
      ? override.effective
      : (requested ?? setting?.effective ?? effective?.effective)
  if (!value) return undefined
  const decoded = decodeValue(value)
  if (markerListUnset(name, decoded)) return undefined
  if (
    !requested &&
    row &&
    typeof decoded === 'string' &&
    decoded === row.default_text &&
    ['integer', 'number'].includes(registryRow(row).allowed.type) &&
    !Number.isFinite(Number(decoded))
  )
    return undefined
  // The CLI reports an unset text option as its default text ("Off" for the
  // disk cache folder). That is a label, not a value the user can edit.
  if (
    !requested &&
    row &&
    registryRow(row).allowed.type === 'string' &&
    !(typeof decoded === 'string' && /^-?\d+(?:\.\d+)?$/.test(decoded)) &&
    decoded === row.default_text
  )
    return undefined
  return typeof decoded === 'string' ||
    typeof decoded === 'number' ||
    typeof decoded === 'boolean'
    ? controlValue(decoded)
    : undefined
}

export function registryDefault(
  row: View['switches'][number],
): RegistryValue | undefined {
  const value = row.values.default && decodeValue(row.values.default)
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  )
    return value
  const allowed = parseAllowed(decodeValue(row.allowed))
  if (
    ['integer', 'number', 'string'].includes(allowed.type) &&
    /^-?\d+(?:\.\d+)?$/.test(row.default_text)
  )
    return allowed.type === 'string'
      ? row.default_text
      : (decodeValue({
          kind: 'number',
          text: row.default_text,
        }) as RegistryValue)
  return allowed.type === 'enum'
    ? allowed.values?.find((value) => String(value) === row.default_text)
    : undefined
}

export function groupRows(
  rows: View['switches'],
  audience: string,
  filter = '',
) {
  const query = filter.toLocaleLowerCase()
  // Object.groupBy is outside this project's configured TypeScript target.
  return (rows ?? [])
    .filter(
      (row) =>
        row.gui_control &&
        row.audience === audience &&
        `${row.label ?? row.name} ${row.name} ${row.purpose} ${row.note ?? ''}`
          .toLocaleLowerCase()
          .includes(query),
    )
    .reduce<Record<string, View['switches']>>((groups, row) => {
      ;(groups[row.group] ??= []).push(row)
      return groups
    }, {})
}
