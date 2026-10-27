import { z } from 'zod'
import { schemaFromAllowed, UnsupportedRegistryError } from '@/lib/registry'
import type { View } from '@/lib/bindings'

type Setting = Pick<View['settings'][number], 'name' | 'allowed'>
type Table = NonNullable<
  NonNullable<View['memory_estimate']>['recommended_context']
>

export function numericSetting(view: View, name: string) {
  const observed = view.settings?.find((row) => row.name === name)
  const row = observed ?? view.switches?.find((row) => row.name === name)
  const value = observed?.effective
  const number =
    value?.kind === 'number' || value?.kind === 'string'
      ? Number(value.text)
      : undefined
  // The registry's own default, for a server that reports no value yet.
  const text = view.switches?.find((row) => row.name === name)?.default_text
  const fallback = /^\d[\d,]*$/.test(text ?? '')
    ? Number((text ?? '').replace(/,/g, ''))
    : undefined
  return {
    row,
    value: Number.isFinite(number) ? number : undefined,
    default: Number.isFinite(fallback) ? fallback : undefined,
  }
}

export function numberConstraint(row?: Setting) {
  const allowed = row?.allowed?.properties
  const type = allowed?.type?.text
  if (type !== 'integer' && type !== 'number')
    throw new UnsupportedRegistryError(
      `${row?.name ?? 'setting'} has no numeric allowed contract`,
    )
  const number = (key: string) => {
    const value = allowed?.[key]
    if (!value || value.kind === 'null') return undefined
    const result = Number(value.text)
    if (!Number.isFinite(result))
      throw new UnsupportedRegistryError(`${row?.name}: ${key}`)
    return result
  }
  return {
    type,
    minimum: number('minimum'),
    maximum: number('maximum'),
    exclusive_minimum: number('exclusive_minimum'),
  } as const
}

export function serverFormSchema(view: View, currentModelTable?: Table | null) {
  const context = numberConstraint(numericSetting(view, '--context').row)
  const sessions = numberConstraint(numericSetting(view, '--sessions').row)
  const table =
    currentModelTable === undefined
      ? (view.saved_estimate
          ? view.saved_estimate.memory_estimate
          : view.memory_estimate
        )?.recommended_context
      : currentModelTable
  return z.object({
    context: schemaFromAllowed(context)
      .pipe(z.number())
      .refine(
        (value) =>
          !table ||
          (value >= table.minimum_context_tokens &&
            value <= table.context_max_tokens),
        'Outside the supported context range.',
      ),
    sessions: schemaFromAllowed(sessions)
      .pipe(z.number())
      .refine(
        (value) => !table || value <= table.max_sessions,
        'Outside the supported sessions range.',
      ),
    model: z.string(),
  })
}

// Presets are looked up, never derived from a memory formula or another session count.
export function contextPreset(table: Table | undefined, sessions: number) {
  return table?.by_sessions?.find((row) => row.sessions === sessions)
}

export function contextFits(
  table: Table | undefined,
  sessions: number,
  context: number,
) {
  const row = contextPreset(table, sessions)
  return row ? row.fits && context <= row.context_per_session : undefined
}
