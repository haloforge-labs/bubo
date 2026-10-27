import type { View } from '@/lib/bindings'

export function recommendedModel(view: View) {
  return view.models?.catalogue.find(
    (entry) =>
      entry.role === 'recommended' && entry.default_status === 'default',
  )
}

export function number(value: number, maximumFractionDigits = 1) {
  return value.toLocaleString(undefined, { maximumFractionDigits })
}

export function rate(value: number) {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
}

export function percent(value: number) {
  return `${number(value, 0)} %`
}

export function milliseconds(value: number) {
  return number(value, 0)
}

export function bytes(
  value: number,
  options: { unit?: 'B' | 'KiB' | 'MiB' | 'GiB'; digits?: number } = {},
) {
  const unit =
    options.unit ??
    (value < 1024
      ? 'B'
      : value < 1024 ** 2
        ? 'KiB'
        : value < 1024 ** 3
          ? 'MiB'
          : 'GiB')
  const power = { B: 0, KiB: 1, MiB: 2, GiB: 3 }[unit]
  const digits = options.digits ?? (unit === 'B' ? 0 : 1)
  return `${(value / 1024 ** power).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: options.digits ?? 1,
  })} ${unit}`
}

export function fileName(path: string) {
  return path.split(/[\\/]/).filter(Boolean).at(-1) || path
}

export function folderPath(path: string) {
  const end = Math.max(path.lastIndexOf('\\'), path.lastIndexOf('/'))
  return end < 0 ? '' : path.slice(0, end + 1)
}

export function activeModelName(view: View) {
  return view.current_model?.display_name || undefined
}

export function duration(seconds: number) {
  return seconds >= 60 && seconds % 60 === 0
    ? `${number(seconds / 60, 0)} min`
    : `${number(seconds, 0)} s`
}

// The connection endpoint is already the CLI's complete OpenAI URL.
export function serverAddresses(view: View): NonNullable<View['endpoints']> {
  if (!view.serving) return { openai_base_url: '', anthropic_base_url: '' }
  return {
    anthropic_base_url: view.endpoints?.anthropic_base_url || '',
    openai_base_url:
      view.endpoints?.openai_base_url || view.connection?.endpoint || '',
  }
}

// These are presentation states; no updater transition or successful check is inferred.
export function updatePresentation(update: View['update']) {
  if (update.error?.code === 'check_required') return 'not_checked'
  return update.state
}
