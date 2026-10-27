import type { View } from '@/lib/bindings'
import { rate } from '@/lib/format'

type Slot = View['slots'][number]
const count = (value: number | undefined): value is number =>
  value != null && Number.isSafeInteger(value) && value >= 0

export function activeSessions(view: View) {
  return view.serving
    ? view.slots.filter(
        (slot) =>
          slot.processing || ['prefilling', 'generating'].includes(slot.state),
      )
    : []
}

export function promptRead(slot: Slot) {
  const total = slot.prompt_tokens
  const cached = slot.cached_prompt_tokens
  const prefilled = slot.prefilled_tokens
  if (
    !count(total) ||
    !count(cached) ||
    !count(prefilled) ||
    total === 0 ||
    cached > total ||
    prefilled > total - cached
  )
    return undefined
  const read = cached + prefilled
  return { total, cached, read, percent: Math.floor((read / total) * 100) }
}

export function liveRate(value: number | undefined) {
  return value != null && Number.isFinite(value) && value >= 0
    ? rate(value)
    : '-'
}

export function liveSessionSummary(view: View) {
  if (view.stale) return 'Waiting for session status'
  const slots = activeSessions(view)
  const slot = slots.find((entry) => entry.state === 'generating') ?? slots[0]
  if (!slot) return view.slots.length ? 'Idle' : 'Session status unavailable'
  const progress = promptRead(slot)
  const summary =
    slot.state === 'generating'
      ? `Writing · ${liveRate(slot.decode_tps)} tok/s`
      : slot.state === 'prefilling'
        ? `Reading prompt · ${progress ? `${progress.percent}%` : '-'}`
        : 'Session status unavailable'
  return slots.length > 1
    ? `Session ${slot.id + 1} · ${summary} · ${slots.length} active sessions`
    : summary
}
