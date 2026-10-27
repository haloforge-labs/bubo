import type { View } from '@/lib/bindings'
import { number } from '@/lib/format'
import { activeSessions, liveRate, promptRead } from '@/lib/live-sessions'
import { LiveRateChart } from './live-rate-chart'

type Slot = View['slots'][number]

function PromptRead({ slot }: { slot: Slot }) {
  const progress = promptRead(slot)
  if (!progress)
    return <span className="text-muted-foreground">Progress unavailable</span>
  const { total, cached, read, percent } = progress
  return (
    <div className="min-w-56 space-y-1">
      <div className="flex items-baseline justify-between gap-4 font-mono text-sm tabular-nums">
        <span>
          {number(read, 0)} / {number(total, 0)}
        </span>
        <span>{percent}%</span>
      </div>
      <div
        role="progressbar"
        aria-label={`Session ${slot.id + 1} prompt read`}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={read}
        aria-valuetext={`${number(read, 0)} of ${number(total, 0)} tokens read; ${number(cached, 0)} cached`}
        className="h-1.5 overflow-hidden rounded-full bg-muted"
      >
        <div
          className="h-full bg-primary"
          style={{ width: `${(read / total) * 100}%` }}
        />
      </div>
      {cached > 0 && (
        <p className="text-xs text-muted-foreground">
          {number(cached, 0)} cached
        </p>
      )}
    </div>
  )
}

export function PromptProgress({ view }: { view: View }) {
  const slots = activeSessions(view)
  const samples = view.serving ? (view.live_samples ?? []) : []
  const groups = new Map<
    string,
    { session: number; active: boolean; samples: typeof samples }
  >()
  for (const slot of slots) {
    const key = `${slot.id}:${slot.request_id ?? 'unknown'}`
    groups.set(key, { session: slot.id + 1, active: true, samples: [] })
  }
  for (const sample of samples) {
    const key = `${sample.slot_id}:${sample.request_id}`
    if (!groups.has(key))
      groups.set(key, {
        session: sample.slot_id + 1,
        active: false,
        samples: [],
      })
    groups.get(key)!.samples.push(sample)
  }
  if (!slots.length && !groups.size) return null
  return (
    <section aria-label="Live requests" className="space-y-3">
      <h2 className="text-lg">Live requests</h2>
      <p className="text-xs text-muted-foreground">
        Counts and rates update with server samples. Cached tokens are already
        read. Elapsed time starts when Bubo first observes the active request.
      </p>
      {!!slots.length && (
        <div className="overflow-x-auto">
          <div
            role="table"
            aria-label="Active sessions"
            className="min-w-[36rem] text-sm"
          >
            <div
              role="row"
              className="grid grid-cols-[4rem_5rem_minmax(14rem,1fr)_7rem_8rem] gap-3 border-b pb-2 text-muted-foreground"
            >
              <span role="columnheader">Session</span>
              <span role="columnheader">Phase</span>
              <span role="columnheader">Progress</span>
              <span role="columnheader">Live tok/s</span>
              <span role="columnheader">Observed elapsed</span>
            </div>
            {slots.map((slot) => (
              <div
                role="row"
                key={slot.id}
                className="grid grid-cols-[4rem_5rem_minmax(14rem,1fr)_7rem_8rem] items-start gap-3 border-b py-3"
              >
                <span role="cell" className="font-mono tabular-nums">
                  {number(slot.id + 1, 0)}
                </span>
                <span role="cell">
                  {slot.state === 'prefilling'
                    ? 'Reading'
                    : slot.state === 'generating'
                      ? 'Writing'
                      : 'Unknown'}
                </span>
                <div role="cell">
                  {slot.state === 'prefilling' ? (
                    <div>
                      <span className="sr-only">Prompt read</span>
                      <PromptRead slot={slot} />
                    </div>
                  ) : slot.state === 'generating' ? (
                    <span className="font-mono tabular-nums">
                      {slot.generated_tokens == null
                        ? '-'
                        : number(slot.generated_tokens, 0)}{' '}
                      tokens written
                    </span>
                  ) : (
                    'Progress unavailable'
                  )}
                </div>
                <span role="cell" className="font-mono tabular-nums">
                  {liveRate(
                    slot.state === 'prefilling'
                      ? slot.prefill_tps
                      : slot.state === 'generating'
                        ? slot.decode_tps
                        : undefined,
                  )}
                </span>
                <span role="cell" className="font-mono tabular-nums">
                  {slot.elapsed_ms == null
                    ? '-'
                    : `${number(slot.elapsed_ms / 1000, 1)} s`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        {[...groups].map(([key, group]) => (
          <LiveRateChart
            key={key}
            {...group}
            sampleTime={view.live_sample_time ?? ''}
          />
        ))}
      </div>
    </section>
  )
}
