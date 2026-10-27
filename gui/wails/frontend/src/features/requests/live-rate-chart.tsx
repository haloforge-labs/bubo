import {
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { View } from '@/lib/bindings'
import { number } from '@/lib/format'
import { liveRate } from '@/lib/live-sessions'
import { yTicks } from '@/components/line-chart'

type Sample = View['live_samples'][number]

export function LiveRateChart({
  session,
  samples,
  sampleTime,
  active,
}: {
  session: number
  samples: readonly Sample[]
  sampleTime: string
  active: boolean
}) {
  const timestamps = samples
    .map((sample) => Date.parse(sample.sample_time))
    .filter(Number.isFinite)
  const observed = Date.parse(sampleTime)
  const anchor = Math.max(
    Number.isFinite(observed) ? observed : -Infinity,
    ...timestamps,
  )
  const points = samples
    .flatMap((sample) => {
      const secondsAgo = (anchor - Date.parse(sample.sample_time)) / 1000
      if (!Number.isFinite(secondsAgo) || secondsAgo < 0 || secondsAgo > 60)
        return []
      const finite = (value: number | undefined) =>
        value != null && Number.isFinite(value) && value >= 0
          ? value
          : undefined
      return [
        {
          secondsAgo,
          decode:
            sample.phase === 'generating'
              ? finite(sample.decode_tps)
              : undefined,
          prefill:
            sample.phase === 'prefilling'
              ? finite(sample.prefill_tps)
              : undefined,
        },
      ]
    })
    .sort((a, b) => b.secondsAgo - a.secondsAgo)
  const ticks = yTicks(
    Math.max(
      1,
      ...points.flatMap((point) => [point.decode ?? 0, point.prefill ?? 0]),
    ),
  )
  const known = points.some(
    (point) => point.decode != null || point.prefill != null,
  )
  return (
    <figure
      aria-label={`Live request rates · Session ${session}`}
      data-window-seconds="60"
      className="min-w-0 space-y-2"
    >
      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span>
          Session {session} · Last 60 s{!active && ' · Just completed'}
        </span>
        <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
          <span
            className="h-0.5 w-4"
            style={{ backgroundColor: 'var(--accent)' }}
          />
          Decode
        </span>
        <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
          <span
            className="h-0.5 w-4"
            style={{ backgroundColor: 'var(--brand)' }}
          />
          Prefill
        </span>
        <span className="text-xs text-muted-foreground">tok/s</span>
      </figcaption>
      {!known && (
        <p className="text-xs text-muted-foreground">
          Rate: - · Waiting for token progress.
        </p>
      )}
      <div className="h-36 w-full font-mono text-xs tabular-nums">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={points}
            margin={{ top: 8, right: 28, bottom: 4, left: 0 }}
          >
            <XAxis
              dataKey="secondsAgo"
              type="number"
              reversed
              domain={[0, 60]}
              ticks={[0, 30, 60]}
              interval={0}
              tickFormatter={(value: number) =>
                value === 0 ? 'now' : `${number(value, 0)} s ago`
              }
              allowDataOverflow
            />
            <YAxis
              type="number"
              domain={[0, ticks.at(-1)!]}
              ticks={ticks}
              width={64}
              tickFormatter={(value: number) =>
                new Intl.NumberFormat('en', {
                  notation: 'compact',
                  maximumFractionDigits: 1,
                }).format(value)
              }
              allowDataOverflow
            />
            <Tooltip
              content={({ active: hovering, payload, label }) =>
                hovering && payload?.length ? (
                  <div className="rounded border bg-popover p-2 text-xs text-popover-foreground shadow-sm">
                    <p>{number(Number(label), 1)} s ago</p>
                    {payload.map((entry) => (
                      <p key={String(entry.dataKey)}>
                        {entry.dataKey === 'decode' ? 'Decode' : 'Prefill'} ·{' '}
                        {liveRate(
                          typeof entry.value === 'number'
                            ? entry.value
                            : undefined,
                        )}{' '}
                        tok/s
                      </p>
                    ))}
                  </div>
                ) : null
              }
            />
            <Line
              type="linear"
              dataKey="decode"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="linear"
              dataKey="prefill"
              stroke="var(--brand)"
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  )
}
