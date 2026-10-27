import {
  Line,
  ComposedChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { number, percent } from '@/lib/format'
import { useMemo } from 'react'

// GPU history alone has a display window, independent of CLI sampling cadence.
const historySeconds = 300
export function ChartTimeWindow() {
  return (
    <>
      Last <span className="tabular-nums">{historySeconds}</span> s
    </>
  )
}

// Tick choice only: Recharts positions the axes, marks, samples and clipping.
export function yTicks(top: number, bounded = false) {
  if (bounded) return Array.from({ length: 5 }, (_, index) => (top * index) / 4)
  const rough = Math.max(top, 1) / 4
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step =
    [1, 2, 2.5, 5, 10].find((value) => value * magnitude >= rough)! * magnitude
  return Array.from({ length: 5 }, (_, index) => step * index)
}

export function MetricLineChart({
  label,
  samples,
  unit,
  maximum,
  showTimeAxis = true,
  showWindow = true,
  compact = false,
}: {
  label: string
  samples: readonly { secondsAgo: number; value: number }[]
  unit: string
  maximum?: number
  showTimeAxis?: boolean
  showWindow?: boolean
  compact?: boolean
}) {
  const visible = useMemo(
    () =>
      samples
        .filter(
          (sample) =>
            sample.secondsAgo >= 0 && sample.secondsAgo <= historySeconds,
        )
        .sort((a, b) => b.secondsAgo - a.secondsAgo),
    [samples],
  )
  const ticks = yTicks(
    maximum ?? Math.max(...visible.map((sample) => sample.value), 1),
    maximum != null,
  )
  const title = label.replace(unit.trim(), '').replace(/\s+/g, ' ').trim()
  const last = visible.at(-1)
  return (
    <figure aria-label={label}>
      <figcaption className="text-sm text-muted-foreground">
        {title} ({unit.trim()}
        {showWindow && (
          <>
            , last <span className="tabular-nums">{historySeconds}</span> s
          </>
        )}
        )
      </figcaption>
      {visible.length >= 5 && (
        <div
          className={`${compact ? 'h-24' : 'h-40'} relative w-full font-mono text-xs tabular-nums`}
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={visible}
              margin={{ top: 12, right: 14, bottom: 4, left: 0 }}
            >
              <XAxis
                tick={showTimeAxis}
                tickLine={showTimeAxis}
                axisLine={showTimeAxis}
                dataKey="secondsAgo"
                type="number"
                reversed
                domain={[0, historySeconds]}
                ticks={[0, 60, 120, 180, 240, 300]}
                tickFormatter={(value: number) =>
                  value === 0 ? 'now' : number(value, 0)
                }
                allowDataOverflow
              />
              <YAxis
                type="number"
                domain={[0, ticks.at(-1)!]}
                ticks={ticks}
                allowDataOverflow
                tickFormatter={(value: number) => number(value, 2)}
                width={48}
              />
              <Tooltip
                allowEscapeViewBox={{ x: false, y: false }}
                content={({ active, payload, label: seconds }) =>
                  active && payload?.length ? (
                    <div
                      aria-label="Sample details"
                      className="rounded border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-sm"
                    >
                      {number(Number(seconds), 0)} s ago ·{' '}
                      {unit.trim() === '%'
                        ? percent(Number(payload[0]?.value))
                        : `${number(Number(payload[0]?.value))} ${unit.trim()}`}
                    </div>
                  ) : null
                }
              />
              <Line
                type="stepAfter"
                dataKey="value"
                stroke="var(--ink)"
                strokeWidth={1}
                dot={false}
                activeDot={false}
                isAnimationActive={false}
              />
              {last && (
                <ReferenceDot
                  x={last.secondsAgo}
                  y={last.value}
                  r={2}
                  fill="var(--ink)"
                  stroke="var(--ink)"
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  )
}
