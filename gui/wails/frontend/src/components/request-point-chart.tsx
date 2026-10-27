import { useCallback, useId, useMemo, useState } from 'react'
import {
  ComposedChart,
  ReferenceDot,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import { number } from '@/lib/format'
import { yTicks } from './line-chart'

export type ChartPoint = {
  secondsAgo: number
  value: number
  requestId?: string
  promptTokens?: number
  cachedTokens?: number
  ttftMs?: number
  cacheKind?: string
}
function duration(ms: number, axis = false) {
  return ms >= 1000
    ? `${axis ? number(ms / 1000, 1) : (ms / 1000).toFixed(1)} s`
    : `${number(ms, 0)} ms`
}
function age(seconds: number) {
  return seconds >= 60
    ? `${number(seconds / 60, 0)} min`
    : `${number(seconds, 0)} s`
}
function pointDescription(
  point: ChartPoint,
  metric: string,
  unit: string,
  relativeToCompletion: boolean,
) {
  return [
    point.requestId,
    `${age(point.secondsAgo)} ${relativeToCompletion ? 'before latest completion' : 'ago'}`,
    unit === 'ms'
      ? `${duration(point.value)} to first token`
      : `${metric} ${number(point.value)} ${unit.trim()}`.trim(),
    point.promptTokens == null
      ? undefined
      : `${number(point.promptTokens, 0)} prompt tokens${point.cachedTokens == null ? '' : ` (${number(point.cachedTokens, 0)} cached)`}`,
    point.cachedTokens == null || point.promptTokens != null
      ? undefined
      : `${number(point.cachedTokens, 0)} cached tokens`,
    point.ttftMs == null || unit === 'ms'
      ? undefined
      : `${duration(point.ttftMs)} to first token`,
    point.cacheKind ? `cache ${point.cacheKind}` : undefined,
  ]
    .filter(Boolean)
    .join(' · ')
}

export function RequestPointChart({
  label,
  metricLabel,
  points,
  unit,
  color = 'var(--accent)',
  relativeToCompletion = false,
}: {
  label: string
  metricLabel: string
  points: readonly ChartPoint[]
  unit: string
  color?: string
  relativeToCompletion?: boolean
}) {
  const [activePoint, setActivePoint] = useState<ChartPoint>()
  const [logScale, setLogScale] = useState(false)
  const tooltipId = useId()
  const visible = useMemo(
    () => [...points].sort((a, b) => b.secondsAgo - a.secondsAgo),
    [points],
  )
  const maximum = Math.max(...visible.map((point) => point.value), 1)
  const ticks = useMemo(
    () =>
      logScale
        ? [
            0,
            ...Array.from(
              { length: Math.max(1, Math.ceil(Math.log10(maximum))) + 1 },
              (_, index) => 10 ** index,
            ),
          ]
        : yTicks(maximum),
    [maximum, logScale],
  )
  // log1p preserves zero observations; labels and tooltips retain real values.
  const position = useCallback(
    (value: number) => (logScale ? Math.log10(1 + value) : value),
    [logScale],
  )
  const axisLabel = useCallback(
    (value: number) =>
      unit === 'ms'
        ? value === 0 && maximum >= 1000
          ? '0 s'
          : duration(value, true)
        : new Intl.NumberFormat('en', {
            notation: 'compact',
            maximumFractionDigits: 1,
          }).format(value),
    [unit, maximum],
  )
  const axisWidth = Math.max(
    52,
    ...ticks.map((value) => axisLabel(value).length * 8 + 16),
  )
  const pointShape = useCallback(
    (props: unknown) => {
      const { cx, cy, payload, index } = props as {
        cx?: number
        cy?: number
        payload?: ChartPoint
        index?: number
      }
      if (cx == null || cy == null || !payload) return <g />
      return (
        <circle
          cx={cx}
          cy={cy}
          r={2.5}
          tabIndex={0}
          role="button"
          data-request-id={payload.requestId}
          data-point-index={index}
          aria-label={pointDescription(
            payload,
            metricLabel,
            unit,
            relativeToCompletion,
          )}
          aria-describedby={tooltipId}
          fill={color}
          stroke={color}
          strokeWidth={1}
          className="cursor-pointer focus:outline-2 focus:outline-offset-2 focus:outline-ring"
          onMouseEnter={() => setActivePoint(payload)}
          onMouseLeave={() => setActivePoint(undefined)}
          onFocus={() => setActivePoint(payload)}
          onBlur={() => setActivePoint(undefined)}
          onClick={() => setActivePoint(payload)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') setActivePoint(undefined)
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              setActivePoint(payload)
            }
          }}
        />
      )
    },
    [color, metricLabel, tooltipId, unit, relativeToCompletion],
  )
  // Keep the SVG stable across tooltip updates so request keyboard focus survives.
  const plot = useMemo(
    () => (
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={visible}
          margin={{ top: 12, right: 20, bottom: 4, left: 4 }}
        >
          <XAxis
            type="number"
            domain={[-0.5, Math.max(visible.length - 0.5, 0.5)]}
            ticks={[
              0,
              Math.floor((visible.length - 1) / 2),
              visible.length - 1,
            ].filter((value, index, all) => all.indexOf(value) === index)}
            tickFormatter={(value: number) =>
              visible[value]?.requestId ?? `#${value + 1}`
            }
            allowDataOverflow
          />
          <YAxis
            type="number"
            domain={[0, position(ticks.at(-1)!)]}
            ticks={ticks.map(position)}
            tickFormatter={(value: number) =>
              axisLabel(
                logScale ? Math.round((10 ** value - 1) * 1000) / 1000 : value,
              )
            }
            width={axisWidth}
            allowDataOverflow
          />
          {visible.map((point, index) => (
            <ReferenceDot
              key={point.requestId ?? index}
              x={index}
              y={position(point.value)}
              shape={(props) => pointShape({ ...props, payload: point, index })}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    ),
    [visible, ticks, pointShape, position, axisLabel, axisWidth, logScale],
  )
  const shownPoint = activePoint?.requestId
    ? visible.find((point) => point.requestId === activePoint.requestId)
    : activePoint
  return (
    <figure aria-label={label} className="min-w-0">
      <figcaption className="text-sm text-muted-foreground">
        {metricLabel} ({unit === 'ms' ? 'ms / s' : unit.trim()})
        <label className="mt-1 flex w-fit items-center gap-1 text-xs">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(event) => setLogScale(event.target.checked)}
          />
          Log scale
        </label>
      </figcaption>
      <div
        className="relative h-40 w-full font-mono text-xs tabular-nums"
        onMouseLeave={() => setActivePoint(undefined)}
        onMouseMove={(event) => {
          let distance = 9
          let nearest: ChartPoint | undefined
          for (const circle of event.currentTarget.querySelectorAll<SVGCircleElement>(
            '[data-point-index]',
          )) {
            const box = circle.getBoundingClientRect()
            const next = Math.hypot(
              event.clientX - box.left - box.width / 2,
              event.clientY - box.top - box.height / 2,
            )
            if (next < distance) {
              distance = next
              nearest = visible[Number(circle.dataset.pointIndex)]
            }
          }
          setActivePoint(nearest)
        }}
      >
        {shownPoint && (
          <div
            id={tooltipId}
            role="tooltip"
            className="pointer-events-none absolute top-0 right-0 z-10 max-w-full rounded border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-sm"
          >
            {pointDescription(
              shownPoint,
              metricLabel,
              unit,
              relativeToCompletion,
            )}
          </div>
        )}
        {plot}
      </div>
      <p className="text-xs text-muted-foreground">
        Request order · oldest to newest
        {logScale && ' · Log scale expands small values'}
      </p>
    </figure>
  )
}
