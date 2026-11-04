import { RequestPointChart } from '@/components/request-point-chart'
import type { View } from '@/lib/bindings'
import { pageGrid } from '@/components/page-grid'
import { requestSamples } from './history'

const metrics = [
  { key: 'decode', label: 'Decode', unit: 'tok/s', color: 'var(--accent)' },
  { key: 'prefill', label: 'Prefill', unit: 'tok/s', color: 'var(--brand)' },
  {
    key: 'ttft',
    label: 'Time to first token',
    unit: 'ms',
    color: 'var(--ink)',
  },
] as const

export function RequestCharts({ view }: { view: View }) {
  const relativeToCompletion = view.last_completed_request?.age_ms == null
  const charts = metrics.map((metric) => ({
    metric,
    points: requestSamples(view, metric.key),
  }))
  // Nothing to plot (every request failed or none reported a value): the table
  // below carries the rows, so no chart section and no per-metric notes.
  if (!charts.some((chart) => chart.points.length)) return null
  return (
    <section aria-label="Request performance" className="space-y-2">
      <p className="text-xs text-muted-foreground">
        One point per completed request · Oldest → newest
        {relativeToCompletion &&
          view.requests.some((row) => row.outcome === 'completed') &&
          ' · Times are relative to the latest completion'}
      </p>
      <div className={pageGrid}>
        {charts.map(({ metric, points }) =>
          points.length ? (
            <RequestPointChart
              key={metric.key}
              label={`${metric.label} per completed request`}
              metricLabel={metric.label}
              unit={metric.unit}
              color={metric.color}
              points={points}
              relativeToCompletion={relativeToCompletion}
            />
          ) : (
            <p key={metric.key} className="text-sm text-muted-foreground">
              {metric.label}: none reported
            </p>
          ),
        )}
      </div>
    </section>
  )
}
