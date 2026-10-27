import { RequestPointChart } from '@/components/request-point-chart'
import type { View } from '@/lib/bindings'
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
  return (
    <section aria-label="Request performance" className="space-y-2">
      <p className="text-xs text-muted-foreground">
        Each point is one completed request. Oldest requests are on the left.
        {relativeToCompletion &&
          view.requests.some((row) => row.outcome === 'completed') &&
          ' Times are relative to the latest completion.'}
      </p>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,18rem),1fr))] gap-4">
        {metrics.map((metric) => {
          const points = requestSamples(view, metric.key)
          return points.length ? (
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
            <div key={metric.key} className="text-sm text-muted-foreground">
              <p>{metric.label}</p>
              <p>
                {metric.key === 'decode'
                  ? 'No decode rates reported for completed requests.'
                  : metric.key === 'prefill'
                    ? 'No prefill rates reported for completed requests.'
                    : 'No time to first token reported for completed requests.'}
              </p>
            </div>
          )
        })}
      </div>
    </section>
  )
}
