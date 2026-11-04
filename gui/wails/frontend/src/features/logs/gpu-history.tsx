import { ChartTimeWindow, MetricLineChart } from '@/components/line-chart'
import type { View } from '@/lib/bindings'
import { pageGrid } from '@/components/page-grid'
import { useGPUHistory } from '@/lib/gpu-history'

const metrics = [
  { key: 'use_pct', label: 'GPU use', unit: '%' },
  { key: 'memory_used_bytes', label: 'GPU memory', unit: 'GiB' },
  { key: 'temperature_c', label: 'GPU temperature', unit: '°C' },
  { key: 'chip_power_w', label: 'Chip power (CPU, GPU, NPU)', unit: 'W' },
] as const

export function GPUHistory({
  view,
  updatedAt,
}: {
  view: View
  updatedAt: number
}) {
  const history = useGPUHistory(view)
  const newest = history.at(-1)?.at ?? updatedAt
  const charts = metrics.flatMap((metric) => {
    const memory = metric.key === 'memory_used_bytes'
    const samples = history.flatMap((sample) => {
      const value = sample.gpu?.[metric.key]
      const secondsAgo = (newest - sample.at) / 1000
      return value == null || secondsAgo < 0 || secondsAgo > 300
        ? []
        : [{ secondsAgo, value: memory ? value / 1024 ** 3 : value }]
    })
    const values = samples.map((sample) => sample.value)
    // Preserve the Server page's five observations and 5% change threshold. A
    // series that never changes (all zeros included) says nothing: omit it first.
    const top = Math.max(...values)
    const bottom = Math.min(...values)
    if (values.length < 5 || top === bottom || top - bottom < top * 0.05)
      return []
    return [{ metric, samples, memory }]
  })
  if (!charts.length) return null
  return (
    <section aria-label="GPU history" className="space-y-2 border-b pb-4">
      <div className="flex items-baseline gap-3">
        <h2 className="text-xl">GPU history</h2>
        <p className="text-xs text-muted-foreground">
          <ChartTimeWindow />
        </p>
      </div>
      <div className={pageGrid}>
        {charts.map(({ metric, samples, memory }) => (
          <MetricLineChart
            key={metric.key}
            label={`${metric.label} history`}
            unit={metric.unit}
            showWindow={false}
            compact
            maximum={
              metric.key === 'use_pct'
                ? 100
                : memory && view.gpu?.memory_total_bytes != null
                  ? view.gpu.memory_total_bytes / 1024 ** 3
                  : undefined
            }
            samples={samples}
          />
        ))}
      </div>
    </section>
  )
}
