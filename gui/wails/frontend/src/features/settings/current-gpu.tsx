import type { View } from '@/lib/bindings'
import { bytes, number, percent } from '@/lib/format'

export function CurrentGPU({ gpu }: { gpu: View['gpu'] }) {
  const view = { gpu }
  const gpuMetrics = [
    { key: 'use_pct', label: 'GPU use', unit: '%' },
    { key: 'memory_used_bytes', label: 'GPU memory', unit: 'GiB' },
    { key: 'temperature_c', label: 'GPU temperature', unit: '°C' },
    { key: 'chip_power_w', label: 'Chip power (CPU, GPU, NPU)', unit: 'W' },
  ] as const
  return (
    <>
      {' '}
      {view.gpu && (
        <div className="space-y-2">
          <section
            aria-label="GPU"
            className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
          >
            {gpuMetrics.map((metric) => {
              const value = view.gpu?.[metric.key]
              if (value == null) return null
              const memory = metric.key === 'memory_used_bytes'
              return (
                <figure key={metric.key} aria-label={metric.label}>
                  <p className="text-xl tabular-nums">
                    {memory
                      ? `${bytes(value, { unit: 'GiB' })}${view.gpu?.memory_total_bytes != null ? ` / ${bytes(view.gpu.memory_total_bytes, { unit: 'GiB' })}` : ''}`
                      : metric.key === 'use_pct'
                        ? percent(value)
                        : `${number(value, 0)} ${metric.unit}`}
                  </p>
                  <figcaption className="text-sm text-muted-foreground">
                    {metric.label}
                  </figcaption>
                </figure>
              )
            })}
          </section>
          {(view.gpu.age_seconds != null ||
            view.gpu.window_seconds != null) && (
            <details className="text-xs text-muted-foreground">
              <summary>Details</summary>
              <p>
                GPU sample:{' '}
                {view.gpu.age_seconds == null ? (
                  'age not reported'
                ) : (
                  <>
                    <span className="tabular-nums">
                      {number(view.gpu.age_seconds)} s
                    </span>{' '}
                    old
                  </>
                )}{' '}
                ·{' '}
                {view.gpu.window_seconds == null ? (
                  'window not reported'
                ) : (
                  <>
                    over{' '}
                    <span className="tabular-nums">
                      {number(view.gpu.window_seconds)} s
                    </span>
                  </>
                )}
              </p>
            </details>
          )}
        </div>
      )}
    </>
  )
}
