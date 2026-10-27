import type { ChartPoint } from '@/components/request-point-chart'
import type { View } from '@/lib/bindings'

// Completion-time alignment uses the newest returned completion. Wall rates must
// never stand in for a missing per-request rate or missing CLI age.
export function requestSamples(
  view: View,
  metric: 'decode' | 'prefill' | 'ttft',
): ChartPoint[] {
  const newest = view.requests
    ?.filter((row) => row.outcome === 'completed')
    .reduce<(typeof view.requests)[number] | undefined>(
      (latest, row) =>
        !latest || Date.parse(row.time) > Date.parse(latest.time)
          ? row
          : latest,
      undefined,
    )
  const age = view.last_completed_request?.age_ms
  if (!newest) return []
  return (view.requests ?? [])
    .flatMap((row) => {
      const value = metric === 'ttft' ? row.ttft_ms : row[`${metric}_tps`]
      if (row.outcome !== 'completed' || value == null) return []
      const secondsAgo =
        (Date.parse(newest.time) - Date.parse(row.time) + (age ?? 0)) / 1000
      if (!Number.isFinite(secondsAgo) || secondsAgo < 0) return []
      return [
        {
          secondsAgo,
          value,
          requestId: row.id,
          promptTokens: row.prompt_tokens,
          cachedTokens: row.cached_tokens,
          ttftMs: row.ttft_ms,
          cacheKind: row.cache_kind,
        },
      ]
    })
    .sort((a, b) => b.secondsAgo - a.secondsAgo)
}
