import { useEffect, useMemo } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { View } from '@/lib/bindings'

type Sample = { at: number; gpu?: View['gpu'] }
function gpuHistoryKey(view: View) {
  return ['gpu-history', view.instance?.instance_id ?? view.generation] as const
}
export function appendSample(
  history: readonly Sample[],
  view: View,
  at: number,
): Sample[] {
  if (history.at(-1)?.at === at) return [...history]
  return [
    ...history.filter((sample) => sample.at >= at - 300_000),
    { at, gpu: view.gpu },
  ].slice(-300)
}

// Only ViewShell mounts the collector. Feature pages subscribe without writing.
export function GPUHistoryCollector({ view, at }: { view: View; at: number }) {
  useGPUHistory(view)
  const client = useQueryClient()
  const identity = view.instance?.instance_id ?? view.generation
  const queryKey = useMemo(() => ['gpu-history', identity] as const, [identity])
  useEffect(() => {
    client.setQueryData<Sample[]>(queryKey, (history) =>
      appendSample(history ?? [], view, at),
    )
  }, [client, queryKey, view, at])
  return null
}
export function useGPUHistory(view: View) {
  return useQuery<Sample[]>({
    queryKey: gpuHistoryKey(view),
    queryFn: async () => [],
    initialData: [],
    enabled: false,
    staleTime: Infinity,
  }).data
}
