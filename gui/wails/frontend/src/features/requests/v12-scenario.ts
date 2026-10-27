import { fixture } from '@/components/test-view'



export function realRequestsView() {
  const view = fixture('ready-busy')
  const rows = [
    ['r3', 17, 0, 1007, 17, 36.8, 'miss'],
    ['r4', undefined, undefined, undefined, undefined, undefined, undefined],
    ['r5', 17, 17, 8, undefined, 45.5, 'memory'],
    ['r6', 580, 0, 563, 1038, 60.2, 'miss'],
    ['r7', 1325, 0, 1046, 1273, 69.7, 'miss'],
    ['r8', 2606, 0, 1876, 1392, 52.4, 'miss'],
    ['r9', 1350, 0, 1036, 1312, undefined, 'miss'],
    ['r10', 1373, 1345, 170, 176, undefined, 'memory'],
    ['r11', 1395, 1368, 171, 167, undefined, 'memory'],
  ] as const
  view.requests = rows.map(
    ([id, prompt, cached, ttft, prefill, decode, cache], index) => ({
      id,
      time: new Date(Date.UTC(2026, 9, 26, 0, 0, index * 4)).toISOString(),
      outcome: id === 'r4' ? 'failed' : 'completed',
      status: id === 'r4' ? 404 : 200,
      error_code: id === 'r4' ? 'model_not_found' : undefined,
      finish_reason: id === 'r4' ? undefined : id === 'r3' ? 'length' : 'stop',
      prompt_tokens: prompt,
      cached_tokens: cached,
      ttft_ms: ttft,
      prefill_tps: prefill,
      decode_tps: decode,
      cache_kind: cache,
    }),
  )
  view.last_completed_request = { ...view.last_completed_request!, age_ms: 0 }
  return view
}
