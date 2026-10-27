import type { View } from '@/lib/bindings'
import { coreView } from '@/components/core-page-test-support'
import { replaySlots } from './progress-scenario'







export function liveRequestView(
  phase: 'reading' | 'writing' = 'writing',
): View {
  const view = coreView('ready-busy')
  view.requests = []
  view.last_completed_request = undefined
  // ready-busy is the ViewBuilder projection of lrel8's actual writing frame.
  if (phase === 'reading') view.slots = replaySlots('cached')
  view.status_feed = {
    state: 'updating',
    sample_time: '2026-10-26T23:15:04Z',
    age_seconds: 0,
  }
  Object.assign(view.slots[0]!, {
    elapsed_ms: 4000,
    prefill_tps: phase === 'reading' ? 4096 : undefined,
    decode_tps: phase === 'writing' ? 54 : undefined,
  })
  Object.assign(view, {
    live_sample_time: view.status_feed.sample_time,
    live_samples: [0, 1, 2, 3].map((index) => ({
      instance_id: 'arranged-live-rate-contract',
      slot_id: 0,
      request_id: view.slots[0]!.request_id!,
      sample_time: `2026-10-26T23:15:0${index + 1}Z`,
      phase: phase === 'reading' ? 'prefilling' : 'generating',
      elapsed_ms: (index + 1) * 1000,
      prefill_tps: phase === 'reading' ? 4096 : undefined,
      decode_tps: phase === 'writing' ? 54 : undefined,
    })),
  })
  return view
}

export function firstCompletionView() {
  const view = liveRequestView()
  const recorded = coreView('ready-busy').requests.find(
    (request) => request.outcome === 'completed',
  )!
  view.requests = [{ ...recorded, time: '2026-10-26T23:15:04Z' }]
  view.slots = replaySlots('idle')
  view.last_completed_request = undefined
  return view
}
