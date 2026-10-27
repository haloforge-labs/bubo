import { realRequestsView } from './v12-scenario'
import facts from './progress-cli-facts.json'

// Owner's recorded r10: 1,373 prompt tokens, 1,345 from cache (SPEC §5).
// Arrange its natural prefill-start state, before the remaining 28 are read.
// Slot identity comes from the real ready-busy CLI recording. No partial
// prefill is invented: this short remainder fits within one prefill step.
export function promptReadingView() {
  const view = realRequestsView()
  // Captured from this worktree's CPU-built bubo serve --list-switches --json.
  view.product = { ...facts.product }
  if (view.update) view.update.current_version = facts.product.version
  for (const setting of [...view.settings, ...view.effective_switches]) {
    if (setting.name === '--context')
      setting.effective = {
        kind: 'number',
        text: String(facts.context_per_session),
      }
  }
  view.requests = view.requests.filter(
    (row) => !['r10', 'r11'].includes(row.id),
  )
  view.slots = replaySlots('cached')
  return view
}

// Both browser scenarios and the real CLI process E2E consume these frames.
export function replaySlots(name: string) {
  return facts.replay
    .find((frame) => frame.name === name)!
    .slots.map((slot) =>
      Object.fromEntries(
        Object.entries(slot).map(([key, value]) => [key, value ?? undefined]),
      ),
    ) as ReturnType<typeof realRequestsView>['slots']
}
