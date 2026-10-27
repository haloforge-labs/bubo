import { test } from 'vitest'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

for (const [state, scenario] of [
  ['Ready', 'ready-idle'],
  ['Stopped', 'stopped-saved-estimate'],
  ['Attached-external', 'external-server'],
] as const) {
  test(`Logs F5b empty log still shows its path while ${state}`, () => {
    const view = fixture(scenario)
    view.logs = []
    view.log_path = 'D:/fixture/server.log'
    return capturePage({
      feature: 'logs',
      scenario: `f5b-empty-${state.toLowerCase()}`,
      view,
    })
  })
}
