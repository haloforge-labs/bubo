import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { fixture } from '@/components/test-view'

// These are Go-built Views from CLI estimate recordings. Each model verdict
// and the saved setup fit remain paired with the same scenario input.
for (const scenario of [
  'memory-gpu-headroom',
  'memory-default-setup',
  'no-fit',
  'memory-smaller-context',
]) {
  test(`Models memory scenario ${scenario}`, () =>
    capturePage({
      feature: 'models',
      scenario: `memory-${scenario}`,
      view: fixture(scenario),
      prepare: async () => {
        const discover = page.getByRole('region', { name: 'Discover models' })
        await expect.element(discover).toBeInTheDocument()
        discover.element().scrollIntoView({ block: 'start' })
      },
    }))
}
