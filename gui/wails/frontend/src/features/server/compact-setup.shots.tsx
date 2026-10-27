import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { coreView } from '@/components/core-page-test-support'

test('Server Inference compact layout', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = true
  await capturePage({
    feature: 'server',
    scenario: 'inference',
    view,
    prepare: async () => {
      const title = page.getByRole('button', { name: 'Inference', exact: true })
      const section =
        title.element().closest('details') ??
        title.element().closest('section')!
      if (section instanceof HTMLDetailsElement) section.open = true
      title.element().scrollIntoView({ block: 'start' })
      await document.fonts.ready
      console.log(
        `INFERENCE_HEIGHT=${Math.round(section.getBoundingClientRect().height)}`,
      )
      expect(section.getBoundingClientRect().height).toBeGreaterThan(0)
    },
  })
})

test('Settings Advanced compact layout', async () => {
  await capturePage({
    feature: 'settings',
    scenario: 'advanced-compact',
    view: coreView('stopped-saved-estimate'),
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await document.fonts.ready
    },
  })
})
