import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { capturePage } from '@/components/page-capture'
import { fixture } from '@/components/test-view'

test('E15: Server shows the default client name', async () => {
  document.documentElement.dataset.theme = inject('theme')
  mountCorePage('/', coreView('cli-page-facts-loopback'))
  await expect
    .element(
      page.getByText(/^Clients see this model as huihui-qwen38-flash-next-/),
    )
    .toBeVisible()
  await document.fonts.ready
  expect(await commands.captureFeature('server', 'e15-client-name')).toMatch(
    /\.png$/,
  )
})

test('E15: Connect snippets use the default client name', () =>
  capturePage({
    feature: 'connect',
    scenario: 'e15-client-name',
    view: fixture('cli-page-facts-loopback'),
  }))

// After the restart with local network access on: every client needs a key, so
// the templates keep the placeholder, and there is no Pi snippet for a network
// address (the CLI gives one for loopback only).
test('E18: Connect after the restart with local network access on', () => {
  const view = fixture('cli-page-facts-authenticated')
  view.key = { ...view.key, lan_access: true, lan_active: true }
  view.pi_snippet = ''
  return capturePage({
    feature: 'connect',
    scenario: 'e18-lan-on-templates',
    view,
    prepare: async () => {
      page
        .getByRole('region', { name: 'Client templates' })
        .element()
        .scrollIntoView({ block: 'start' })
    },
  })
})
