import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import { capturePage } from '@/components/page-capture'
import { coreView } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('Stopped Server starts the setup on screen without a separate Save', async () => {
  const view = coreView('stopped-saved-estimate')
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  await capturePage({
    feature: 'server',
    scenario: 'stopped-unsaved',
    view,
    prepare: async () => {
      await page.getByLabelText('Sessions', { exact: true }).fill('2')
      await expect
        .element(
          page.getByRole('button', { name: 'Start server', exact: true }),
        )
        .toBeEnabled()
      expect(document.body.textContent).not.toContain(
        'Save server setup before starting or restarting.',
      )
    },
  })
})

test('Tight Server keeps one current memory panel', async () => {
  await capturePage({
    feature: 'server',
    scenario: 'tight-memory',
    view: coreView('memory-gpu-headroom'),
    prepare: async () => {
      const panel = page.getByRole('region', {
        name: 'Memory fit',
        exact: true,
      })
      await expect
        .element(panel.getByRole('heading', { name: 'Tight', exact: true }))
        .toBeVisible()
      await expect
        .element(
          page.getByRole('button', { name: 'Start server', exact: true }),
        )
        .toBeDisabled()
      expect(document.body.textContent).not.toContain(
        'Blocked: memory is tight.',
      )
      panel.element().scrollIntoView({ block: 'center' })
    },
  })
})

test('Running Server', async () => {
  await capturePage({
    feature: 'server',
    scenario: 'running',
    view: coreView('ready-idle'),
    prepare: async () => {
      await expect
        .element(page.getByLabelText('Server status').getByRole('status'))
        .toHaveTextContent('Running')
    },
  })
})
