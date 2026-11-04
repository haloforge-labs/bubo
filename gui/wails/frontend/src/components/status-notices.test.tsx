import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

// A saved setting and a saved support bundle are not errors: the first is a plain
// status line, the second is confirmed by the Logs page itself and nowhere else.
test('a restart notice is a status line, never the red error box', async () => {
  const view = coreView('ready-idle')
  view.notice = {
    code: 'settings_saved',
    detail: 'Restart to apply these settings.',
  }
  mountCorePage('/', view)
  await expect
    .element(
      page
        .getByRole('status')
        .filter({ hasText: 'Restart to apply these settings.' }),
    )
    .toBeVisible()
  expect(document.querySelector('.error-note')).toBeNull()
})

test('a saved support bundle is not repeated by the shell', async () => {
  const view = coreView('ready-idle')
  view.notice = { code: 'support_bundle_saved', detail: 'C:\\x\\bundle.zip' }
  mountCorePage('/logs' as '/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Logs', level: 1 }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Support bundle saved')
  expect(document.querySelector('.error-note')).toBeNull()
})

test('the header holds only the sidebar toggle on Server and has no separator', async () => {
  mountCorePage('/', coreView('stopped-saved-estimate'))
  await expect
    .element(page.getByRole('heading', { name: 'Server', level: 1 }))
    .toBeVisible()
  const header = document.querySelector('header')!
  expect(header.querySelector('[data-slot="separator"]')).toBeNull()
  expect(header.getBoundingClientRect().height).toBeLessThan(64)
})
