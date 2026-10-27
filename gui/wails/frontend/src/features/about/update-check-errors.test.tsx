import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions, viewKey } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import AboutPage from './index'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const missing = {
  code: 'update_information_missing',
  component: 'update',
  stage: '',
  remedy:
    'manifest returned HTTP 404 URL: https://github.com/haloforge-labs/bubo/releases/latest/download/bubo-release.json',
  retryable: true,
}

test('a feed 404 check shows one plain panel and no download failure', async () => {
  const view = fixture('update-none')
  const mounted = renderView(<AboutPage />, view)
  cleanup = mounted.cleanup
  vi.spyOn(actions, 'CheckUpdate').mockImplementation(async () => {
    view.update.state = 'failed'
    view.update.check_failed = true
    view.update.error = missing
    mounted.client.setQueryData(viewKey, structuredClone(view))
    throw missing
  })
  await page
    .getByRole('button', { name: 'Check for updates', exact: true })
    .click()
  await expect
    .element(
      page.getByText(
        'No update information was found. Bubo checks again later.',
        { exact: true },
      ),
    )
    .toBeVisible()
  expect(page.getByRole('alert').elements()).toHaveLength(1)
  expect(document.body.textContent).not.toMatch(
    /download could not connect|couldn't reach|retry the same action|update_information_missing/,
  )
  await page.getByText('Details', { exact: true }).click()
  await expect
    .element(page.getByText(missing.remedy, { exact: true }))
    .toBeVisible()
  expect(actions.CheckUpdate).toHaveBeenCalledTimes(1)
})

test.each([
  [
    'network',
    "Bubo can't check for updates: it couldn't reach the update server.",
  ],
  [
    'update_server_error',
    'The update server could not provide update information. Bubo checks again later.',
  ],
] as const)(
  'a %s check failure describes the actual cause once',
  async (code, sentence) => {
    const view = fixture('update-check-failed')
    view.update.error = { ...missing, code }
    cleanup = renderView(<AboutPage />, view).cleanup
    await expect
      .element(page.getByText(sentence, { exact: true }))
      .toBeVisible()
    expect(page.getByRole('alert').elements()).toHaveLength(1)
    expect(document.body.textContent).not.toContain(
      'The download could not connect.',
    )
  },
)
