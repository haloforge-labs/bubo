import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('a lifecycle warning of the server is one plain sentence; its code and remedy are under Details', async () => {
  const view = coreView('ready-idle')
  view.server_warnings = [
    { code: 'queue_pressure', remedy: 'Wait for a free session.' },
  ]
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Other GPU work is slowing the server.', { exact: true }),
    )
    .toBeVisible()
  const detail = page.getByText('queue_pressure: Wait for a free session.')
  await expect.element(detail).toBeInTheDocument()
  expect(detail.element().closest('details')).not.toBeNull()
})

test('View warnings (for example unreadable settings) are shown on every page', async () => {
  const view = coreView('ready-idle')
  view.warnings = ['Local logs could not be written.']
  mountCorePage('/models', view)
  await expect
    .element(page.getByText('Local logs could not be written.'))
    .toBeVisible()
})

test('a CLI that Windows cannot start names the Visual C++ runtime and links to it', async () => {
  const open = vi.spyOn(actions, 'OpenExternal').mockResolvedValue({})
  const view = coreView('failed-with-remedy')
  view.failure = {
    code: 'vcredist_missing',
    source: 'supervisor',
    time: '',
    remedy: '',
    forced: false,
  }
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Bubo needs the Microsoft Visual C++ runtime.', {
        exact: true,
      }),
    )
    .toBeVisible()
  await expect
    .element(page.getByText('Install it, then open Bubo again.'))
    .toBeVisible()
  await page.getByRole('button', { name: 'Download the runtime' }).click()
  expect(open).toHaveBeenCalledExactlyOnceWith(
    'https://aka.ms/vs/17/release/vc_redist.x64.exe',
  )
})

test('a warning code the app does not know shows no line', async () => {
  const view = coreView('ready-idle')
  view.server_warnings = [{ code: 'future_code', remedy: 'Do something.' }]
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Do something.')
})
