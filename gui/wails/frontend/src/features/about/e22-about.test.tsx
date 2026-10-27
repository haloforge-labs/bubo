import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import AboutPage from './index'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})



















const border = (name: string) =>
  getComputedStyle(page.getByRole('button', { name, exact: true }).element())
    .borderTopWidth

test('6a: after an update the card says one line and Dismiss belongs to it', async () => {
  const view = fixture('update-current')
  view.update.current_version = '0.1.1'
  view.update.last_result_notice = 'Updated to 0.1.1.'
  cleanup = renderView(<AboutPage />, view).cleanup
  await expect
    .element(page.getByText('Updated to 0.1.1.', { exact: true }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Bubo is up to date.')
  await expect
    .element(page.getByRole('button', { name: 'Dismiss' }))
    .toBeVisible()
})

test('6a: with no notice the card says it is up to date and has no Dismiss', async () => {
  cleanup = renderView(<AboutPage />, fixture('update-current')).cleanup
  await expect
    .element(page.getByText('Bubo is up to date.', { exact: true }))
    .toBeVisible()
  expect(page.getByRole('button', { name: 'Dismiss' }).elements()).toHaveLength(
    0,
  )
})

test('6b: Check for updates is a secondary button when current and primary when unchecked', async () => {
  cleanup = renderView(<AboutPage />, fixture('update-current')).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Check for updates' }))
    .toBeVisible()
  expect(border('Check for updates')).toBe('1px')
  cleanup()
  document.body.replaceChildren()
  cleanup = renderView(<AboutPage />, fixture('update-none')).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Check for updates' }))
    .toBeVisible()
  expect(border('Check for updates')).toBe('0px')
})

test('6c, 6d: Downloading… while it downloads; Installing… and Later disabled while it installs', async () => {
  let finish: (value: unknown) => void = () => undefined
  vi.spyOn(actions, 'DownloadUpdate').mockImplementation(
    () => new Promise((resolve) => (finish = resolve)) as never,
  )
  vi.spyOn(actions, 'InstallUpdate').mockImplementation(
    () => new Promise((resolve) => (finish = resolve)) as never,
  )
  const available = renderView(<AboutPage />, fixture('update-available'))
  cleanup = available.cleanup
  await page.getByRole('button', { name: /^Download update/ }).click()
  await expect
    .element(page.getByRole('button', { name: 'Downloading…' }))
    .toBeDisabled()
  expect(document.body.textContent).not.toContain('Working…')
  finish({})
  cleanup()
  document.body.replaceChildren()

  cleanup = renderView(<AboutPage />, fixture('update-ready')).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Later', exact: true }))
    .toBeEnabled()
  await page
    .getByRole('button', { name: 'Install and restart', exact: true })
    .click()
  await expect
    .element(page.getByRole('button', { name: 'Installing…' }))
    .toBeDisabled()
  await expect
    .element(page.getByRole('button', { name: 'Later', exact: true }))
    .toBeDisabled()
  expect(document.body.textContent).not.toContain('Working…')
  finish({})
})

test('6e: the privacy note beside Check for updates stays', async () => {
  cleanup = renderView(<AboutPage />, fixture('update-current')).cleanup
  await expect
    .element(page.getByText(/sends your installed version in its User-Agent/))
    .toBeVisible()
})

test('E-1: the update fixtures offer real 0.1.1 over published 0.1.0', () => {
  for (const state of ['available', 'progress', 'ready']) {
    const { update, product } = fixture(`update-${state}`)
    expect(update.current_version, state).toBe('0.1.0')
    expect(product?.version, state).toBe('0.1.0')
    expect(update.available_version, state).toBe('0.1.1')
  }
})

// Review of the update captures: the update_interrupted
// card printed its sentence twice (the saved notice and the error box) and had
// Later and Dismiss side by side, both of which dismiss the same update.
// Failure list, written before the code:
//  7a. the same sentence shows twice on one card;
//  7b. two buttons with one effect (Later and Dismiss) show together;
//  7c. the fix hides the sentence or Dismiss when they are the only copy.
test('7: an interrupted update says its sentence once and offers one dismiss', async () => {
  const sentence =
    'The update stopped before it changed anything. Your installed version is unchanged.'
  const view = fixture('update-failed')
  view.update.error = {
    code: 'update_interrupted',
    component: 'update',
    stage: '',
    remedy: sentence,
    retryable: true,
  }
  view.update.last_result_notice = sentence
  view.update.check_failed = false
  cleanup = renderView(<AboutPage />, view).cleanup
  await expect.element(page.getByRole('alert')).toHaveTextContent(sentence)
  // The closed Details may repeat the CLI text; the visible card says it once.
  // The error box carries it; no separate notice line repeats it.
  const notices = [
    ...document.querySelectorAll('[data-update] [role="status"]'),
  ]
  expect(
    notices.filter((node) => node.textContent?.includes(sentence)),
  ).toHaveLength(0) // 7a
  const dismissers = ['Later', 'Dismiss'].filter(
    (name) =>
      page.getByRole('button', { name, exact: true }).elements().length > 0,
  )
  expect(dismissers).toHaveLength(1) // 7b 7c
  // 7c for the after-update line ("Updated to 0.1.1." with Dismiss) is test 6a.
})
