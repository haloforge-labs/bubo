import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('no cell names a control the row does not have', async () => {
  const view = coreView('no-fit')
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Discover models' }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain(
    'Select this model to check memory',
  )
  // Only the selected model has a verdict; the others have an empty cell.
  await expect
    .element(page.getByText(view.fit.sentence, { exact: true }))
    .toBeVisible()
})

test('a download shows one whole-job line on Models', async () => {
  const view = coreView('models-download-in-progress')
  view.models!.download!.overall = {
    bytes_done: Math.round(50.0 * 1024 ** 3),
    bytes_total: Math.round(100.0 * 1024 ** 3),
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByText('50.0 GiB of 100.0 GiB · 50 %', { exact: false }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('left for this file')
})
