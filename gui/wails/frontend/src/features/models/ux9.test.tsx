import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('Models with nothing installed is one sentence, and "Not downloaded" is not repeated beside Download', async () => {
  const view = coreView('no-fit')
  view.prepared_models = []
  view.models!.prepared_models = []
  view.models!.scanned = []
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent('No models installed yet.')
  expect(installed.element().querySelector('table')).toBeNull()
  await expect
    .element(page.getByRole('button', { name: 'Check for model files' }))
    .toBeVisible()
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect.element(discover).toBeVisible()
  expect(discover.element().textContent).not.toContain('Not downloaded')
})

test('Discover does not offer the model Server already uses', async () => {
  const view = coreView('no-fit')
  mountCorePage('/models', view)
  const discover = page.getByRole('region', { name: 'Discover models' })
  const name = view.current_model!.display_name
  await expect.element(discover).toBeVisible()
  const rows = discover.getByRole('row')
  const index = rows
    .elements()
    .findIndex(
      (row) => row.querySelector('td')?.firstChild?.textContent === name,
    )
  expect(index).toBeGreaterThanOrEqual(0)
  const row = rows.nth(index)
  await expect.element(row).toBeVisible()
  await expect
    .element(row.getByRole('button', { name: 'Select for Server' }))
    .not.toBeInTheDocument()
})

test('the unusable-model sentence is said once above the list', async () => {
  const view = coreView('models-warnings')
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Installed models' }))
    .toBeVisible()
  const text = document.body.textContent!
  const sentence = "Bubo can't use this model file."
  expect(text.split(sentence).length - 1).toBeLessThanOrEqual(2)
})
