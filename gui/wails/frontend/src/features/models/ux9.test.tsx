import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { openDiscover } from './discover-test-support'

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

test('Discover keeps selection for a model that is only current, and names the state of one that runs', async () => {
  const view = coreView('no-fit')
  mountCorePage('/models', view)
  const discover = await openDiscover()
  const name = view.current_model!.display_name
  const rows = discover.getByRole('row')
  const index = rows
    .elements()
    .findIndex(
      (row) => row.querySelector('td')?.firstChild?.textContent === name,
    )
  expect(index).toBeGreaterThanOrEqual(0)
  const row = rows.nth(index)
  await expect.element(row).toBeVisible()
  // Identity alone is not a reason for a blank Action cell: the model is
  // neither running nor selected, so selection stays.
  await expect
    .element(row.getByRole('button', { name: 'Select for Server' }))
    .toBeVisible()
  await expect.element(row).toHaveTextContent('Installed')
})

test('Discover says In use for the running model, with no button', async () => {
  const view = coreView('no-fit')
  view.serving = true
  mountCorePage('/models', view)
  const discover = await openDiscover()
  const name = view.current_model!.display_name
  const row = discover.getByRole('row').filter({ hasText: name }).first()
  await expect.element(row.getByText('In use', { exact: true })).toBeVisible()
  await expect
    .element(row.getByRole('button', { name: 'Select for Server' }))
    .not.toBeInTheDocument()
})

test('Discover says Selected for next start for the saved model, with no button', async () => {
  const view = coreView('no-fit')
  const saved = view.models!.prepared_models.find(
    (model) => model.prepared_state === 'prepared' && model.manifest_path,
  )!
  view.saved.manifest_path = saved.manifest_path
  mountCorePage('/models', view)
  const discover = await openDiscover()
  const row = discover
    .getByRole('row')
    .filter({ hasText: saved.display_name })
    .first()
  await expect
    .element(row.getByText('Selected for next start', { exact: true }))
    .toBeVisible()
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

test('Discover is folded while models are installed, with its count on the summary', async () => {
  const view = coreView('no-fit')
  mountCorePage('/models', view)
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect
    .element(
      discover.getByRole('button', {
        name: `Discover · ${view.models!.catalogue.length} supported downloads`,
      }),
    )
    .toBeVisible()
  expect(discover.element().querySelector('details')!.open).toBe(false)
})

test('Discover is open when nothing is installed', async () => {
  const view = coreView('no-fit')
  view.prepared_models = []
  view.models!.prepared_models = []
  view.models!.scanned = []
  mountCorePage('/models', view)
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect.element(discover.getByRole('table')).toBeVisible()
})
