import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import {
  coreView,
  mountCorePage,
  recordedPlan,
} from '@/components/core-page-test-support'
import { bridgeReply } from '@/components/bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('job: find, plan, download, see installed, choose for Server without another download', async () => {
  const view = coreView('models-download-in-progress')
  // The recorded list holds Flash Next and four other packages. This job
  // downloads Flash Next, so it starts as the one package that is not there.
  const local = structuredClone(view.models!.prepared_models[0]!)
  const entry = view.models!.catalogue.find((entry) => entry.id === local.id)!
  view.models!.prepared_models = view.models!.prepared_models.filter(
    (row) => row.id !== local.id,
  )
  view.prepared_models = view.models!.prepared_models
  view.state = 'Stopped'
  view.instance = undefined
  view.models!.download = undefined
  entry.estimate.memory_estimate = {
    ...entry.estimate.memory_estimate!,
    fits: true,
    recommended: true,
  }
  const plan = recordedPlan()
  const download = vi
    .spyOn(actions, 'Download')
    .mockResolvedValue(bridgeReply({ plan }))
  const confirm = vi
    .spyOn(actions, 'ConfirmDownload')
    .mockResolvedValue(bridgeReply({}))
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  const { replaceView } = mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect.element(discover).toBeVisible()
  const rows = discover.getByRole('row')
  const index = rows
    .elements()
    .findIndex(
      (row) =>
        row.querySelector('td')?.firstChild?.textContent === entry.display_name,
    )
  expect(index).toBeGreaterThanOrEqual(0)
  const row = rows.nth(index)
  await expect.element(installed).not.toHaveTextContent(entry.display_name)
  await row.getByRole('button', { name: 'Download', exact: true }).click()
  await expect
    .element(page.getByRole('dialog'))
    .toHaveTextContent('Review the planned files and space needed.')
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Download', exact: true })
    .click()
  expect(download).toHaveBeenCalledExactlyOnceWith(entry.id)
  expect(confirm).toHaveBeenCalledExactlyOnceWith(entry.id)
  const progress = coreView('models-download-in-progress').models!.download!
  replaceView({ ...view, models: { ...view.models!, download: progress } })
  await expect.element(installed).not.toHaveTextContent(entry.display_name)
  await expect
    .element(row.getByRole('button', { name: 'Downloading', exact: true }))
    .toBeDisabled()
  const complete = structuredClone(view)
  complete.prepared_models = [local]
  complete.models!.prepared_models = [local]
  replaceView(complete)
  await expect.element(installed).toHaveTextContent(local.display_name)
  await expect.element(row).toHaveTextContent('Installed')
  // Server uses another model now, so this one can be chosen.
  complete.current_model = {
    ...complete.current_model!,
    catalogue_id: '',
    manifest_path: '',
  }
  replaceView(structuredClone(complete))
  await expect
    .element(row.getByRole('button', { name: 'Download', exact: true }))
    .not.toBeInTheDocument()
  await row.getByRole('button', { name: 'Select for Server' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: { manifest_path: local.manifest_path },
    clear: ['model_path'],
  })
  const selected = structuredClone(complete)
  selected.saved.manifest_path = local.manifest_path
  replaceView(selected)
  // The selected model is not offered again.
  await expect
    .element(row.getByRole('button', { name: 'Select for Server' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('link', { name: 'Open Server', exact: true }))
    .toBeVisible()
})

test('a not_prepared default stays in Discover and never exposes the full installed size', async () => {
  // The recorded answer for a computer with no model: the status row says
  // not_prepared and the models list has no package and no loose file.
  const view = coreView('runtime-ready-model-missing')
  expect(view.prepared_models[0]!.prepared_state).toBe('not_prepared')
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent('No models installed yet.')
  expect(installed.element().querySelectorAll('tbody tr')).toHaveLength(0)
})

// E19: an incomplete package is listed (it can be repaired where it is) but it
// says so, cannot be selected and does not show the size of a complete one.
test('a partial package is listed as incomplete, without the full installed size', async () => {
  const view = coreView('models-download-in-progress')
  const row = structuredClone(view.models!.prepared_models[0]!)
  const full = row.size_bytes
  row.prepared_state = 'partial'
  // One package on its own: the recorded list has four more and a loose file.
  view.models!.prepared_models = [row]
  view.models!.scanned = []
  view.models!.download = undefined // no job: the row shows its own state
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent('Incomplete')
  expect(installed.element().textContent).not.toContain(
    `${(full / 1024 ** 3).toFixed(1)} GiB`,
  )
  await expect
    .element(installed.getByRole('button', { name: 'Select for Server' }))
    .toBeDisabled()
})
