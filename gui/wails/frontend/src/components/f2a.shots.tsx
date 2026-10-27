import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedTransfer } from './recorded-observations'
import { coreView, mountCorePage, recordedPlan } from './core-page-test-support'

// Supplement the recorded captures with transfer details changed by F2a.
async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

test('F2a one whole-job line; the exact current file is a detail', async () => {
  const view = coreView('models-download-in-progress')
  const plan = recordedPlan()
  const job = view.models!.download!
  job.plan = plan
  const observed = recordedTransfer()
  expect(observed.overall).toEqual(coreView('preparing').progress!.overall)
  Object.assign(job, observed.file)
  job.file_position = { index: 4, total: plan.files.length }
  // The supervisor adds up the plan's files; this fixture states that total.
  job.overall = observed.overall
  const { client } = mountCorePage('/models', view)
  client.setQueryData(
    ['transfer-samples', `model-${job.id}-${job.kind}`],
    observed.samples,
  )
  const region = page.getByRole('region', { name: 'Model download progress' })
  await expect
    .element(region)
    .toHaveTextContent(
      '103.7 GiB of 108.6 GiB · 95 % · 140.0 MiB/s · about 36 s left',
    )
  await expect
    .element(region)
    .toHaveTextContent(`File 4 of ${plan.files.length}: 11.3 GiB of 11.3 GiB`)
  await shot('models', 'per-file-progress')
})

test('F2a missing folder is prose and has no copy action', async () => {
  const view = coreView()
  view.models!.prepared_models[0]!.manifest_path = undefined
  view.models!.prepared_models[0]!.model_path = undefined
  mountCorePage('/models', view)
  await expect
    .element(page.getByText('Folder not reported', { exact: true }))
    .toBeVisible()
  await shot('models', 'folder-not-reported')
})
