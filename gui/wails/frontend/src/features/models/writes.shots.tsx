import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import {
  coreView,
  mountCorePage,
  recordedPlan,
  gemmaQatPlanFixture,
} from '@/components/core-page-test-support'
import { bytes } from '@/components/model-facts'

test('Models download plan shows the CLI remaining-space observation', async () => {
  document.documentElement.dataset.theme = inject('theme')
  // The dialog asks to start a download: the page itself is idle (no
  // "Downloading model" in the top bar).
  const view = coreView()
  const plan = recordedPlan(true)
  view.models!.download = {
    ...coreView('models-download-in-progress').models!.download!,
    state: 'awaiting_confirm',
    plan,
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('dialog'))
    .toHaveTextContent(
      `Needs ${bytes(plan.required_bytes)} (${bytes(plan.files.reduce((total, file) => total + file.size, 0))} + ${bytes(plan.volumes.find((volume) => volume.required_bytes === plan.required_bytes)!.safety_margin_bytes!, { unit: 'GiB', digits: 0 })} spare) · ${Object.values(plan.paths)[0]!.slice(0, 2)} has ${bytes(plan.free_bytes)} free`,
    )
  await document.fonts.ready
  await commands.captureFeature('models', 'download-space-after')
})

test('Models Huihui QAT consent lists the three CLI-pinned files', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  const plan = structuredClone(gemmaQatPlanFixture.plan)
  const entry = view.models!.catalogue.find((row) => row.id === plan.id)!
  view.models!.download = {
    ...coreView('models-download-in-progress').models!.download!,
    id: entry.id,
    state: 'awaiting_confirm',
    plan: plan as unknown as ReturnType<typeof recordedPlan>,
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('dialog'))
    .toHaveTextContent(entry.display_name)
  await page.getByText('Files (3)', { exact: true }).click()
  const files = page.getByRole('list', { name: 'Download files' })
  for (const file of plan.files)
    await expect.element(files).toHaveTextContent(bytes(file.size))
  await document.fonts.ready
  await commands.captureFeature('models', 'download-plan-huihui-qat')
})
