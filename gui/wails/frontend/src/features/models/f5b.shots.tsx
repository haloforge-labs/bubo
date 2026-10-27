import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'
import {
  mountCorePage,
  recordedPlan,
} from '@/components/core-page-test-support'

// While the first download runs the CLI lists no package yet: the models list
// of that moment is the recorded list of a computer with no model.
function viewFor(state: string, scenario: string) {
  const view = fixture(scenario)
  if (state === 'partial') {
    const none = fixture('runtime-ready-model-missing').models!
    view.models = {
      ...view.models!,
      prepared_models: none.prepared_models,
      scanned: none.scanned,
    }
  }
  return view
}

for (const [state, scenario] of [
  ['not_prepared', 'not-prepared'],
  ['partial', 'models-download-in-progress'],
  ['prepared', 'memory-gpu-headroom'],
] as const) {
  test(`Models F5b job phase ${state}`, () => {
    return capturePage({
      feature: 'models',
      scenario: `f5b-job-${state}`,
      view: viewFor(state, scenario),
      prepare: async () => {
        if (state !== 'prepared')
          await expect
            .element(page.getByRole('region', { name: 'Installed models' }))
            .toHaveTextContent('No models installed yet.')
        else {
          await expect
            .element(page.getByRole('region', { name: 'Installed models' }))
            .not.toHaveTextContent('No models installed yet.')
          await expect
            .element(page.getByRole('region', { name: 'Installed models' }))
            .not.toHaveTextContent('Folder not reported')
        }
      },
    })
  })
}

test('Models F5b download plan lists the exact CLI files and sizes', async () => {
  document.documentElement.dataset.theme = inject('theme')
  // The dialog offers a download: the page is idle (no "Downloading model" in
  // the top bar) and the model is one that is not installed.
  const view = fixture()
  view.models!.download = {
    ...fixture('models-download-in-progress').models!.download!,
    state: 'awaiting_confirm',
    plan: recordedPlan(true),
  }
  mountCorePage('/models', view)
  await expect.element(page.getByRole('dialog')).toBeVisible()
  await page
    .getByText(`Files (${recordedPlan(true).files.length})`, { exact: true })
    .click()
  await expect
    .element(page.getByRole('list', { name: 'Download files' }))
    .toBeVisible()
  await document.fonts.ready
  await commands.captureFeature('models', 'f5b-download-files')
})
