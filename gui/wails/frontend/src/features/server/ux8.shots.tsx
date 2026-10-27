import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedTransfer } from '@/components/recorded-observations'
import {
  coreView,
  mountCorePage,
  recordedRuntimePlan,
} from '@/components/core-page-test-support'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

test('UX8 M3: Server shows one whole-job line during a download', async () => {
  const view = coreView('models-download-in-progress')
  // The supervisor adds up the plan's files; this fixture states that total.
  const observed = recordedTransfer()
  view.progress!.overall = observed.overall
  const { client } = mountCorePage('/', view)
  client.setQueryData(
    ['transfer-samples', `prepare-${view.generation}`],
    observed.samples,
  )
  await expect
    .element(
      page.getByText(
        '103.7 GiB of 108.6 GiB · 95 % · 140.0 MiB/s · about 36 s left',
      ),
    )
    .toBeVisible()
  await shot('server', 'ux8-download-whole-job')
})

test('UX8 minors: the engine step shows its size once the plan is known', async () => {
  const view = coreView('not-prepared')
  view.preparation = [recordedRuntimePlan()]
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('1 Engine files · Next')
  await shot('onboarding', 'ux8-engine-step-size')
})
