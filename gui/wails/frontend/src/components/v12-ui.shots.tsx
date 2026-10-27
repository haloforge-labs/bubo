import { shortByShown } from '@/components/model-facts'
import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { capturePage } from './page-capture'
import {
  coreView,
  mountCorePage,
  recordedPlan,
  recordedRuntimePlan,
  combinedShortSpaceView,
} from './core-page-test-support'

test('V12 Connect stopped', () =>
  capturePage({
    feature: 'connect',
    scenario: 'v12-stopped',
    view: coreView('stopped'),
  }))

test('V12 combined consent names the drive and space to free in plain text', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const combined = combinedShortSpaceView()
  mountCorePage('/onboarding', combined)
  await expect
    .element(
      page.getByText(
        `Free up space on C: ${shortByShown(combined.preparation.reduce((sum, plan) => sum + plan.volumes[0]!.bytes_to_write!, 0) + Math.max(...combined.preparation.map((plan) => plan.volumes[0]!.safety_margin_bytes!)), Math.min(...combined.preparation.map((plan) => plan.volumes[0]!.free_bytes)))} more is needed. Then try again.`,
        { exact: true },
      ),
    )
    .toBeVisible()
  await document.fonts.ready
  expect(
    await commands.captureFeature('onboarding', 'v12-combined-space-short'),
  ).toMatch(/\.png$/)
})

for (const details of [false, true]) {
  test(`V12 model step details ${details}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    const view = coreView('models-download-in-progress')
    view.prepared = false
    view.runtime.readiness.runtime = true
    view.runtime.readiness.model = false
    view.preparation = [recordedRuntimePlan()]
    view.models!.download!.plan = recordedPlan()
    view.models!.download!.file_position = {
      index: 2,
      total: view.models!.download!.plan!.files.length,
    }
    view.models!.download!.overall = {
      bytes_done: 18361026752,
      bytes_total: 116617136608,
    }
    view.progress = {
      component: 'runtime',
      phase: 'download',
      file_id: view.preparation[0]!.files[2]!.file_id,
      bytes_done: view.preparation[0]!.files[2]!.size,
      bytes_total: view.preparation[0]!.files[2]!.size,
      resume_offset: 0,
      verification_method: '',
      overall: { bytes_done: 1014229359, bytes_total: 1014229359 },
    }
    mountCorePage('/onboarding', view)
    await expect
      .element(page.getByRole('heading', { name: 'Get started', exact: true }))
      .toBeVisible()
    if (details)
      await page.getByText('Details', { exact: true }).first().click()
    await document.fonts.ready
    expect(
      await commands.captureFeature(
        'onboarding',
        details ? 'v12-model-details' : 'v12-model-step',
      ),
    ).toMatch(/\.png$/)
    if (details) {
      const files = page.getByRole('list', { name: 'Download file progress' })
      files.element().scrollIntoView({ block: 'start' })
      expect(
        await commands.captureFeature(
          'onboarding',
          'v12-model-file-list-start',
        ),
      ).toMatch(/\.png$/)
      files.element().lastElementChild!.scrollIntoView({ block: 'end' })
      expect(
        await commands.captureFeature('onboarding', 'v12-model-file-list-end'),
      ).toMatch(/\.png$/)
    }
  })
}

test('V12 engine step details', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('not-prepared')
  view.phase = 'preparing_engine'
  view.preparation = [recordedRuntimePlan()]
  const files = view.preparation[0]!.files
  view.progress = {
    component: 'runtime',
    phase: 'download',
    file_id: files[0]!.file_id,
    bytes_done: 0,
    bytes_total: files[0]!.size,
    resume_offset: 0,
    verification_method: '',
    overall: {
      bytes_done: 0,
      bytes_total: files.reduce((sum, file) => sum + file.size, 0),
    },
  }
  view.preparation_file = { index: 1, total: files.length }
  mountCorePage('/onboarding', view)
  await expect
    .element(page.getByLabelText('Preparation progress'))
    .toBeVisible()
  await page.getByText('Details', { exact: true }).click()
  await document.fonts.ready
  expect(
    await commands.captureFeature('onboarding', 'v12-engine-details'),
  ).toMatch(/\.png$/)
})

test('V12 finished model notice', () => {
  const view = coreView('models-download-in-progress')
  view.models!.download!.state = 'done'
  view.models!.download!.phase = 'verify'
  view.models!.download!.message = 'The model is ready.'
  return capturePage({
    feature: 'models',
    scenario: 'v12-finished-model',
    view,
  })
})
