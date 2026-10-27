import { bridgeReply } from '@/components/bridge-test-reply'
import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import {
  coreView,
  mountCorePage,
  lowerMemoryShortPlanFixture,
  lowerMemoryPlanFixture,
} from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

for (const name of ['ready-idle', 'models-download-in-progress']) {
  test(`Models: ${name}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    mountCorePage('/models', coreView(name))
    await expect
      .element(page.getByRole('heading', { name: 'Models', exact: true }))
      .toBeVisible()
    await document.fonts.ready
    expect(await commands.captureFeature('models', name)).toMatch(/\.png$/)
    if (name === 'ready-idle') {
      page
        .getByRole('region', { name: 'Discover models' })
        .element()
        .scrollIntoView()
      expect(await commands.captureFeature('models', 'discover')).toMatch(
        /\.png$/,
      )
    }
  })
}

for (const enoughSpace of [true, false]) {
  test(`Models: download plan, enough space ${enoughSpace}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    const recorded = lowerMemoryPlanFixture.plan
    const plan = (enoughSpace
      ? recorded
      : lowerMemoryShortPlanFixture.plan) as unknown as NonNullable<
      Awaited<ReturnType<typeof actions.CheckModel>>['plan']
    >
    vi.spyOn(actions, 'Download').mockResolvedValue(bridgeReply({ plan }))
    const view = coreView()
    view.prepared_models = []
    mountCorePage('/models', view)
    const entry = view.models!.catalogue.find((row) => row.id === plan.id)!
    await page
      .getByRole('row')
      .filter({ hasText: entry.display_name })
      .getByRole('button', { name: 'Download', exact: true })
      .click()
    await expect.element(page.getByRole('dialog')).toBeVisible()
    await document.fonts.ready
    expect(
      await commands.captureFeature(
        'models',
        enoughSpace ? 'download-plan' : 'download-space-short',
      ),
    ).toMatch(/\.png$/)
  })
}

test('Models: synthetic imported local file and missing shards', async () => {
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue(
    bridgeReply({
      error: { code: 'invalid_arguments', remedy: 'Choose a different model.' },
    }),
  )
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  // This selection-write fixture has a CLI fit verdict that permits selection.
  const estimate = view.models!.catalogue.find(
    (entry) => entry.id === view.prepared_models[0]!.id,
  )!.estimate
  estimate.memory_estimate = {
    ...estimate.memory_estimate!,
    fits: true,
    recommended: false,
  }
  const base = {
    path: 'D:/fixture/local.gguf',
    file_name: 'local.gguf',
    folder: 'D:/fixture',
    size_bytes: view.prepared_models[0]!.size_bytes,
    shards_found: 1,
    shards_expected: 1,
    kind: 'gguf',
    runnable: true,
    model_name: 'Synthetic imported model',
    estimate: view.models!.catalogue[0]!.estimate,
  }
  view.models!.scanned = [
    base,
    {
      ...base,
      path: 'D:/fixture/incomplete.gguf',
      file_name: 'incomplete.gguf',
      runnable: false,
      model_name: 'Synthetic incomplete model',
      reason: 'Part 3 of 3 is missing.',
    },
  ]
  view.models!.folders = ['D:/fixture']
  mountCorePage('/models', view)
  await expect.element(page.getByText('Synthetic imported model')).toBeVisible()
  await document.fonts.ready
  expect(await commands.captureFeature('models', 'synthetic-imported')).toMatch(
    /\.png$/,
  )
  await page.getByRole('button', { name: 'Select for Server' }).first().click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('A server setting needs attention.')
  expect(
    await commands.captureFeature('models', 'preferences-rejected'),
  ).toMatch(/\.png$/)
})
