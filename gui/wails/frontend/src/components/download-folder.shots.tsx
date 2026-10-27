import { recordedFolderView } from '@/components/recorded-observations'
import { recordedPackagesView } from '@/components/recorded-observations'
import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  mountCorePage,
  recordedPlan,
  flashShortPlanFixture,
} from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

// E19 shots: the Settings row (default and chosen folder), the model consent with enough space and short on C: with the folder offer,
// and Models with packages from two folders and an imported package.

async function shot(feature: string, scenario: string) {
  await document.fonts.ready
  expect(await commands.captureFeature(feature, scenario)).toMatch(/\.png$/)
}

async function settingsApp() {
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  page
    .getByRole('region', { name: 'Model download folder' })
    .element()
    .scrollIntoView({ block: 'start' })
}

test('Settings download folder: default', async () => {
  document.documentElement.dataset.theme = inject('theme')
  mountCorePage('/settings', recordedFolderView())
  await settingsApp()
  await shot('settings', 'download-folder-default')
})

test('Settings download folder: a chosen folder', async () => {
  document.documentElement.dataset.theme = inject('theme')
  mountCorePage('/settings', recordedFolderView('D:\\Bubo downloads'))
  await settingsApp()
  await shot('settings', 'download-folder-chosen')
})

function consent(chosen: boolean) {
  return (chosen
    ? recordedPlan()
    : structuredClone(flashShortPlanFixture.plan)) as unknown as NonNullable<
    Awaited<ReturnType<typeof actions.CheckModel>>['plan']
  >
}

for (const [scenario, chosen] of [
  ['consent-folder-chosen', true],
  ['consent-short-on-c', false],
] as const) {
  test(`Model consent: ${scenario}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    vi.spyOn(actions, 'Download').mockResolvedValue(
      bridgeReply({ plan: consent(chosen) }),
    )
    const view = recordedFolderView(chosen ? 'selected' : undefined)
    view.prepared_models = []
    view.models!.prepared_models = []
    mountCorePage('/models', view)
    await page
      .getByRole('button', { name: 'Download', exact: true })
      .first()
      .click()
    await expect.element(page.getByRole('dialog')).toBeVisible()
    await shot('models', scenario)
  })
}

test('Models: packages from two folders', async () => {
  document.documentElement.dataset.theme = inject('theme')
  mountCorePage('/models', recordedPackagesView().view)
  await expect
    .element(page.getByRole('region', { name: 'Installed models' }))
    .toBeVisible()
  await shot('models', 'packages-two-folders')
})

test('Models: an imported package and an incomplete one', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const { view, rows } = recordedPackagesView()
  const imported = {
    ...rows[0]!,
    manifest_path: 'E:\\Imported\\flash-next\\bubo-package.json',
    repair_output: undefined,
  }
  const partial = { ...rows[1]!, prepared_state: 'partial' }
  view.models!.prepared_models = [imported, partial]
  view.prepared_models = [imported, partial]
  view.models!.folders = ['E:\\Imported']
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Installed models' }))
    .toBeVisible()
  await shot('models', 'packages-imported-and-incomplete')
})
