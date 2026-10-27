import { recordedPackagesView } from '@/components/recorded-observations'
import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  coreView,
  mountCorePage,
  recordedPlan,
  flashShortPlanFixture,
} from '@/components/core-page-test-support'
import { capturePage } from '@/components/page-capture'
import { fixture } from '@/components/test-view'

// E21 coverage: screens the recordings of the candidate exe make real. The
// views are the scenarios Go built from those recordings. Where a test sets a
// fact itself (the short drive, the second copy) it says so.
afterEach(() => vi.restoreAllMocks())

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

// The two real CLI previews show the full default package on S: and C:.
function defaultModelPlan(short = false) {
  return short
    ? ({
        ...structuredClone(flashShortPlanFixture.plan),
        licence: structuredClone(flashShortPlanFixture.plan.licence),
      } as unknown as ReturnType<typeof recordedPlan>)
    : recordedPlan()
}

async function openDefaultConsent(plan: ReturnType<typeof defaultModelPlan>) {
  const view = coreView('runtime-ready-model-missing')
  const modelPath = plan.paths.model!
  view.download_folder = {
    ...view.download_folder,
    path: modelPath.slice(0, modelPath.lastIndexOf('\\models\\')),
    drive: modelPath.slice(0, 2),
    free_bytes: plan.free_bytes,
  }
  vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: plan as NonNullable<
        Awaited<ReturnType<typeof actions.CheckModel>>['plan']
      >,
    }),
  )
  // From Get started, as a stranger reaches it (E22 E-5).
  mountCorePage('/onboarding', view)
  await page
    .getByRole('button', { name: 'Download model…', exact: true })
    .click()
  return page.getByRole('dialog')
}

test('E21 default model download dialog: default folder', async () => {
  const dialog = await openDefaultConsent(defaultModelPlan())
  await expect.element(dialog).toHaveTextContent('Needs')
  await expect
    .element(dialog)
    .toHaveTextContent(
      Object.values(defaultModelPlan().paths)[0]!.split('\\models\\')[0]!,
    )
  await expect.element(dialog).not.toHaveTextContent('Short by')
  await shot('models', 'e21-consent-default-folder')
})

test('E21 default model download dialog: short on C:', async () => {
  // The recording has room; the test lowers the free space of the drive.
  const plan = defaultModelPlan(true)
  const dialog = await openDefaultConsent(plan)
  await expect.element(dialog.getByRole('alert')).toHaveTextContent('Short by')
  await expect
    .element(dialog.getByRole('button', { name: 'Change folder…' }))
    .toBeEnabled()
  await shot('models', 'e21-consent-short-on-c')
})

test('E21 Settings download folder row is disabled while the server runs', async () => {
  // The running scenario's own View decides: Go reports the folder as locked.
  const view = coreView('ready-idle')
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  const row = page.getByRole('region', { name: 'Model download folder' })
  await expect
    .element(row.getByRole('button', { name: 'Change…' }))
    .toBeDisabled()
  await expect
    .element(row)
    .toHaveTextContent(
      "The download folder can't change while the server or a download is running.",
    )
  await shot('settings', 'e21-download-folder-locked')
})

test('E21 Models: the incomplete package row, in view, with its repair action', () =>
  capturePage({
    feature: 'models',
    scenario: 'e21-incomplete-row',
    view: fixture('stopped'),
    prepare: async () => {
      const installed = page.getByRole('region', { name: 'Installed models' })
      const row = installed.getByRole('row').filter({ hasText: 'Incomplete' })
      await expect.element(row).toBeVisible()
      // The recorded Qwen3.6 package lost a file and sits in a prepare layout,
      // so the CLI names the folder that repairs it (repair_output).
      await expect
        .element(row.getByRole('button', { name: 'Check or repair files' }))
        .toBeEnabled()
      row.element().scrollIntoView({ block: 'center' })
    },
  }))

test('E21 Models: one package with its fit resolved by the CLI estimate', () =>
  capturePage({
    feature: 'models',
    scenario: 'e21-package-fit',
    view: fixture('stopped-saved-estimate'),
    prepare: async () => {
      const installed = page.getByRole('region', { name: 'Installed models' })
      await expect.element(installed).toHaveTextContent('Fits')
      installed.element().scrollIntoView({ block: 'start' })
    },
  }))

test('E21 Server: two copies of one package are told apart in the Model list', async () => {
  // The recorded list holds each package once; the test adds the second copy
  // by changing the folder of a copy of the row (packagesView).
  const { view } = recordedPackagesView()
  mountCorePage('/', view)
  const options = page
    .getByRole('combobox', { name: 'Model' })
    .getByRole('option')
  await expect.element(options.first()).toBeInTheDocument()
  const labels = options.elements().map((option) => option.textContent ?? '')
  expect(labels).toHaveLength(2)
  expect(new Set(labels).size).toBe(2)
  // E22 (K15-03, T15-10): show the choices as a list. A closed native select cannot be captured
  // open; size=2 draws the same options in the same element.
  const select = document.getElementById('server-model') as HTMLSelectElement
  select.size = 2
  await shot('server', 'e21-model-two-copies')
})
