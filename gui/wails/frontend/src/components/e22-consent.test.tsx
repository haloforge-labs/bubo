import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { actions, type View } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  coreView,
  folderView,
  mountCorePage,
  recordedPlan,
  recordedRuntimePlan,
} from '@/components/core-page-test-support'
import { PlanSummary } from '@/components/model-download'
import { shortByShown, bytes } from '@/components/model-facts'

afterEach(() => vi.restoreAllMocks())

// E22 fixes 1 and 2 (UX15 K15-01 = T15-01, T15-02, T15-03).
//
// Failure list, written before the code:
//  1a. Settings, server running, folder path not known: the row prints
//      "(default)" alone.                       -> the row prints no label.
//  1b. The consent, folder path not known: "Downloads to the download folder".
//                                              -> the clause is left out.
//  1c. The consent, folder path known: the path shares its line with the
//      space figures (and wraps over four lines).
//                                              -> the path is alone on a line.
//  2a. The consent prints "To download", "Free space needed ... (includes 2 GiB
//      to spare)" and "C: has ..." on separate lines.
//                                              -> one line: "Needs 110.6 GiB
//      (108.6 GiB + 2 GiB spare) · C: has 121.6 GiB free".
//  2b. The shortfall is computed from the raw bytes, so it can disagree with
//      the figures printed (110.6 needed, 100.0 free, "Short by 10.7").
//                                              -> computed from the figures shown.
//  2c. "Short by" shows only when the CLI says the space is not enough.
const GiB = 1024 ** 3

function plan(options: { folder: string; free: number; required?: number }) {
  const recorded = recordedPlan(true)
  const drive = options.folder.slice(0, 2) + '\\'
  const required = options.required ?? recorded.required_bytes
  return {
    ...recorded,
    drive,
    enough_space: options.free >= required,
    free_bytes: options.free,
    required_bytes: required,
    paths: { model: `${options.folder}\\models\\${recorded.id}` },
    volumes: recorded.volumes.map((volume) => ({
      ...volume,
      name: drive,
      free_bytes: options.free,
      required_bytes: required,
    })),
  } as NonNullable<Awaited<ReturnType<typeof actions.CheckModel>>['plan']>
}

async function openConsent(view: View, shown: ReturnType<typeof plan>) {
  view.prepared_models = []
  view.models!.prepared_models = []
  vi.spyOn(actions, 'Download').mockResolvedValue(bridgeReply({ plan: shown }))
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  return page.getByRole('dialog')
}

test('1a: Settings never prints "(default)" alone', async () => {
  // The running scenario: Go has not named the folder (path is empty).
  const view = coreView('ready-idle')
  expect(view.download_folder.path).toBe('')
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  const row = page.getByRole('region', { name: 'Model download folder' })
  await expect.element(row).toBeVisible()
  expect(row.element().textContent).not.toContain('(default)')
  // The lock reason stays: it explains the disabled button.
  await expect.element(row).toHaveTextContent("can't change while")
})

test('1a: with the path known the row prints it with "(default)", also while running', async () => {
  const view = folderView()
  view.download_folder.change = {
    allowed: false,
    reason_code: 'download_folder_locked',
    reason: 'The server is running. Stop it to change the download folder.',
  }
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  const row = page.getByRole('region', { name: 'Model download folder' })
  await expect.element(row).toHaveTextContent('Bubo\\models')
  await expect.element(row).toHaveTextContent('(default)')
})

test('1b: the consent leaves the "Downloads to" clause out when the folder is not known', async () => {
  const view = coreView()
  view.download_folder = {
    ...folderView().download_folder,
    path: '',
  }
  const dialog = await openConsent(
    view,
    plan({ folder: 'C:\\Users\\fixture\\Bubo', free: 892 * GiB }),
  )
  await expect.element(dialog).toHaveTextContent('Needs')
  expect(dialog.element().textContent).not.toContain('Downloads to')
  expect(dialog.element().textContent).not.toContain('the download folder')
})

test('1c, 2a: the path is alone on its line, then one numbers line', async () => {
  const dialog = await openConsent(
    folderView('D:\\Bubo downloads'),
    plan({ folder: 'D:\\Bubo downloads', free: 892 * GiB }),
  )
  const pathLine = dialog.getByText('Downloads to').element().closest('p')!
  expect(pathLine.textContent).toBe('Downloads to D:\\Bubo downloads')
  const numbers = pathLine.nextElementSibling!
  expect(numbers.textContent).toBe(
    'Needs 110.6 GiB (108.6 GiB + 2 GiB spare) · D: has 892.0 GiB free',
  )
  // The same facts are not printed a second time.
  const text = dialog.element().textContent!
  expect(text).not.toContain('Free space needed')
  expect(text).not.toContain('To download')
  expect(text.match(/has 892\.0 GiB free/g)).toHaveLength(1)
  expect(dialog.getByRole('alert').elements()).toHaveLength(0)
  expect(text).not.toContain('Short by')
})

test('2a: when the parts do not add up to the total, the line says "includes" and keeps the download figure', async () => {
  const shown = plan({
    folder: 'D:\\Bubo downloads',
    free: 892 * GiB,
    required: 118 * GiB,
  })
  const dialog = await openConsent(folderView('D:\\Bubo downloads'), shown)
  await expect
    .element(dialog)
    .toHaveTextContent(
      'Needs 118.0 GiB (includes 2 GiB spare) · D: has 892.0 GiB free',
    )
  await expect.element(dialog).toHaveTextContent('To download: 108.6 GiB')
})

test('2b, 2c: the shortfall is computed from the figures shown', async () => {
  // 110.649 GiB needed prints "110.6"; 100.0 free. The raw difference rounds up
  // to 10.7; the figures on screen give 10.6.
  const dialog = await openConsent(
    folderView(),
    plan({
      folder: 'C:\\Users\\fixture\\Bubo',
      free: 100 * GiB,
      required: Math.round(110.649 * GiB),
    }),
  )
  await expect
    .element(dialog)
    .toHaveTextContent('Needs 110.6 GiB (108.6 GiB + 2 GiB spare)')
  await expect.element(dialog).toHaveTextContent('C: has 100.0 GiB free')
  await expect
    .element(dialog.getByRole('alert'))
    .toHaveTextContent('Short by 10.6 GiB.')
})

test('2b: shortByShown subtracts the printed figures and never reads 0.0', () => {
  expect(shortByShown(Math.round(110.649 * GiB), 100 * GiB)).toBe('10.6 GiB')
  expect(shortByShown(110.6 * GiB, 100 * GiB)).toBe('10.6 GiB')
  // Equal when printed, but the CLI says short: the smallest figure that reads.
  expect(shortByShown(100.04 * GiB, 100.01 * GiB)).toBe('0.1 GiB')
  // Small plans print in MiB.
  expect(shortByShown(500 * 1024 ** 2, 300 * 1024 ** 2)).toBe('200.0 MiB')
})

test('2a: the engine plan keeps "includes unpacking" in the same one-line shape', async () => {
  const runtime = recordedRuntimePlan() as unknown as Parameters<
    typeof PlanSummary
  >[0]['plan']
  render(<PlanSummary plan={runtime} />)
  await expect
    .element(page.getByText(/^Needs [\d,.]+ GiB \(includes unpacking\) · /))
    .toHaveTextContent(
      `Needs ${bytes(runtime.required_bytes)} (includes unpacking) · `,
    )
  expect(runtime.files.map((file) => file.size)).toEqual([
    758050981, 116831755, 139346623,
  ])
  expect(document.body.textContent).not.toContain('Free space needed')
})
