import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { actions, type View } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  folderView as stoppedView,
  mountCorePage,
  recordedPlan,
  recordedRuntimePlan,
} from '@/components/core-page-test-support'
import { downloadFolderHelp } from '@/components/download-folder'
import { PlanSummary } from '@/components/model-download'

afterEach(() => vi.restoreAllMocks())

// E19: the download folder decides only where NEW downloads go. These tests
// were written before the code; the fixtures set the View facts Go reports.
const GiB = 1024 ** 3

async function openAppSettings() {
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  return page.getByRole('region', { name: 'Model download folder' })
}

test('Settings shows the default download folder, its free space and the help', async () => {
  mountCorePage('/settings', stoppedView())
  const row = await openAppSettings()
  await expect.element(row).toHaveTextContent('Bubo\\models')
  await expect.element(row).toHaveTextContent('(default)')
  await expect.element(row).toHaveTextContent('C: has 41.0 GiB free')
  await expect.element(row).toHaveTextContent(downloadFolderHelp)
  // One line about the engine files lives in the Runtime folder row, not here.
  expect(row.element().textContent).not.toContain('kept apart')
  await expect
    .element(row.getByRole('button', { name: 'Change…' }))
    .toBeEnabled()
  // The default is already in use, so there is nothing to return to.
  expect(
    row.getByRole('button', { name: 'Use default' }).elements(),
  ).toHaveLength(0)
})

// UX14 T14-05 / K14-04: the path and "(default)" in every state; the free-space
// line only when the figure is known (never "not read yet").
test('Settings prints the default path and leaves the free-space line out until it is known', async () => {
  const view = stoppedView()
  view.download_folder.free_bytes = undefined
  view.download_folder.drive = ''
  mountCorePage('/settings', view)
  const row = await openAppSettings()
  await expect.element(row).toHaveTextContent('Bubo\\models')
  await expect.element(row).toHaveTextContent('(default)')
  expect(row.element().textContent).not.toContain('not read')
  expect(row.element().textContent).not.toContain('free')
  expect(row.element().textContent).not.toContain('Default location')
})

// UX14 K14-05 / N14-04: "Use default" is the same secondary button as "Change…".
test('Use default looks like Change…: the same bordered secondary button', async () => {
  mountCorePage('/settings', stoppedView('D:\\Bubo downloads'))
  const row = await openAppSettings()
  const change = row.getByRole('button', { name: 'Change…' }).element()
  const reset = row.getByRole('button', { name: 'Use default' }).element()
  const border = (element: Element) => {
    const style = getComputedStyle(element)
    return `${style.borderTopWidth} ${style.borderTopStyle}`
  }
  expect(border(reset)).toBe(border(change))
  expect(border(reset)).toContain('solid')
})

test('Settings shows a chosen folder with Use default beside Change', async () => {
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/settings', stoppedView('D:\\Bubo downloads'))
  const row = await openAppSettings()
  await expect.element(row).toHaveTextContent('D:\\Bubo downloads')
  await expect.element(row).toHaveTextContent('D: has 892.0 GiB free')
  await row.getByRole('button', { name: 'Use default' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: ['data_folder'],
  })
})

test('Change opens the picker; a cancelled picker changes nothing and says nothing', async () => {
  const browse = vi
    .spyOn(actions, 'BrowseDownloadFolder')
    .mockResolvedValue(
      bridgeReply({ cancelled: true, file: '', folder: '', added: false }),
    )
  mountCorePage('/settings', stoppedView())
  const row = await openAppSettings()
  await row.getByRole('button', { name: 'Change…' }).click()
  expect(browse).toHaveBeenCalledOnce()
  expect(page.getByRole('alert').elements()).toHaveLength(0)
})

test('Change shows the typed refusal of an unusable folder', async () => {
  vi.spyOn(actions, 'BrowseDownloadFolder').mockResolvedValue(
    bridgeReply({
      error: { code: 'download_folder_unwritable', remedy: 'access is denied' },
    }),
  )
  mountCorePage('/settings', stoppedView())
  const row = await openAppSettings()
  await row.getByRole('button', { name: 'Change…' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent("Bubo can't write to this folder.")
})

test('while the server or a download runs the folder cannot change and the reason shows', async () => {
  const view = stoppedView('D:\\Bubo downloads')
  view.download_folder.change = {
    allowed: false,
    reason_code: 'download_folder_locked',
    reason: 'The server is running. Stop it to change the download folder.',
  }
  const browse = vi.spyOn(actions, 'BrowseDownloadFolder')
  mountCorePage('/settings', view)
  const row = await openAppSettings()
  await expect
    .element(row.getByRole('button', { name: 'Change…' }))
    .toBeDisabled()
  await expect
    .element(row.getByRole('button', { name: 'Use default' }))
    .toBeDisabled()
  await expect
    .element(row)
    .toHaveTextContent(
      "The download folder can't change while the server or a download is running.",
    )
  expect(browse).not.toHaveBeenCalled()
})

// ---------------------------------------------------------------------------
// The model consent

function consentPlan(folder: string, free: number, required: number) {
  const recorded = recordedPlan(true)
  // The CLI names a drive with its root; the screen writes it "C:".
  const drive = folder.slice(0, 2) + '\\'
  return {
    ...recorded,
    drive,
    enough_space: free >= required,
    free_bytes: free,
    required_bytes: required,
    paths: { model: `${folder}\\models\\${recorded.id}` },
    volumes: recorded.volumes.map((volume) => ({
      ...volume,
      name: drive,
      free_bytes: free,
      required_bytes: required,
    })),
  } as NonNullable<Awaited<ReturnType<typeof actions.CheckModel>>['plan']>
}

async function openConsent(view: View) {
  const view2 = view
  view2.prepared_models = []
  view2.models!.prepared_models = []
  mountCorePage('/models', view2)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  return page.getByRole('dialog')
}

test('the consent shows where the files go and the space there, in plain view', async () => {
  const view = stoppedView('D:\\Bubo downloads')
  vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: consentPlan('D:\\Bubo downloads', 892 * GiB, 110.6 * GiB),
    }),
  )
  const dialog = await openConsent(view)
  // UX14 T14-01: the folder as Settings names it, not the package path.
  await expect
    .element(dialog)
    .toHaveTextContent('Downloads to D:\\Bubo downloads')
  await expect.element(dialog).toHaveTextContent('D: has 892.0 GiB free')
  expect(dialog.element().textContent).not.toContain('\\models\\')
  expect(dialog.element().textContent).not.toContain('D:\\ has')
  // Not inside a closed disclosure.
  expect(
    dialog
      .getByText('Download folder', { exact: true })
      .elements()
      .filter((element) => element.tagName === 'SUMMARY'),
  ).toHaveLength(0)
  await expect
    .element(dialog.getByRole('button', { name: 'Change folder…' }))
    .toBeEnabled()
})

test('a short drive offers the folder change next to freeing space', async () => {
  const view = stoppedView()
  vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: consentPlan('C:\\Users\\fixture\\Bubo', 41 * GiB, 110.6 * GiB),
    }),
  )
  const dialog = await openConsent(view)
  await expect
    .element(dialog.getByRole('alert'))
    .toHaveTextContent(
      /Short by .* Free up space on C:, or choose another folder\./,
    )
  await expect
    .element(dialog.getByRole('button', { name: 'Change folder…' }))
    .toBeEnabled()
  await expect
    .element(dialog.getByRole('button', { name: 'Download', exact: true }))
    .toBeDisabled()
})

test('Change folder saves the choice and plans again, so the space line and the shortfall follow', async () => {
  const view = stoppedView()
  const download = vi
    .spyOn(actions, 'Download')
    .mockResolvedValueOnce(
      bridgeReply({
        plan: consentPlan('C:\\Users\\fixture\\Bubo', 41 * GiB, 110.6 * GiB),
      }),
    )
    .mockResolvedValueOnce(
      bridgeReply({
        plan: consentPlan('D:\\Big drive', 892 * GiB, 110.6 * GiB),
      }),
    )
  const browse = vi
    .spyOn(actions, 'BrowseDownloadFolder')
    .mockResolvedValue(
      bridgeReply({ cancelled: false, folder: 'D:\\Big drive' }),
    )
  const dialog = await openConsent(view)
  await expect.element(dialog.getByRole('alert')).toBeVisible()
  await dialog.getByRole('button', { name: 'Change folder…' }).click()
  expect(browse).toHaveBeenCalledOnce()
  expect(download).toHaveBeenCalledTimes(2)
  await expect.element(dialog).toHaveTextContent('Downloads to D:\\Big drive')
  await expect.element(dialog).toHaveTextContent('D: has 892.0 GiB free')
  expect(dialog.getByRole('alert').elements()).toHaveLength(0)
  await expect
    .element(dialog.getByRole('button', { name: 'Download', exact: true }))
    .toBeEnabled()
})

test('a cancelled picker in the consent keeps the plan and plans nothing again', async () => {
  const view = stoppedView()
  const download = vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: consentPlan('C:\\Users\\fixture\\Bubo', 41 * GiB, 110.6 * GiB),
    }),
  )
  vi.spyOn(actions, 'BrowseDownloadFolder').mockResolvedValue(
    bridgeReply({ cancelled: true, folder: '' }),
  )
  const dialog = await openConsent(view)
  await dialog.getByRole('button', { name: 'Change folder…' }).click()
  expect(download).toHaveBeenCalledOnce()
})

test('the consent cannot change the folder while a download or the server runs', async () => {
  const view = stoppedView()
  view.download_folder.change = {
    allowed: false,
    reason_code: 'download_folder_locked',
    reason: 'A model download is running.',
  }
  vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: consentPlan('C:\\Users\\fixture\\Bubo', 892 * GiB, 110.6 * GiB),
    }),
  )
  const dialog = await openConsent(view)
  const change = dialog.getByRole('button', { name: 'Change folder…' })
  await expect.element(change).toBeDisabled()
  // UX14 T14-02 / K14-02: the lock line sits right under the disabled button.
  const line = dialog
    .getByText(
      "The download folder can't change while the server or a download is running.",
    )
    .element()
  expect(
    change.element().parentElement!.contains(line) ||
      change.element().nextElementSibling === line,
  ).toBe(true)
})

test('a short drive with a locked folder does not offer to choose another folder', async () => {
  const view = stoppedView()
  view.download_folder.change = {
    allowed: false,
    reason_code: 'download_folder_locked',
    reason: 'A model download is running.',
  }
  vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: consentPlan('C:\\Users\\fixture\\Bubo', 41 * GiB, 110.6 * GiB),
    }),
  )
  const dialog = await openConsent(view)
  await expect
    .element(dialog.getByRole('alert'))
    .toHaveTextContent(/Short by .* Free up space on C: and try again\./)
  expect(dialog.getByRole('alert').element().textContent).not.toContain(
    'choose another folder',
  )
})

// UX14 N14-03 (Krug): the engine consent names its folder "Engine folder", and
// it grows no folder control: the engine files do not follow the download folder.
test('the engine plan has an Engine folder disclosure and no folder control', async () => {
  const plan = recordedRuntimePlan() as unknown as Parameters<
    typeof PlanSummary
  >[0]['plan']
  render(<PlanSummary plan={plan} />)
  await expect.element(page.getByText('Engine folder')).toBeVisible()
  expect(page.getByText('Download folder').elements()).toHaveLength(0)
  expect(
    page.getByRole('button', { name: 'Change folder…' }).elements(),
  ).toHaveLength(0)
})
