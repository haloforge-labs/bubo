import { bridgeReply } from '@/components/bridge-test-reply'
import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { bytes, folderPath } from '@/lib/format'
import { actions, subscribeViewEvents } from '@/lib/bindings'
import {
  coreView,
  folderView,
  mountCorePage,
  recordedPlan,
} from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('folder copy uses the containing folder and active identity uses the catalogue name', async () => {
  const view = coreView()
  // The first row of the CLI's models list is the first row on the page.
  const model = view.models!.prepared_models[0]!
  view.active_model = {
    catalogue_id: model.id,
    manifest_path: model.manifest_path,
  }
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const open = vi.spyOn(actions, 'OpenModelFolder').mockResolvedValue({})
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent('In use')
  expect(installed.element().textContent).not.toContain('In use:')
  await installed
    .getByRole('button', { name: 'Copy folder path' })
    .first()
    .click()
  expect(copy).toHaveBeenCalledExactlyOnceWith(folderPath(model.manifest_path!))
  await installed.getByRole('button', { name: 'Open folder' }).first().click()
  expect(open).toHaveBeenCalledExactlyOnceWith(folderPath(model.manifest_path!))
  expect(installed.element().textContent).not.toContain(
    'Memory fit is not reported',
  )
})

test('Open folder preserves the native typed rejection', async () => {
  vi.spyOn(actions, 'OpenModelFolder').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'model_folder_not_allowed',
        remedy: 'Choose a CLI-reported model folder.',
      },
    }),
  )
  mountCorePage('/models', coreView())
  await page.getByRole('button', { name: 'Open folder' }).first().click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Choose a CLI-reported model folder.')
})

function mockDownload() {
  const plan = recordedPlan() as NonNullable<
    Awaited<ReturnType<typeof actions.CheckModel>>['plan']
  >
  const read = vi
    .spyOn(actions, 'Download')
    .mockResolvedValue(bridgeReply({ plan }))
  const confirm = vi
    .spyOn(actions, 'ConfirmDownload')
    .mockResolvedValue(bridgeReply({}))
  const cancel = vi
    .spyOn(actions, 'CancelDownload')
    .mockResolvedValue(bridgeReply({}))
  return { plan, read, confirm, cancel }
}

test('Installed has CLI names, sizes and paths; Discover includes disabled catalogue info', async () => {
  const view = coreView()
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect
    .element(installed)
    .toHaveTextContent(view.models!.prepared_models[0]!.display_name)
  await expect
    .element(installed)
    .toHaveTextContent(
      folderPath(view.models!.prepared_models[0]!.manifest_path!),
    )
  expect(installed.element().textContent).not.toContain('Server alias')
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect.element(discover).toBeVisible()
  // Match the CLI name exactly; the lower-memory name shares its prefix.
  const names = Array.from(
    discover.element().querySelectorAll('tbody tr td:first-child'),
  ).map((cell) => cell.firstChild?.textContent)
  for (const entry of view.models!.catalogue) {
    expect(names).toContain(entry.display_name)
    if (!entry.download_enabled) {
      await expect
        .element(
          discover
            .getByRole('row')
            .filter({ hasText: entry.display_name })
            .getByRole('button', { name: 'Download', exact: true }),
        )
        .toBeDisabled()
    }
  }
})

test('one plan opens before consent; Download confirms once and never reads the plan twice', async () => {
  const { read, confirm, plan } = mockDownload()
  const view = coreView()
  // The ready-idle recording does not name the folder: the consent names the
  // one Go reports.
  view.download_folder = folderView().download_folder
  view.prepared_models = []
  view.models!.prepared_models = []
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  const dialog = page.getByRole('dialog', {
    name: `Download ${coreView().models!.catalogue[0]!.display_name}`,
  })
  await expect.element(dialog).toBeVisible()
  // The space to spare is the CLI's own margin for the tightest volume.
  const spare = plan.volumes.find(
    (volume) => volume.required_bytes === plan.required_bytes,
  )?.safety_margin_bytes
  expect(spare).toBeGreaterThan(0)
  expect(read).toHaveBeenCalledExactlyOnceWith(plan.id)
  expect(confirm).not.toHaveBeenCalled()
  // UX14 T14-01: the consent names the download folder as Settings does, never
  // the package path.
  await expect
    .element(dialog)
    .toHaveTextContent(`Downloads to ${view.download_folder.path}`)
  expect(dialog.element().textContent).not.toContain(
    Object.values(plan.paths)[0]!,
  )
  await expect
    .element(dialog.getByRole('link', { name: plan.licence!.name }))
    .toHaveAttribute('href', plan.licence!.url)
  await expect
    .element(dialog)
    .toHaveTextContent('By downloading you accept the')
  await expect
    .element(dialog)
    .toHaveTextContent(
      `Needs ${bytes(plan.required_bytes)} (${bytes(plan.files.reduce((total, file) => total + file.size, 0))} + ${bytes(spare!, { unit: 'GiB', digits: 0 })} spare) · ${Object.values(plan.paths)[0]!.slice(0, 2)} has ${bytes(plan.free_bytes)} free`,
    )
  await dialog.getByRole('button', { name: 'Download', exact: true }).click()
  expect(confirm).toHaveBeenCalledExactlyOnceWith(plan.id)
  expect(read).toHaveBeenCalledTimes(1)
  await expect.element(dialog).not.toBeInTheDocument()
})

test('Not now cancels the waiting plan without confirming', async () => {
  const { cancel, confirm } = mockDownload()
  const view = coreView()
  view.prepared_models = []
  view.models!.prepared_models = []
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Not now' })
    .click()
  expect(cancel).toHaveBeenCalledExactlyOnceWith()
  expect(confirm).not.toHaveBeenCalled()
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
})

test('insufficient space or absent licence disables acceptance', async () => {
  const { plan, read, confirm } = mockDownload()
  read.mockResolvedValue(
    bridgeReply({
      plan: {
        ...plan,
        required_bytes: plan.free_bytes + 41_000_000,
        enough_space: false,
      } as typeof plan,
    }),
  )
  const view = coreView()
  view.prepared_models = []
  view.models!.prepared_models = []
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  await expect
    .element(
      page
        .getByRole('dialog')
        .getByRole('button', { name: 'Download', exact: true }),
    )
    .toBeDisabled()
  await expect.element(page.getByRole('alert')).toHaveTextContent('Short by')
  expect(confirm).not.toHaveBeenCalled()
})

test('typed confirm failure preserves the dialog, code and remedy', async () => {
  const { confirm } = mockDownload()
  confirm.mockResolvedValue(
    bridgeReply({
      error: {
        code: 'plan_changed',
        remedy: 'Open the plan again.',
        component: '',
        stage: '',
        retryable: true,
      },
    }),
  )
  const view = coreView()
  view.prepared_models = []
  view.models!.prepared_models = []
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Download', exact: true })
    .first()
    .click()
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Download', exact: true })
    .click()
  await expect
    .element(page.getByRole('dialog').getByRole('alert'))
    .toHaveTextContent('plan_changed: Open the plan again.')
  await expect.element(page.getByRole('dialog')).toBeVisible()
})

test('download progress stays outside the dialog and stream updates are rendered', async () => {
  const view = coreView('models-download-in-progress')
  const { client } = mountCorePage('/models', view)
  const receive = new Map<string, (payload: unknown) => void>()
  const stop = subscribeViewEvents(client, (name, callback) => {
    receive.set(name, callback)
    return () => receive.delete(name)
  })
  try {
    const progress = page.getByRole('region', {
      name: 'Model download progress',
    })
    const job = view.models!.download!
    await expect
      .element(progress)
      .toHaveTextContent(
        `${bytes(job.bytes_done)} of ${bytes(job.bytes_total!)}`,
      )
    const download = [...receive.entries()].find(([name]) =>
      name.startsWith('download'),
    )![1]
    const later = job.bytes_done + 1024 ** 3
    download({ ...view.models!.download!, bytes_done: later })
    await expect
      .element(progress)
      .toHaveTextContent(`${bytes(later)} of ${bytes(job.bytes_total!)}`)
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
    const cancel = vi
      .spyOn(actions, 'CancelDownload')
      .mockResolvedValue(bridgeReply({}))
    await page.getByRole('button', { name: 'Cancel download' }).click()
    expect(cancel).toHaveBeenCalledOnce()
  } finally {
    stop()
  }
})

test('failed jobs keep their remedy and Retry reads a fresh plan', async () => {
  const { read } = mockDownload()
  const view = coreView('models-download-in-progress')
  view.models!.download = {
    ...view.models!.download!,
    state: 'failed',
    error: {
      code: 'fixture_download',
      remedy: 'Fixture retry remedy.',
      exit_code: 1,
    },
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture retry remedy.')
  await page.getByRole('button', { name: 'Try again' }).click()
  expect(read).toHaveBeenCalledExactlyOnceWith(view.models!.download.id)
  await expect.element(page.getByRole('dialog')).toBeVisible()
})

test('model file and folder import use browse bindings only and display their results', async () => {
  const file = vi.spyOn(actions, 'BrowseModelFile').mockResolvedValue(
    bridgeReply({
      file: 'D:/fixture/local.gguf',
      folder: 'D:/fixture',
      added: true,
      cancelled: false,
      message: 'Fixture file folder added.',
    }),
  )
  const folder = vi.spyOn(actions, 'BrowseModelFolder').mockResolvedValue(
    bridgeReply({
      file: '',
      folder: 'D:/fixture',
      added: false,
      cancelled: false,
      message: 'Fixture folder already listed.',
    }),
  )
  const download = vi.spyOn(actions, 'Download')
  mountCorePage('/models', coreView())
  await page.getByRole('button', { name: 'Browse model file…' }).click()
  await expect
    .element(page.getByText('Fixture file folder added.'))
    .toBeVisible()
  await page.getByRole('button', { name: 'Browse model folder…' }).click()
  await expect
    .element(page.getByText('Fixture folder already listed.'))
    .toBeVisible()
  expect(file).toHaveBeenCalledOnce()
  expect(folder).toHaveBeenCalledOnce()
  expect(download).not.toHaveBeenCalled()
})

test('Select for Server sends a partial model patch and shows a typed CLI rejection', async () => {
  const view = coreView()
  view.serving = false
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'fixture_rejected',
        remedy: 'Choose a different model.',
      },
    }),
  )
  mountCorePage('/models', view)
  await page.getByRole('button', { name: 'Select for Server' }).first().click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Choose a different model.')
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { manifest_path: view.models!.prepared_models[0]!.manifest_path },
    clear: ['model_path'],
  })
})

test('an awaiting plan is restored after navigating back to Models', async () => {
  const view = coreView('models-download-in-progress')
  view.models!.download = {
    ...view.models!.download!,
    state: 'awaiting_confirm',
    plan: recordedPlan(),
  }
  const confirm = vi
    .spyOn(actions, 'ConfirmDownload')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/models', view)
  await expect
    .element(
      page.getByRole('dialog', {
        name: `Download ${coreView().models!.catalogue[0]!.display_name}`,
      }),
    )
    .toBeVisible()
  expect(confirm).not.toHaveBeenCalled()
})

test('active model identity marks the installed catalogue row despite a custom served name', async () => {
  const view = coreView()
  const model = view.models!.prepared_models[0]!
  view.active_model = {
    catalogue_id: model.id,
    manifest_path: model.manifest_path,
  }
  view.instance!.served_model = 'Custom fixture alias'
  mountCorePage('/models', view)
  const row = page
    .getByRole('row')
    .filter({ hasText: model.display_name })
    .first()
  // E22: the status sits under the name, in the row's first cell.
  await expect.element(row.getByText('In use', { exact: true })).toBeVisible()
})

test('in-use rows do not offer selection and the running server holds the memory', async () => {
  const view = coreView()
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  // "In use" is said once, under the name of the one row the server runs,
  // and that row has no button. The recorded list holds other packages; a
  // running server does not stop them being selected for the next start.
  await expect
    .element(installed.getByRole('button', { name: 'In use', exact: true }))
    .not.toBeInTheDocument()
  const rows = installed.getByRole('row')
  const inUse = rows.filter({ hasText: 'In use' })
  expect(inUse.elements()).toHaveLength(1)
  await expect
    .element(inUse.getByRole('button', { name: 'Select for Server' }))
    .not.toBeInTheDocument()
  expect(installed.element().textContent!.match(/In use/g) ?? []).toHaveLength(
    1,
  )
  await expect
    .element(page.getByRole('region', { name: 'Discover models' }))
    .toHaveTextContent(
      "Bubo's server holds the memory now. Stop it to switch models.",
    )
  expect(document.body.textContent).not.toContain(
    'Not loadable in this version',
  )
})

test('per-file progress uses the exact plan file index without exposing roles', async () => {
  const view = coreView('models-download-in-progress')
  const plan = recordedPlan()
  view.models!.download!.plan = plan
  view.models!.download!.file_id = plan.files[1]!.file_id
  view.models!.download!.file_position = { index: 2, total: plan.files.length }
  // Go's whole-job number is what makes the current file a detail.
  view.models!.download!.overall = { bytes_done: 3, bytes_total: 12 }
  mountCorePage('/models', view)
  const progress = page.getByRole('region', { name: 'Model download progress' })
  await expect
    .element(progress)
    .toHaveTextContent(
      `File 2 of ${plan.files.length}: ${bytes(view.models!.download!.bytes_done)} of ${bytes(view.models!.download!.bytes_total!)}`,
    )
  expect(progress.element().textContent).not.toContain(plan.files[1]!.file_id)
})

test('a new file starts fresh speed observations and never inherits the previous file ETA', async () => {
  const view = coreView('models-download-in-progress')
  const job = view.models!.download!
  const { client, replaceView } = mountCorePage('/models', view)
  job.bytes_done = 1024 ** 2
  job.bytes_total = 100 * 1024 ** 2
  const at = Date.now()
  client.setQueryData(
    [
      'transfer-samples',
      `model-${job.id}-${job.kind}-${job.phase}-${job.file_id}`,
    ],
    [
      { bytes: 0, at: at - 1000 },
      { bytes: job.bytes_done, at },
    ],
  )
  replaceView(structuredClone(view))
  const progress = page.getByRole('region', { name: 'Model download progress' })
  await expect.element(progress).toHaveTextContent('1.0 MiB/s')
  job.file_id = 'synthetic-next-file'
  job.bytes_done = 2
  replaceView(structuredClone(view))
  await expect
    .element(progress)
    .toHaveTextContent('Current file: 2 B of 100.0 MiB')
  expect(progress.element().textContent).not.toMatch(/MiB\/s|left/)
})
