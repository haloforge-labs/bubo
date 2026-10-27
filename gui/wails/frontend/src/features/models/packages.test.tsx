import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  coreView,
  mountCorePage,
  packagesView,
} from '@/components/core-page-test-support'
import { modelChoices } from '@/components/model-facts'

afterEach(() => vi.restoreAllMocks())

// E19: Models lists every package the CLI reports in prepared_models, each with
// its own folder. The GUI keeps no list of models and no paths of its own.

test('the same package in two folders is two rows, each with its own folder', async () => {
  const { view } = packagesView()
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect
    .element(installed)
    .toHaveTextContent('AppData\\Local\\Bubo\\models')
  await expect
    .element(installed)
    .toHaveTextContent('D:\\Bubo downloads\\models')
  expect(
    installed.getByRole('button', { name: 'Open folder' }).elements(),
  ).toHaveLength(2)
})

test('Select for Server picks the package of that row by its manifest path', async () => {
  const { view, rows } = packagesView()
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  const buttons = installed.getByRole('button', { name: 'Select for Server' })
  await buttons.nth(1).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: { manifest_path: rows[1]!.manifest_path },
    clear: ['model_path'],
  })
})

test('only the row that runs is in use when two folders hold the same package', async () => {
  const { view, rows } = packagesView()
  view.state = 'Ready'
  view.serving = true
  view.owned_server = true
  view.current_model = {
    ...view.current_model!,
    manifest_path: rows[1]!.manifest_path!,
  }
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  const rowsOnScreen = installed.getByRole('row')
  // header + two package rows
  expect(rowsOnScreen.elements()).toHaveLength(3)
  await expect.element(rowsOnScreen.nth(1)).not.toHaveTextContent('In use')
  await expect.element(rowsOnScreen.nth(2)).toHaveTextContent('In use')
})

test('an incomplete package is listed as incomplete and cannot be selected', async () => {
  const { view, rows } = packagesView()
  const partial = { ...rows[1]!, prepared_state: 'partial' }
  view.models!.prepared_models = [rows[0]!, partial]
  view.prepared_models = [rows[0]!, partial]
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent('Incomplete')
  await expect
    .element(
      installed.getByRole('button', { name: 'Select for Server' }).nth(1),
    )
    .toBeDisabled()
  // Never a start choice for the Server page either.
  expect(modelChoices(view).map((choice) => choice.path)).toEqual([
    rows[0]!.manifest_path,
  ])
})

test('an imported package folder is one package row, not its shards', async () => {
  const { view, rows } = packagesView()
  const imported = {
    ...rows[0]!,
    manifest_path: 'E:\\Imported\\flash-next\\bubo-package.json',
    repair_output: undefined,
  }
  view.models!.prepared_models = [imported]
  view.prepared_models = [imported]
  view.models!.scanned = []
  view.models!.folders = ['E:\\Imported']
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  expect(installed.getByRole('row').elements()).toHaveLength(2)
  await expect.element(installed).toHaveTextContent('E:\\Imported\\flash-next')
  expect(installed.element().textContent).not.toContain('.gguf')
})

test('the server estimate of a package uses its folder, wherever it is', () => {
  const { view, rows } = packagesView()
  const choices = modelChoices(view)
  expect(choices.map((choice) => choice.subject)).toEqual([
    'C:\\Users\\fixture\\AppData\\Local\\Bubo\\models\\' + rows[0]!.id,
    'D:\\Bubo downloads\\models\\' + rows[1]!.id,
  ])
})

test('Discover calls a model Installed when any folder has it', async () => {
  const { view, rows } = packagesView()
  // Only the second folder has it; the default catalogue state says not downloaded.
  view.models!.prepared_models = [rows[1]!]
  view.prepared_models = [rows[1]!]
  for (const entry of view.models!.catalogue)
    if (entry.id === rows[1]!.id) entry.prepared_state = 'not_prepared'
  mountCorePage('/models', view)
  const discover = page.getByRole('region', { name: 'Discover models' })
  await expect.element(discover).toHaveTextContent('Installed')
})

test('Check or repair acts where the row says, and only a row with a repair output has it', async () => {
  const { view, rows } = packagesView()
  const imported = {
    ...rows[0]!,
    manifest_path: 'E:\\Imported\\flash-next\\bubo-package.json',
    repair_output: undefined,
  }
  view.models!.prepared_models = [rows[1]!, imported]
  view.prepared_models = [rows[1]!, imported]
  const plan = vi
    .spyOn(actions, 'RepairPackage')
    .mockResolvedValue(bridgeReply({ error: { code: 'busy', remedy: 'x' } }))
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  const buttons = installed.getByRole('button', {
    name: 'Check or repair files',
  })
  expect(buttons.elements()).toHaveLength(1)
  await buttons.click()
  expect(plan).toHaveBeenCalledExactlyOnceWith(
    rows[1]!.id,
    'D:\\Bubo downloads',
  )
})

test('a part file is not a model: sidecar rows are not listed as installed or incomplete', async () => {
  const view = coreView('stopped')
  // The recorded list's incomplete package is left out: this test is about the
  // part file alone.
  view.models!.prepared_models = view.models!.prepared_models.filter(
    (row) => row.prepared_state === 'prepared',
  )
  view.models!.scanned = [
    {
      path: 'E:\\Parts\\mmproj.gguf',
      file_name: 'mmproj.gguf',
      folder: 'E:\\Parts',
      size_bytes: 1,
      shards_found: 1,
      shards_expected: 1,
      kind: 'sidecar',
      runnable: false,
    },
  ] as unknown as NonNullable<typeof view.models>['scanned']
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  expect(installed.element().textContent).not.toContain('mmproj.gguf')
  expect(installed.element().textContent).not.toContain('Incomplete')
})

test('a package with a vision projector says it reads images', async () => {
  const { view, rows } = packagesView()
  const withVision = {
    ...rows[0]!,
    capabilities: { ...rows[0]!.capabilities, mmproj: true },
  }
  const without = {
    ...rows[1]!,
    capabilities: { ...rows[1]!.capabilities, mmproj: false },
  }
  view.models!.prepared_models = [withVision, without]
  view.prepared_models = [withVision, without]
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed.getByText('Reads images')).toBeVisible()
  expect(installed.getByText('Reads images').elements()).toHaveLength(1)
})

test('the memory fit line belongs to the one package that is current, not to every copy', async () => {
  const { view, rows } = packagesView()
  view.fit = { ...view.fit, class: 'fits', sentence: 'Fits fixture sentence.' }
  view.current_model = {
    ...view.current_model!,
    manifest_path: rows[1]!.manifest_path!,
  }
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  expect(
    installed.element().textContent!.split('Fits fixture sentence.').length - 1,
  ).toBe(1)
})

// UX14 T14-03 / N14-02 (Krug): each row's fit comes from its own estimate, by
// that row's folder; a row that cannot be selected shows no fit text.
test('each selectable row shows the fit of its own folder; an incomplete row shows none', async () => {
  const { view, rows } = packagesView()
  const third = {
    ...rows[1]!,
    manifest_path: 'E:\\Third\\bubo-package.json',
    prepared_state: 'partial',
  }
  view.models!.prepared_models = [rows[0]!, rows[1]!, third]
  view.prepared_models = view.models!.prepared_models
  // The saved setup belongs to none of these rows.
  view.current_model = {
    ...view.current_model!,
    catalogue_id: '',
    manifest_path: '',
  }
  const folder = (path: string) => path.replace(/[\\/][^\\/]*$/, '')
  const estimate = vi
    .spyOn(actions, 'Estimate')
    .mockImplementation(async (subject: string) =>
      bridgeReply({
        estimate: {},
        fit: { sentence: `Fits at ${subject.slice(0, 1)}.` },
      }),
    )
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  await expect.element(installed).toHaveTextContent('Fits at C.')
  await expect.element(installed).toHaveTextContent('Fits at D.')
  const subjects = estimate.mock.calls.map((call) => call[0])
  expect(subjects).toContain(folder(rows[0]!.manifest_path!))
  expect(subjects).toContain(folder(rows[1]!.manifest_path!))
  // The incomplete row is never estimated and shows no fit text.
  expect(subjects.some((subject) => subject.startsWith('E:'))).toBe(false)
  expect(installed.element().textContent).not.toContain('Fits at E')
})

// UX14 T14-04: the row actions are borderless text on one line. E21w W6: they
// share a line where the window has room for them (1400 px wide); in a narrower
// window they wrap rather than run out of the box (at 900 px they need two
// lines). installed-fit.test.tsx checks the fit at the default and the minimum
// window.
test('the Installed row actions are borderless and share one line', async () => {
  await page.viewport(1400, 900)
  try {
    await installedActionsShareOneLine()
  } finally {
    await page.viewport(1100, 760) // vitest.config.ts size
  }
})

async function installedActionsShareOneLine() {
  const { view } = packagesView()
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  const first = installed.getByRole('row').nth(1)
  const names = ['Open folder', 'Copy folder path', 'Check or repair files']
  const tops = names.map((name) => {
    const button = first.getByRole('button', { name }).element()
    expect(getComputedStyle(button).borderTopWidth).toBe('0px')
    return Math.round(button.getBoundingClientRect().top)
  })
  expect(tops, `tops of ${names.join(', ')}`).toEqual([
    tops[0],
    tops[0],
    tops[0],
  ])
}

// UX14 N14-01 (Tufte): one word for an incomplete package and one size rule.
test('an incomplete package, prepared or loose, says Incomplete and shows no size', async () => {
  const { view, rows } = packagesView()
  const partial = { ...rows[1]!, prepared_state: 'partial' }
  view.models!.prepared_models = [rows[0]!, partial]
  view.prepared_models = view.models!.prepared_models
  view.models!.scanned = [
    {
      path: 'E:\\Loose\\half.gguf',
      file_name: 'half.gguf',
      folder: 'E:\\Loose',
      size_bytes: 5 * 1024 ** 3,
      shards_found: 1,
      shards_expected: 3,
      kind: 'model',
      runnable: false,
      reason: 'Shards are missing.',
    },
  ] as unknown as NonNullable<typeof view.models>['scanned']
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toBeVisible()
  const text = installed.element().textContent!
  expect(text).not.toContain('Files incomplete')
  expect(text.split('Incomplete').length - 1).toBe(2)
  expect(text).not.toContain('5.0 GiB')
  const cells = installed.getByRole('row').elements()
  expect(cells.slice(2).every((row) => row.textContent!.includes('—'))).toBe(
    true,
  )
})
