import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'

afterEach(() => vi.restoreAllMocks())

// The Draft file and Image file rows: shown for a plain model file, saved as the
// CLI's own --mtp-model / --mmproj settings, absent for a package model.
function plainFileView() {
  const view = coreView('stopped-saved-estimate')
  const template = view.switches.find(
    (row) => row.name === '--reasoning-effort',
  )!
  const override = view.switch_overrides.find(
    (row) => row.name === '--reasoning-effort',
  )!
  for (const name of ['--mtp-model', '--mmproj']) {
    view.switches.push({
      ...template,
      name,
      label: name,
      group: 'Speed',
      audience: 'advanced',
      gui_control: true,
      allowed: {
        kind: 'object',
        properties: { type: { kind: 'string', text: 'string' } },
      },
    })
    view.switch_overrides.push({ ...override, name })
  }
  view.saved.model_path = 'S:\\models\\plain-00001-of-00002.gguf'
  view.saved.manifest_path = undefined
  view.saved.switch_overrides = {}
  return view
}

test('a plain model file offers a draft file and an image file and saves the choice', async () => {
  const view = plainFileView()
  const browse = vi.spyOn(actions, 'BrowseSideFile').mockResolvedValue({
    cancelled: false,
    file: 'C:\\side\\mtp.gguf',
  } as never)
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('group', { name: 'Files and cache' }))
    .toBeVisible()
  await expect
    .element(page.getByText('MTP draft file', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('mmproj file', { exact: true }))
    .toBeVisible()
  // One grid row, two cells: the draft file left, the mmproj file right.
  const cell = (name: string) =>
    document.querySelector(`[data-setting="${name}"]`)!.getBoundingClientRect()
  const draft = cell('--mtp-model')
  const image = cell('--mmproj')
  expect(image.top).toBe(draft.top)
  expect(image.left).toBeGreaterThan(draft.right - 1)
  await page.getByRole('button', { name: 'Browse for MTP draft file' }).click()
  expect(browse).toHaveBeenCalledExactlyOnceWith('draft')
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: '--mtp-model', value: 'C:\\side\\mtp.gguf' }],
    clear: [],
  })
})

test('a cancelled dialog saves nothing; a chosen file can be cleared', async () => {
  const view = plainFileView()
  view.saved.switch_overrides = { '--mmproj': 'C:\\side\\mm.gguf' }
  const browse = vi
    .spyOn(actions, 'BrowseSideFile')
    .mockResolvedValue({ cancelled: true, file: '' } as never)
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await expect
    .element(page.getByText('mm.gguf', { exact: false }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Browse for mmproj file' }).click()
  expect(browse).toHaveBeenCalledExactlyOnceWith('image')
  expect(save).not.toHaveBeenCalled()
  await page.getByRole('button', { name: 'Clear mmproj file' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: ['--mmproj'],
  })
})

test('a package model shows no helper file rows', async () => {
  const view = plainFileView()
  view.saved.manifest_path = 'S:\\pkg\\bubo-package.json'
  view.saved.model_path = undefined
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Save server setup' }))
    .toBeVisible()
  expect(
    page.getByRole('button', { name: 'Browse for MTP draft file' }).query(),
  ).toBeNull()
  expect(
    page.getByRole('button', { name: 'Browse for mmproj file' }).query(),
  ).toBeNull()
})

test('a file the CLI says does not fit is shown with its reason and not saved', async () => {
  const view = plainFileView()
  vi.spyOn(actions, 'BrowseSideFile').mockResolvedValue({
    cancelled: false,
    file: 'C:\\side\\other-model-mtp.gguf',
    checked: true,
    fits: false,
    reason: 'This draft file is for a different model.',
  } as never)
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await page.getByRole('button', { name: 'Browse for MTP draft file' }).click()
  await expect
    .element(
      page
        .getByRole('alert')
        .getByText('This draft file is for a different model.'),
    )
    .toBeVisible()
  expect(save).not.toHaveBeenCalled()
})

test('a fitting file shows the CLI note; an unchecked file says so and is still saved', async () => {
  const view = plainFileView()
  const browse = vi.spyOn(actions, 'BrowseSideFile')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  browse.mockResolvedValueOnce({
    cancelled: false,
    file: 'C:\\side\\mtp.gguf',
    checked: true,
    fits: true,
    note: 'The file’s origin cannot be checked from its headers.',
  } as never)
  await page.getByRole('button', { name: 'Browse for MTP draft file' }).click()
  await expect
    .element(page.getByText('Fits this model.', { exact: false }))
    .toBeVisible()
  expect(save).toHaveBeenCalledTimes(1)
  browse.mockResolvedValueOnce({
    cancelled: false,
    file: 'C:\\side\\mm.gguf',
    checked: false,
  } as never)
  await page.getByRole('button', { name: 'Browse for mmproj file' }).click()
  await expect
    .element(
      page.getByText('Bubo could not check this file against the model.'),
    )
    .toBeVisible()
  expect(save).toHaveBeenCalledTimes(2)
})
