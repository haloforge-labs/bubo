import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { bridgeReply } from '@/components/bridge-test-reply'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { guiError } from '@/lib/gui-errors'
import { folderPath } from '@/lib/format'

afterEach(() => vi.restoreAllMocks())

// E21w W2. "Check or repair files" on a package whose downloaded file fails its
// pin ends with integrity_failed. The CLI never deletes or replaces that file
// (it may be the user's own file or a hard link), so the same job fails the same
// way every time. The walk showed "The downloaded files did not pass
// verification." and Try again, which cannot work.
//
// Ways this could fail, written before the code:
//
//	D1 the box does not name the damaged file.
//	D2 Try again is offered while the file is still there (it cannot work).
//	D3 the box does not say what to do: delete that file, then choose
//	   "Check or repair files".
//	D4 the box offers no Open folder, or Open folder opens a folder the app
//	   refuses (the file sits in a subfolder of the package; only the package
//	   folder is a known model folder).
//	D5 "Check or repair files", which the box names, is not on the page.
//	D6 the box is hidden, or Dismiss is gone.
//	D7 the CLI detail has a shape we cannot read: the box crashes, or tells the
//	   user to delete a file it cannot name. integrity_failed has other CLI
//	   sources (a supplied file that does not match: the user's own file, which
//	   must not be deleted; a partial that changed during its check: Try again
//	   works), so an unread detail keeps the plain sentence and Try again.
//	   (model_prepare.cpp Error(err, 6, ...))
//	D8 the same failure in a first download (kind download) differs.
//	D9 another failure loses its Try again.
//	D10 the wording is not in one place (gui-errors.ts), so a CLI repair would
//	   need edits in several files.

// The CLI's own line (src/cli/prepare/model_prepare.cpp Different()): the path,
// the reason, then the instruction. The package folder comes from the recording.
function damaged(
  kind: 'repair' | 'download',
  detail?: (path: string) => string,
) {
  const view = coreView()
  // The package is not the one in use, so its row offers Check or repair files.
  view.serving = false
  view.current_model = undefined
  const row = view.models!.prepared_models[0]!
  const folder = folderPath(row.manifest_path!)
  // The CLI's --output for a package is the root that holds models\<id>\ (the
  // recordings of the models list). Only a package in that layout has a repair.
  row.repair_output = folder.slice(
    0,
    folder.length - `models\\${row.id}\\`.length - 1,
  )
  const file = `${folder}draft_head\\draft-out-q4k.gguf`
  const template = coreView('models-download-in-progress').models!.download!
  view.models!.download = {
    ...template,
    id: row.id,
    kind,
    state: 'failed',
    repair_output: kind === 'repair' ? row.repair_output : undefined,
    error: {
      code: 'integrity_failed',
      remedy: 'Remove the invalid file. Run prepare again.',
      exit_code: 6,
      detail: detail
        ? detail(file)
        : `${file}: SHA-256 differs from the pin; delete it and run bubo prepare again`,
    },
    message: 'Remove the invalid file. Run prepare again.',
  }
  return { view, row, folder, file }
}

for (const kind of ['repair', 'download'] as const) {
  test(`a damaged file (${kind}) is named, Open folder opens its package, and Try again is gone`, async () => {
    const { view, folder, file } = damaged(kind)
    const open = vi.spyOn(actions, 'OpenModelFolder').mockResolvedValue({})
    mountCorePage('/models', view)
    const box = page.getByRole('region', { name: 'Model download progress' })
    await expect.element(box).toBeVisible()
    // D1 D3
    const alert = box.getByRole('alert')
    await expect.element(alert).toHaveTextContent('draft-out-q4k.gguf')
    await expect.element(alert).toHaveTextContent(/delete that file/i)
    await expect.element(alert).toHaveTextContent('Check or repair files')
    await expect.element(box).toHaveTextContent(file)
    // D2
    expect(box.getByRole('button', { name: 'Try again' }).elements()).toEqual(
      [],
    )
    // D6
    await expect
      .element(box.getByRole('button', { name: 'Dismiss' }))
      .toBeVisible()
    // D4: the package folder, never the subfolder of the file.
    await box.getByRole('button', { name: 'Open folder' }).click()
    expect(open).toHaveBeenCalledExactlyOnceWith(folder)
    // D5: the instruction names a button that exists.
    await expect
      .element(
        page
          .getByRole('region', { name: 'Installed models' })
          .getByRole('button', { name: 'Check or repair files' })
          .first(),
      )
      .toBeVisible()
  })
}

test('a detail the app cannot read names nothing to delete and keeps Try again', async () => {
  // The CLI's supplied-file mismatch line (model_prepare.cpp): the user's own file.
  const { view } = damaged(
    'repair',
    (path) => `${path}: supplied-file SHA-256 or size mismatch`,
  )
  vi.spyOn(actions, 'OpenModelFolder').mockResolvedValue(bridgeReply({}))
  mountCorePage('/models', view)
  const box = page.getByRole('region', { name: 'Model download progress' })
  const alert = box.getByRole('alert')
  await expect
    .element(alert)
    .toHaveTextContent('The downloaded files did not pass verification.')
  expect(alert.element().textContent ?? '').not.toMatch(/delete/i)
  await expect
    .element(box.getByRole('button', { name: 'Try again' }))
    .toBeVisible()
  expect(box.getByRole('button', { name: 'Open folder' }).elements()).toEqual(
    [],
  )
  await expect
    .element(box.getByRole('button', { name: 'Dismiss' }))
    .toBeVisible()
})

test('another failed job keeps its Try again', async () => {
  const { view } = damaged('repair')
  view.models!.download!.error = {
    code: 'transport_failed',
    remedy: 'Run the command again.',
    exit_code: 7,
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('button', { name: 'Try again' }))
    .toBeVisible()
})

// D10: one place for the words, and no recovery button that cannot work.
// D7: without a named file there is nothing to delete, so Try again stays.
test('integrity_failed wording has one source and offers no retry for a named file', () => {
  const none = guiError({ code: 'integrity_failed', remedy: 'x' })
  expect(none.recovery).toBe('retry')
  expect(none.message).toBe('The downloaded files did not pass verification.')
  const named = guiError({
    code: 'integrity_failed',
    remedy: 'x',
    detail:
      'C:\\models\\pkg\\a.gguf: SHA-256 differs from the pin; delete it and run bubo prepare again',
  })
  expect(named.recovery).toBe('none')
  expect(named.message).toContain('a.gguf')
  expect(named.message).toContain('Check or repair files')
})
