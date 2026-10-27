import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView } from '@/components/core-page-test-support'
import { capturePage } from '@/components/page-capture'
import { folderPath } from '@/lib/format'

// E22 E-7: the damaged-file box with its current wording (a file that fails its
// pin, named by the CLI's own detail line). The state is the one the
// damaged-file test builds: a failed repair job on a package that is not in use.
test('E22 Models: the damaged-file box', () => {
  const view = coreView()
  view.serving = false
  view.current_model = undefined
  const row = view.models!.prepared_models[0]!
  const folder = folderPath(row.manifest_path!)
  row.repair_output = folder.slice(
    0,
    folder.length - `models\\${row.id}\\`.length - 1,
  )
  const file = `${folder}draft_head\\draft-out-q4k.gguf`
  const template = coreView('models-download-in-progress').models!.download!
  view.models!.download = {
    ...template,
    id: row.id,
    kind: 'repair',
    state: 'failed',
    repair_output: row.repair_output,
    error: {
      code: 'integrity_failed',
      remedy: 'Remove the invalid file. Run prepare again.',
      exit_code: 6,
      detail: `${file}: SHA-256 differs from the pin; delete it and run bubo prepare again`,
    },
    message: 'Remove the invalid file. Run prepare again.',
  }
  return capturePage({
    feature: 'models',
    scenario: 'e22-damaged-file',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('region', { name: 'Model download progress' }))
        .toBeVisible()
    },
  })
})
