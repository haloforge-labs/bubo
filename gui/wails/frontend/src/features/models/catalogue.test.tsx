import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { modelChoices } from '@/components/model-facts'

afterEach(() => vi.restoreAllMocks())

// E14 item 4: a catalogue model that is not the default package is found in
// the models list (not in status), and starts by its model file.
function withCatalogueModel() {
  const view = coreView('stopped-saved-estimate')
  const other = view.models!.catalogue.find(
    (entry) => !entry.default_status || entry.default_status === 'none',
  )!
  view.models!.prepared_models = [
    ...view.models!.prepared_models,
    {
      id: other.id,
      display_name: other.display_name,
      manifest_path: undefined,
      model_path: 'D:/models/other/other-main.gguf',
      size_bytes: other.download_bytes,
      prepared_state: 'prepared',
      capabilities: {
        context_max: undefined,
        mtp: false,
        twin: false,
        mmproj: false,
      },
    } as NonNullable<typeof view.models>['prepared_models'][number],
  ]
  return { view, other }
}

test('the picker offers a catalogue model that status does not report, started by its model file', () => {
  const { view, other } = withCatalogueModel()
  expect(view.prepared_models.some((row) => row.id === other.id)).toBe(false)
  const choice = modelChoices(view).find((entry) => entry.id === other.id)!
  expect(choice.kind).toBe('file')
  expect(choice.path).toBe('D:/models/other/other-main.gguf')
  expect(choice.name).toBe(other.display_name)
})

test('Models lists it as installed and selects it with the model_path preference', async () => {
  const { view, other } = withCatalogueModel()
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  await expect.element(installed).toHaveTextContent(other.display_name)
  await installed
    .getByRole('row')
    .filter({ hasText: other.display_name })
    .getByRole('button', { name: 'Select for Server' })
    .click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: { model_path: 'D:/models/other/other-main.gguf' },
    clear: ['manifest_path'],
  })
})

test('the Server picker names it', async () => {
  const { view, other } = withCatalogueModel()
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('option', { name: other.display_name }))
    .toBeInTheDocument()
})
