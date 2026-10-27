import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import type { View } from '@/lib/bindings'
import recorded from '@/scenarios/not-prepared.json'
import ready from '@/scenarios/ready-idle.json'
import {
  shortBy,
  modelChoices,
  packageStatus,
  recommendedModel,
  activeModelChoice,
  modelPreferencePatch,
} from '@/components/model-facts'

const view = recorded as unknown as View

// Port of legacy models.test.js: retain display, status, catalogue and
// action contracts; remove old frontend memory/space arithmetic and demo tests.
test('package status preserves reviewed words and actions; unknown state has no action', () => {
  expect(packageStatus('prepared')).toEqual({
    word: 'On this computer',
    action: 'check',
    actionLabel: 'Check or repair files',
  })
  expect(packageStatus('not_prepared').word).toBe('Not downloaded')
  expect(packageStatus('partial')).toEqual({
    word: 'Incomplete',
    action: 'repair',
    actionLabel: 'Check or repair files',
  })
  expect(packageStatus('future-state').action).toBe('none')
})
test('catalogue supplies every displayed name, size, licence and supported download', () => {
  const entries = view.models!.catalogue
  for (const entry of entries) {
    expect(entry.display_name.length).toBeGreaterThan(0)
    expect(entry.download_bytes).toBeGreaterThan(0)
    if (!entry.download_enabled) expect(entry.disabled_reason).toBeTruthy()
  }
  const model = recommendedModel(view)!
  expect(model.licence.name).toBeTruthy()
  expect(model.licence.url).toMatch(/^https:/)
  expect(
    model.files
      .filter((file) => file.downloadable)
      .reduce((total, file) => total + file.bytes, 0),
  ).toBe(model.download_bytes)
})
test('recommended selection follows role and CLI default, independent of model ID', () => {
  const clone = structuredClone(view)
  const model = recommendedModel(clone)!
  model.id = 'synthetic-extension'
  model.display_name = 'Synthetic recommended model'
  expect(recommendedModel(clone)).toBe(model)
})
test('picker retains CLI names and paths, excluding files the CLI cannot run', () => {
  const installedView = ready as unknown as View
  const choices = modelChoices(installedView)
  expect(choices[0]?.name).toBe(
    installedView.models!.prepared_models[0]?.display_name,
  )
  expect(choices[0]?.path).toBe(
    installedView.models!.prepared_models[0]?.manifest_path,
  )
  const clone = structuredClone(installedView)
  clone.models!.prepared_models[0]!.prepared_state = 'partial'
  // The recorded list holds other packages and a loose file; only the
  // incomplete package drops out of the choices.
  expect(modelChoices(clone).map((choice) => choice.path)).toEqual(
    choices
      .map((choice) => choice.path)
      .filter((path) => path !== choices[0]?.path),
  )
})
test('Select for Server, folder removal and import are wired to frozen bindings', () => {
  const source = readFileSync(new URL('./index.tsx', import.meta.url), 'utf8')
  for (const action of [
    'PatchPreferences',
    'RemoveModelFolder',
    'BrowseModelFile',
    'BrowseModelFolder',
  ])
    expect(source).toContain(`actions.${action}`)
})

test('active identity selects catalogue ID or file path independently of a served display name', () => {
  const clone = structuredClone(ready) as unknown as View
  const model = clone.prepared_models[0]!
  clone.current_model = {
    catalogue_id: model.id,
    display_name: model.display_name,
    model_path: '',
    manifest_path: model.manifest_path!,
  }
  clone.instance!.served_model = 'Custom serving alias'
  expect(activeModelChoice(clone)?.id).toBe(model.id)
  clone.current_model = {
    catalogue_id: '',
    display_name: 'model.gguf',
    model_path: 'D:/fixture/model.gguf',
    manifest_path: '',
  }
  expect(activeModelChoice(clone)).toBeUndefined()
  expect(modelPreferencePatch('file', 'D:/fixture/model.gguf')).toEqual({
    set: { model_path: 'D:/fixture/model.gguf' },
    clear: ['manifest_path'],
  })
})

test('the space short by reads in GiB with one decimal, rounded up, and never 0.0', () => {
  const gib = 1024 ** 3
  expect(shortBy(24 * gib)).toBe('24.0 GiB')
  expect(shortBy(24 * gib + 1)).toBe('24.1 GiB')
  expect(shortBy(0.1 * gib)).toBe('0.1 GiB')
  // Under 0.1 GiB it is MiB.
  expect(shortBy(50 * 1024 ** 2)).toBe('50 MiB')
  expect(shortBy(1)).toBe('1 MiB')
})
