import { expect, test } from 'vitest'
import { modelChoices } from '@/components/model-facts'
import { packagesView } from '@/components/core-page-test-support'

// E22 fix 7 (UX15 K15-03, T15-10).
//
// Failure list, written before the code:
//  7a. Two copies of one package are labelled "{name} (C:\Users\fixture\...)": the
//      part that tells them apart sits at the end, where a closed select cuts it.
//      -> "{name} · {drive}\…\{leaf folder}": the drive comes first.
//  7b. The label carries the whole folder path.
//      -> the label stays short: no profile folder in it.
//  7c. Two copies on one drive have the same drive and the same leaf folder.
//      -> the label takes one more folder from the right until the labels differ.
//  7d. A package that exists once keeps its plain name.
const id = 'huihui-qwen38-flash-next-ud-q4-k-xl-v1'

test('7a, 7b: two copies are told apart by the drive, first and short', () => {
  const { view } = packagesView()
  const copies = modelChoices(view).filter((choice) => choice.id === id)
  expect(copies.map((choice) => choice.name)).toEqual([
    `C:\\…\\${id} · Qwen3.8 Flash Next · Huihui abliterated · UD-Q4_K_XL`,
    `D:\\…\\${id} · Qwen3.8 Flash Next · Huihui abliterated · UD-Q4_K_XL`,
  ])
  for (const copy of copies) {
    expect(copy.name).not.toContain('Users')
    expect(copy.name.indexOf(':')).toBeLessThan(40)
  }
})

test('7c: two copies on one drive take more of the folder until they differ', () => {
  const { view, rows } = packagesView()
  rows[1]!.manifest_path = `C:\\Other\\models\\${id}\\bubo-package.json`
  rows[0]!.manifest_path = `C:\\Mine\\models\\${id}\\bubo-package.json`
  view.models!.prepared_models = rows
  view.prepared_models = rows
  const names = modelChoices(view)
    .filter((choice) => choice.id === id)
    .map((choice) => choice.name)
  expect(new Set(names).size).toBe(2)
  expect(names[0]).toContain('Mine')
  expect(names[1]).toContain('Other')
})

test('7d: a package that exists once keeps its plain name', () => {
  const { view, rows } = packagesView()
  view.models!.prepared_models = [rows[0]!]
  view.prepared_models = [rows[0]!]
  expect(modelChoices(view).map((choice) => choice.name)).toEqual([
    'Qwen3.8 Flash Next · Huihui abliterated · UD-Q4_K_XL',
  ])
})
