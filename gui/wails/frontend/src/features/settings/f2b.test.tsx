import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import { actions } from '@/lib/bindings'
import { RegistrySection } from './registry-section'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('the registry renderer keeps one sampling editor and preserves the View write route', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--temperature')!
  view.switches = [row]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .poll(
      () => document.querySelectorAll(`[data-setting="${row.name}"]`).length,
    )
    .toBe(1)
  await page.getByRole('spinbutton', { name: row.label! }).fill('0.65')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: '0.65' }],
    clear: [],
  })
})

test('a CLI-annotated random seed shows an empty field with Random as its placeholder', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--seed')!
  view.switches = [row]
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  const seed = page.getByRole('spinbutton', { name: 'Seed', exact: true })
  await expect.element(seed).toHaveAttribute('placeholder', 'Random')
  await expect.element(seed).toHaveValue(null)
})

test('a fixed random seed saves its exact scalar', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--seed')!
  view.switches = [row]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await page.getByRole('spinbutton', { name: 'Seed', exact: true }).fill('5')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: '5' }],
    clear: [],
  })
})

test('reasoning has an accent slider with supplied endpoints and permits values outside its suggestion', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--think-marker-bias')!
  view.switches = [row]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  // The label is the registry's own.
  const label = 'Rethinking penalty (fixed)'
  const range = page.getByRole('slider', { name: `${label} suggested range` })
  const suggested = row.allowed.properties!.suggested!.properties!
  await expect.element(range).toHaveAttribute('min', suggested.minimum!.text!)
  await expect.element(range).toHaveAttribute('max', suggested.maximum!.text!)
  expect(range.element().classList.contains('accent-primary')).toBe(true)
  // The suggested range is part of the row's About popup.
  await page.getByRole('button', { name: `About ${label}` }).click()
  // The range is said once, not as a range and again as a suggestion.
  await expect
    .element(page.getByText('Suggested:', { exact: false }))
    .toHaveTextContent(
      `Suggested: from ${suggested.minimum!.text} up to ${suggested.maximum!.text}. Other values are accepted.`,
    )
  expect(document.body.textContent).not.toContain('Suggested range')
  const outside = Number(suggested.maximum!.text) + 1
  await page.getByRole('spinbutton', { name: label }).fill(String(outside))
  await expect.element(range).toBeDisabled()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: String(outside) }],
    clear: [],
  })
})
