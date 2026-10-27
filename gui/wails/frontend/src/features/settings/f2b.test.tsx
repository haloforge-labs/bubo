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
  await page.getByRole('textbox', { name: row.label! }).fill('0.65')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: '0.65' }],
    clear: [],
  })
})

test('a CLI-annotated random seed shows words and saves its exact scalar', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--seed')!
  view.switches = [row]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('textbox', { name: 'Seed', exact: true }))
    .toHaveValue('random')
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await page.getByRole('textbox', { name: 'Seed', exact: true }).fill('5')
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
  const range = page.getByRole('slider', {
    name: `${row.label} suggested range`,
  })
  const suggested = row.allowed.properties!.suggested!.properties!
  await expect.element(range).toHaveAttribute('min', suggested.minimum!.text!)
  await expect.element(range).toHaveAttribute('max', suggested.maximum!.text!)
  expect(range.element().classList.contains('accent-primary')).toBe(true)
  await expect
    .element(page.getByText('Suggested minimum', { exact: false }))
    .toHaveTextContent(suggested.minimum!.text!)
  await expect
    .element(page.getByText('Suggested maximum', { exact: false }))
    .toHaveTextContent(suggested.maximum!.text!)
  const outside = Number(suggested.maximum!.text) + 1
  await page.getByRole('spinbutton', { name: row.label! }).fill(String(outside))
  await expect.element(range).toBeDisabled()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: String(outside) }],
    clear: [],
  })
})
