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

test('Settings sends preference and generic edits atomically and retains both after rejection', async () => {
  const view = fixture()
  const template = view.switches.find((row) => row.name === '--temperature')!
  view.switches = ['--fixture-pref', '--fixture-env'].map((name) => ({
    ...template,
    name,
    label: name,
    audience: 'advanced',
  }))
  view.switch_overrides = view.switches.map((row, index) => ({
    name: row.name,
    saved: '1',
    effective: { kind: 'string', text: '1' },
    write_block: '',
    restart_state: 'not_running',
    write_via: index === 0 ? 'preference' : 'override',
    ...(index === 0 ? { preference_field: 'temperature' } : {}),
  }))
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'fixture_rejected',
      remedy: 'Keep both edits.',
      component: 'fixture',
      stage: 'save',
      retryable: false,
    },
  })
  const legacy = vi.spyOn(actions, 'SetSwitchOverrides').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="advanced" />,
    view,
  ).cleanup
  for (const row of view.switches)
    await page
      .getByRole('textbox', { name: row.name, exact: true })
      .fill('0.75')
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: view.switches.map((row) => ({
      name: row.name,
      value: '0.75',
    })),
  })
  expect(legacy).not.toHaveBeenCalled()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Keep both edits.')
  for (const row of view.switches)
    await expect
      .element(page.getByRole('textbox', { name: row.name, exact: true }))
      .toHaveValue('0.75')
  await expect
    .element(page.getByRole('button', { name: 'Save changes', exact: true }))
    .toBeEnabled()
})

test('the marker-list placeholder is unset and never written as a filename', async () => {
  const view = fixture()
  const row = view.switches.find(
    (row) => row.name === 'GUFO_THINK_MARKER_LIST',
  )!
  view.switches = [row]
  const state = view.switch_overrides.find((state) => state.name === row.name)!
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience={row.audience} />,
    view,
  ).cleanup
  await expect
    .element(
      page.getByText('Custom marker-list file: not set (built-in list)', {
        exact: true,
      }),
    )
    .toBeVisible()
  cleanup()
  state.saved = 'C:/fixture/markers.txt'
  cleanup = renderView(
    <RegistrySection view={view} audience={row.audience} />,
    view,
  ).cleanup
  await page
    .getByRole('textbox', { name: row.label!, exact: true })
    .fill('none')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({ set: {}, clear: [row.name] })
})

test('unmodified CLI defaults in hidden Debug rows do not block an Advanced edit', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--draft-tokens')!
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="advanced" />,
    view,
  ).cleanup
  await page.getByRole('textbox', { name: 'Find a setting' }).fill(row.name)
  await page.getByRole('textbox', { name: row.label!, exact: true }).fill('6')
  await page.getByRole('button', { name: 'Save changes' }).click()
  await expect.poll(() => save.mock.calls.length).toBe(1)
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: '6' }],
    clear: [],
  })
})
