import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import SettingsPage from './index'
import { RegistrySection } from './registry-section'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('Settings opens App and exposes only App and Advanced tabs', async () => {
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  await expect
    .element(page.getByRole('tab', { name: 'App' }))
    .toHaveAttribute('data-state', 'active')
  await expect
    .poll(() =>
      page
        .getByRole('tab')
        .elements()
        .map((tab) => tab.textContent),
    )
    .toEqual(['App', 'Advanced'])
  await expect
    .element(page.getByRole('switch', { name: 'Check for updates at start' }))
    .toBeVisible()
  await expect
    .element(page.getByText('Temperature', { exact: true }))
    .not.toBeInTheDocument()
})

test('App preferences remain available when the CLI reports no switch rows', async () => {
  const view = fixture()
  view.switches = []
  cleanup = renderView(<SettingsPage />, view).cleanup
  await expect
    .element(page.getByRole('switch', { name: 'Check for updates at start' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('tab', { name: 'Advanced' }))
    .toBeVisible()
})

test('Advanced keeps user rows without a Server section reachable once', async () => {
  const view = fixture()
  const template = view.switches[0]!
  view.switches = [
    {
      ...template,
      name: '--synthetic-setting',
      label: 'Synthetic extension',
      audience: 'user',
      group: 'Fixture extensions',
      gui_control: true,
    },
    {
      ...template,
      name: '--temperature',
      label: 'Temperature',
      audience: 'user',
      group: 'Behaviour',
      gui_control: true,
    },
  ]
  cleanup = renderView(<SettingsPage />, view).cleanup
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByText('Synthetic extension', { exact: true }))
    .toBeVisible()
  expect(
    document.querySelectorAll('[data-setting="--synthetic-setting"]'),
  ).toHaveLength(1)
  await expect
    .element(page.getByText('Temperature', { exact: true }))
    .not.toBeInTheDocument()
})

test.each([
  '--profile',
  '--port',
  '--npu-prefill',
  '--reasoning-effort',
  '--context',
  '--sessions',
])(
  '%s writes and clears the preference named by the View from the registry renderer',
  async (name) => {
    const view = fixture()
    const row = view.switches.find((row) => row.name === name)!
    const route = view.switch_overrides.find((row) => row.name === name)!
    expect(route.write_via).toBe('preference')
    const field = route.preference_field!
    const allowed = row.allowed.properties!
    const value =
      name === '--profile'
        ? view.profiles.filter((profile) => profile.gui)[0]!.name
        : route.preference_type === 'number' ||
            route.preference_type === 'integer'
          ? 7
          : (allowed.values?.items?.[0]?.text ?? 'fixture-value')
    view.switches = [row]
    view.saved = { ...view.saved, [field]: value }
    route.saved = String(
      name === '--profile'
        ? view.profiles.filter((profile) => profile.gui)[1]!.name
        : typeof value === 'number'
          ? value - 1
          : (allowed.values?.items?.[1]?.text ?? 'fixture-before'),
    )
    const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
    const override = vi.spyOn(actions, 'SetSwitchOverrides')
    cleanup = renderView(
      <RegistrySection view={view} audience="user" />,
      view,
    ).cleanup
    const type = allowed.type?.text
    if (type === 'enum')
      await page
        .getByRole('combobox', {
          name: name === '--profile' ? 'Mode' : row.label!,
        })
        .selectOptions('0')
    else
      await page
        .getByRole(
          type === 'integer' || type === 'number' ? 'spinbutton' : 'textbox',
          { name: row.label! },
        )
        .fill(String(value))
    await expect
      .element(page.getByRole('button', { name: 'Save changes', exact: true }))
      .toBeEnabled()
    await page
      .getByRole('button', { name: 'Save changes', exact: true })
      .click()
    expect(patch).toHaveBeenLastCalledWith({
      set: {},
      clear: [],
      switch_values: [{ name, value: String(value) }],
    })
    expect(override).not.toHaveBeenCalled()
    await page
      .getByRole('button', { name: 'Reset to default', exact: true })
      .click()
    await page
      .getByRole('button', { name: 'Save changes', exact: true })
      .click()
    expect(patch).toHaveBeenLastCalledWith({ set: {}, clear: [field] })
  },
)

test('an argv host row retains its CLI text control and current host scalar', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--host')!
  const state = view.switch_overrides.find((entry) => entry.name === row.name)!
  view.switches = [row]
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  const control = page.getByRole('textbox', { name: row.label! })
  await expect.element(control).toHaveValue(state.saved!)
  await expect.element(page.getByRole('switch')).not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await control.fill('fixture-host')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: row.name, value: 'fixture-host' }],
  })
})

test('CLI text port preserves its scalar and reports an unsupported app transport from the supervisor', async () => {
  const view = fixture()
  view.switches = view.switches.filter((row) => row.name === '--port')
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'unsupported',
      remedy: 'Fixture integer transport required.',
      retryable: false,
      component: '',
      stage: '',
    },
  })
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page
    .getByRole('textbox', { name: view.switches[0]!.label! })
    .fill('7.5')
  await page.getByRole('button', { name: 'Save changes' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture integer transport required.')
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: view.switches[0]!.name, value: '7.5' }],
  })
})

test('B5 suggested range is a hint and permits a value beyond the suggestion', async () => {
  const view = fixture()
  const row = view.switches.find(
    (row) =>
      row.allowed.properties?.suggested &&
      row.allowed.properties.type?.text === 'number',
  )!
  view.switches = [row]
  const suggestedMax = Number(
    row.allowed.properties!.suggested!.properties!.maximum!.text,
  )
  const value = suggestedMax + 1
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page
    .getByRole('button', { name: `About ${row.label}`, exact: true })
    .click()
  await expect
    .element(page.getByText('Suggested:', { exact: false }))
    .toBeVisible()
  const input = page.getByRole('spinbutton', { name: row.label! })
  expect(input.element().getAttribute('max')).toBeNull()
  await input.fill(String(value))
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: String(value) }],
    clear: [],
  })
})

test('a dummy registry row appears in Settings without any feature code change', async () => {
  const view = fixture()
  view.switches = [
    {
      ...view.switches[0]!,
      name: '--synthetic-setting',
      label: 'Synthetic extension',
      purpose: 'A new fixture setting.',
      audience: 'user',
      group: 'Fixture extensions',
      gui_control: true,
      allowed: {
        kind: 'object',
        properties: {
          type: { kind: 'string', text: 'number' },
          minimum: { kind: 'number', text: '0' },
        },
      },
    },
  ]
  view.settings = []
  view.switch_overrides = [
    {
      name: '--synthetic-setting',
      saved: undefined,
      effective: { kind: 'number', text: '0.5' },
      write_block: '',
      restart_state: 'pending',
    },
  ]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  view.effective_switches = [
    {
      name: '--synthetic-setting',
      effective: { kind: 'number', text: '0.5' },
      source: 'fixture',
      audience: 'user',
      group: 'Fixture extensions',
    },
  ]
  cleanup = renderView(<SettingsPage />, view).cleanup
  await page.getByRole('tab', { name: 'Advanced' }).click()
  const field = page.getByRole('spinbutton', { name: 'Synthetic extension' })
  await expect.element(field).toHaveValue(0.5)
  await expect
    .element(page.getByText('used after restart', { exact: true }))
    .toBeVisible()
  await field.fill('0.75')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: '--synthetic-setting', value: '0.75' }],
    clear: [],
  })
})

test('Advanced has a collapsed Debug subsection and obeys audience and gui_control', async () => {
  const view = fixture()
  const template = view.switches[0]!
  view.switches = [
    {
      ...template,
      name: '--main',
      label: 'Main row',
      audience: 'user',
      group: 'Fixture extensions',
      gui_control: true,
    },
    {
      ...template,
      name: '--advanced',
      label: 'Advanced row',
      audience: 'advanced',
      gui_control: true,
    },
    {
      ...template,
      name: '--debug',
      label: 'Debug row',
      audience: 'debug',
      gui_control: true,
    },
    {
      ...template,
      name: '--hidden',
      label: 'Hidden row',
      audience: 'user',
      gui_control: false,
    },
  ]
  cleanup = renderView(<SettingsPage />, view).cleanup
  await expect
    .element(page.getByText('Main row', { exact: true }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByText('Hidden row', { exact: true }))
    .not.toBeInTheDocument()
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  await expect
    .element(page.getByText('Main row', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('Advanced row', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('Debug row', { exact: true }))
    .not.toBeInTheDocument()
  await page.getByRole('button', { name: 'Debug', exact: true }).click()
  await expect
    .element(page.getByText('Debug row', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('These settings are for troubleshooting.'))
    .toBeVisible()
})

test('missing write eligibility is neutral and never invents an editable boolean', async () => {
  const view = fixture()
  view.switches = [
    {
      ...view.switches[0]!,
      label: 'Unreported fixture toggle',
      audience: 'user',
      gui_control: true,
      allowed: {
        kind: 'object',
        properties: { type: { kind: 'string', text: 'boolean' } },
      },
    },
  ]
  view.settings = []
  view.effective_switches = []
  view.switch_overrides = []
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect.element(page.getByRole('switch')).not.toBeInTheDocument()
  await expect
    .element(
      page.getByText("This setting can't be changed from the app.", {
        exact: true,
      }),
    )
    .toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
})

test('App drafts retain edits across a View poll and save only edited fields', async () => {
  const view = fixture()
  const mounted = renderView(<SettingsPage />, view)
  cleanup = mounted.cleanup
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await expect
    .element(page.getByRole('switch', { name: 'Check for updates at start' }))
    .toBeChecked()
  const runtime = page.getByRole('textbox', { name: 'Runtime folder' })
  await runtime.fill('C:\\fixture-runtime')
  await mounted.client.invalidateQueries()
  await expect.element(runtime).toHaveValue('C:\\fixture-runtime')
  await page.getByRole('button', { name: 'Save app settings' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { runtime_path: 'C:\\fixture-runtime' },
    clear: [],
  })
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('App preferences saved.')
})

test('model folders use the typed add and remove bindings and surface rejection', async () => {
  const view = fixture()
  view.saved.model_folders = ['C:\\fixture-models']
  const remove = vi.spyOn(actions, 'RemoveModelFolder').mockRejectedValue({
    code: 'folder_busy',
    remedy: 'Try again after the scan.',
  })
  const add = vi
    .spyOn(actions, 'AddModelFolder')
    .mockResolvedValue({ value: 'C:\\another-fixture' })
  cleanup = renderView(<SettingsPage />, view).cleanup
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await page.getByRole('button', { name: 'Remove folder' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('folder_busy: Try again after the scan.')
  expect(remove).toHaveBeenCalledWith('C:\\fixture-models')
  await page
    .getByRole('textbox', { name: 'Add a model folder' })
    .fill('C:\\another-fixture')
  await page.getByRole('button', { name: 'Add folder', exact: true }).click()
  expect(add).toHaveBeenCalledWith('C:\\another-fixture')
})

test('Server-owned user groups do not create Settings tabs or duplicate editors', async () => {
  const view = fixture()
  cleanup = renderView(<SettingsPage />, view).cleanup
  await expect
    .poll(() =>
      page
        .getByRole('tab')
        .elements()
        .map((tab) => tab.textContent),
    )
    .toEqual(['App', 'Advanced'])
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByText('Temperature', { exact: true }))
    .not.toBeInTheDocument()
  expect(
    document.querySelectorAll('[data-setting="--temperature"]'),
  ).toHaveLength(0)
})

test('a registry save preserves typed CLI rejection and an explicit clear restores the CLI default', async () => {
  const view = fixture()
  const template = view.switches[0]!
  view.switches = [
    {
      ...template,
      name: '--fixture-control',
      label: 'Fixture control',
      audience: 'user',
      gui_control: true,
      allowed: {
        kind: 'object',
        properties: { type: { kind: 'string', text: 'boolean' } },
      },
    },
  ]
  view.switch_overrides = [
    {
      name: '--fixture-control',
      saved: 'true',
      effective: { kind: 'boolean', boolean: false },
      write_block: '',
      restart_state: 'unknown_until_restart',
    },
  ]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'fixture_rejected',
      component: 'fixture',
      stage: 'save',
      retryable: false,
      remedy: 'Review this value.',
    },
  })
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page.getByRole('switch', { name: 'Fixture control' }).click()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    switch_values: [{ name: '--fixture-control', value: 'false' }],
    clear: [],
  })
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Review this value.')
  await expect
    .element(page.getByText('Request saved.', { exact: false }))
    .not.toBeInTheDocument()
  save.mockResolvedValue({})
  await page
    .getByRole('button', { name: 'Reset to default', exact: true })
    .click()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    clear: ['--fixture-control'],
  })
  await expect
    .element(page.getByText('Request saved.', { exact: false }))
    .toHaveTextContent('Validated when the server starts.')
  await expect
    .element(page.getByText('used after restart', { exact: false }))
    .toBeVisible()
})

test('write eligibility comes from the View and never from echo or a switch name', async () => {
  const view = fixture()
  view.switches = [
    {
      ...view.switches[0]!,
      name: '--fixture-blocked',
      label: 'Blocked fixture',
      audience: 'user',
      gui_control: true,
    },
  ]
  view.switch_overrides = [
    {
      name: '--fixture-blocked',
      saved: undefined,
      effective: { kind: 'boolean', boolean: false },
      write_block: 'secret',
      restart_state: 'unchanged',
    },
  ]
  const save = vi.spyOn(actions, 'PatchPreferences')
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByText("This setting can't be changed from the app.", {
        exact: true,
      }),
    )
    .toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  expect(save).not.toHaveBeenCalled()
})

test('App writes explicit false and clears defaults without copying the saved snapshot', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await page.getByRole('switch', { name: 'Check for updates at start' }).click()
  await page.getByRole('button', { name: 'Save app settings' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: { check_for_updates: false },
    clear: [],
  })
  await page.getByRole('button', { name: 'Use app defaults' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    clear: [
      'start_gui_at_login',
      'close_to_tray',
      'start_server_with_gui',
      'check_for_updates',
      'runtime_path',
    ],
  })
})

test('App preference rejection remains in place and never shows a saved notice', async () => {
  vi.spyOn(actions, 'PatchPreferences').mockRejectedValue({
    code: 'fixture_rejected',
    remedy: 'Review app preferences.',
  })
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await page
    .getByRole('textbox', { name: 'Runtime folder' })
    .fill('D:/fixture/runtime')
  await page.getByRole('button', { name: 'Save app settings' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Review app preferences.')
  await expect
    .element(page.getByText('App preferences saved.'))
    .not.toBeInTheDocument()
})

test('asynchronous startup CLI rejection stays visible while saved override controls remain clearable', async () => {
  const view = fixture('switch-override-rejected')
  cleanup = renderView(<SettingsPage />, view).cleanup
  await page.getByText('Details', { exact: true }).first().click()
  await expect
    .element(
      page.getByText(`${view.failure!.code}: ${view.failure!.remedy}`, {
        exact: true,
      }),
    )
    .toBeVisible()
})

test('a numeric CLI effective value stays clean and an edit saves through its string registry control', async () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--temperature')!
  view.switches = [row]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('textbox', { name: row.label! }))
    .toHaveValue('1')
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await page.getByRole('textbox', { name: row.label! }).fill('0.9')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: row.name, value: '0.9' }],
    clear: [],
  })
})

test('the recorded pending environment boolean retains its CLI-declared wire values', async () => {
  const view = fixture('switch-overrides-pending')
  const entry = view.switch_overrides.find((entry) => entry.saved === '0')!
  view.switches = view.switches
    .filter((row) => row.name === entry.name)
    .map((row) => ({ ...row, audience: 'user' }))
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  const control = page.getByRole('switch', { name: view.switches[0]!.label! })
  await expect.element(control).not.toBeChecked()
  await expect
    .element(page.getByText('Set by you: Off', { exact: true }))
    .toBeVisible()
  await control.click()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: entry.name, value: '1' }],
    clear: [],
  })
})

test('the image input switch reads as its label: On means images are disabled', async () => {
  const view = fixture('lan-on')
  view.switches = view.switches
    .filter((row) => row.name === '--no-vision')
    // The recording predates the CLI's new label for this row.
    .map((row) => ({ ...row, audience: 'user', label: 'Disable image input' }))
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  const control = page.getByRole('switch', { name: 'Disable image input' })
  await expect.element(control).not.toBeChecked()
  await control.click()
  await expect.element(control).toBeChecked()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [{ name: '--no-vision', value: 'true' }],
    clear: [],
  })
})

test('an unsupported control still permits clearing its saved request', async () => {
  const view = fixture()
  view.switches = [
    {
      ...view.switches[0]!,
      name: '--fixture-future-format',
      audience: 'user',
      gui_control: true,
      allowed: {
        kind: 'object',
        properties: {
          type: { kind: 'string', text: 'string' },
          format: { kind: 'string', text: 'fixture-future' },
        },
      },
    },
  ]
  view.switch_overrides = [
    {
      name: '--fixture-future-format',
      saved: 'fixture',
      effective: { kind: 'null' },
      write_block: '',
      restart_state: 'unknown_until_restart',
    },
  ]
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByText("This setting can't be changed here."))
    .toBeVisible()
  await expect
    .element(
      page.getByText('Unsupported registry constraint', { exact: false }),
    )
    .toBeInTheDocument()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  await page.getByRole('button', { name: 'Reset to default' }).click()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: ['--fixture-future-format'],
  })
})

test('reasoning bias and loop breaker use the same generic switch write as sampling', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  const view = fixture()
  const bias = view.switches.find((row) => row.name === '--think-marker-bias')!
  const loop = view.switches.find((row) => row.name === '--loop-breaker')!
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page
    .getByRole('textbox', { name: 'Filter user settings' })
    .fill(bias.name)
  await page.getByRole('spinbutton', { name: bias.label! }).fill('-0.5')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    switch_values: [{ name: bias.name, value: '-0.5' }],
    clear: [],
  })
  await page
    .getByRole('textbox', { name: 'Filter user settings' })
    .fill(loop.name)
  await page.getByRole('combobox', { name: loop.label! }).selectOptions('1')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    switch_values: [{ name: loop.name, value: 'off' }],
    clear: [],
  })
})
