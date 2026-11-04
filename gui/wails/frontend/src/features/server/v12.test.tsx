import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'

// The recorded registry now carries the real rows these tests add; replace them.
function dropRows(view: ReturnType<typeof coreView>, ...names: string[]) {
  view.switches = view.switches.filter((row) => !names.includes(row.name))
  view.switch_overrides = view.switch_overrides.filter(
    (row) => !names.includes(row.name),
  )
}

// The default window: a row has the room for its hint beside the control.
beforeEach(() => page.viewport(1280, 829))
afterEach(() => vi.restoreAllMocks())

test('Server always offers import and the CLI largest-context row', async () => {
  const view = coreView('stopped-saved-estimate')
  mountCorePage('/', view)
  await expect
    .element(
      page.getByRole('link', { name: 'Add a model you have', exact: true }),
    )
    .toHaveAttribute('href', '/models#import')
  await expect.element(page.getByText(/^Max [\d,]+ ·/)).toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Use largest context' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('link', { name: 'Connect', exact: true }).last())
    .toHaveAttribute('href', '/connect')
})

test('last request reports full cache hits and TTFT without an invented prefill rate', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  request.prompt_tokens = 17
  request.cached_tokens = 17
  request.prefill_tps = undefined
  request.ttft_ms = 8
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Prefill: all 17 tokens from cache', { exact: true }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'Time to first token' }))
    .toHaveTextContent('8 ms')
})

test('busy port saves the CLI port scalar inline', async () => {
  const view = coreView('stopped-saved-estimate')
  view.failure = {
    code: 'port_occupied',
    remedy: 'Fixture busy address',
    source: 'fixture',
    time: '',
    forced: false,
  }
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await page.getByLabelText('New server port').fill('9099')
  await page.getByRole('button', { name: 'Save port' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: '--port', value: '9099' }],
  })
})

test('Tight opt-in writes only consent and waits for supervisor admission', async () => {
  const view = coreView('memory-gpu-headroom')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const { replaceView } = mountCorePage('/', view)
  const start = page.getByRole('button', { name: 'Start server', exact: true })
  const checkbox = page.getByRole('checkbox', { name: /^Override:/ })
  await expect.element(start).toBeDisabled()
  await checkbox.click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { allow_tight_setup: true },
    clear: [],
  })
  await expect.element(start).toBeDisabled()
  replaceView({
    ...view,
    saved: { ...view.saved, allow_tight_setup: true },
    actions: {
      ...view.actions,
      start: { allowed: true, reason_code: '', reason: '' },
    },
  })
  await expect.element(checkbox).toBeChecked()
  await expect.element(start).toBeEnabled()
  await page
    .getByRole('spinbutton', { name: 'Context per session', exact: true })
    .fill('65536')
  await expect.element(checkbox).not.toBeChecked()
  await expect.element(checkbox).toBeDisabled()
  // Start saves and rechecks this draft; the saved Tight setup needs fresh consent.
  await expect.element(start).toBeEnabled()
})

test('Profile belongs in Settings and saves through its CLI-driven preference route', async () => {
  const view = coreView('memory-gpu-headroom')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await page
    .getByRole('combobox', { name: 'Profile', exact: true })
    .selectOptions('1')
  await expect
    .element(page.getByRole('button', { name: 'Save changes', exact: true }))
    .toBeEnabled()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [
      {
        name: '--profile',
        value: view.profiles.filter((profile) => profile.gui)[1]!.name,
      },
    ],
  })
})

test('partial cache hits show new and cached counts beside the CLI rate and real served name', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  request.prompt_tokens = 1373
  request.cached_tokens = 1345
  const served = view.models!.catalogue[0]!.id
  view.instance!.served_model = served
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('figure', { name: 'Prefill tok/s' }))
    .toHaveTextContent('28 new + 1,345 cached')
  await expect
    .element(page.getByText('Clients see this model as', { exact: false }))
    .toHaveTextContent(`Clients see this model as ${served}. (bubo also works)`)
})

test('Server shows the context cache option only when the CLI reports it', async () => {
  const view = coreView('stopped-saved-estimate')
  dropRows(view, '--kv-cache-type')
  const template = view.switches.find(
    (row) => row.name === '--reasoning-effort',
  )!
  const override = view.switch_overrides.find(
    (row) => row.name === '--reasoning-effort',
  )!
  view.switches.push({
    ...template,
    name: '--kv-cache-type',
    label: 'Smaller context cache (Q8)',
    group: 'Memory',
    audience: 'user',
    gui_control: true,
    default_text: 'f16',
    values: { ...template.values, default: { kind: 'string', text: 'f16' } },
    allowed: {
      kind: 'object',
      properties: {
        type: { kind: 'string', text: 'enum' },
        values: {
          kind: 'array',
          items: [
            { kind: 'string', text: 'f16' },
            { kind: 'string', text: 'q8_0' },
          ],
        },
      },
    },
  })
  view.switch_overrides.push({ ...override, name: '--kv-cache-type' })
  mountCorePage('/', view)
  // The KV cache is one line of the files grid, under the helper files.
  await expect
    .element(page.getByRole('group', { name: 'Files and cache' }))
    .toBeVisible()
  expect(page.getByRole('region', { name: 'Context cache' }).query()).toBeNull()
  await expect
    .element(page.getByText('KV cache', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('combobox', { name: 'KV cache' }))
    .toHaveDisplayValue('f16')
  await expect
    .element(page.getByRole('button', { name: 'About KV cache' }))
    .toBeVisible()
  expect(
    [
      ...document.querySelectorAll(
        '[data-setting="--kv-cache-type"] select option',
      ),
    ].map((option) => option.textContent),
  ).toEqual(['f16', 'Q8'])
  expect(
    document.querySelector('[data-setting="--kv-cache-type"] details')!
      .textContent,
  ).toContain('Q8: the KV cache takes about half the memory')
})

test('Server shows Prefill mode and Refusal direction as direct choices before Start', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = true
  dropRows(view, '--prefill-mode', '--refusal-direction')
  const template = view.switches.find(
    (row) => row.name === '--reasoning-effort',
  )!
  const override = view.switch_overrides.find(
    (row) => row.name === '--reasoning-effort',
  )!
  const choice = (
    name: string,
    label: string,
    group: string,
    first: string,
    second: string,
  ) => {
    view.switches.push({
      ...template,
      name,
      label,
      group,
      audience: 'user',
      gui_control: true,
      purpose:
        name === '--prefill-mode'
          ? 'Experimental. Up to 16 % faster when a prompt has 4,096 or more new tokens; shorter prompts read as normal.'
          : template.purpose,
      default_text: first,
      values: { ...template.values, default: { kind: 'string', text: first } },
      allowed: {
        kind: 'object',
        properties: {
          type: { kind: 'string', text: 'enum' },
          values: {
            kind: 'array',
            items: [
              { kind: 'string', text: first },
              { kind: 'string', text: second },
            ],
          },
        },
      },
    })
    view.switch_overrides.push({ ...override, name })
  }
  choice('--prefill-mode', 'Prefill mode', 'Speed', 'normal', 'fast')
  choice(
    '--refusal-direction',
    'In-engine abliteration (Qwen3.8 Flash Next)',
    'Behaviour',
    'off',
    'on',
  )
  mountCorePage('/', view)
  expect(
    page.getByRole('region', { name: 'Speed and behaviour' }).query(),
  ).toBeNull()
  await expect
    .element(
      page.getByRole('combobox', { name: 'Prefill mode (experimental)' }),
    )
    .toHaveDisplayValue('Normal')
  await expect
    .element(
      page.getByRole('combobox', {
        name: 'In-engine abliteration (Qwen3.8 Flash Next)',
      }),
    )
    .toHaveDisplayValue('Off')
  // The label carries the mark; the full text sits in the row's About popup.
  expect(page.getByText('Experimental', { exact: true }).query()).toBeNull()
  const about = (name: string) =>
    document.querySelector(`[data-setting="${name}"] details`)!
  expect((about('--prefill-mode') as HTMLDetailsElement).open).toBe(false)
  await page
    .getByRole('button', { name: 'About Prefill mode (experimental)' })
    .click()
  await expect
    .element(
      // The CLI's own text; the GUI adds none.
      page.getByText(
        'Experimental. Up to 16 % faster when a prompt has 4,096 or more new tokens; shorter prompts read as normal.',
        { exact: true },
      ),
    )
    .toBeVisible()
  // Close the first popup: it would cover the next row's ⓘ.
  await page
    .getByRole('button', { name: 'About Prefill mode (experimental)' })
    .click()
  await page
    .getByRole('button', {
      name: 'About In-engine abliteration (Qwen3.8 Flash Next)',
    })
    .click()
  await expect
    .element(
      page.getByText('Removes refusals while the model writes', {
        exact: false,
      }),
    )
    .toBeVisible()
  await expect
    .element(page.getByText('Off by default.', { exact: false }))
    .toBeVisible()
  expect(about('--refusal-direction').textContent).toContain(
    'Used after the server restarts.',
  )
  expect(document.body.textContent).not.toContain('Refusal direction')
  expect(
    page
      .getByRole('button', {
        name: 'Change Prefill mode (experimental)',
        exact: true,
      })
      .query(),
  ).toBeNull()
})

test('Prefill mode is off with the CLI reason when the CLI says fast mode does not work for the model', async () => {
  const build = (available: boolean | undefined) => {
    const view = coreView('stopped-saved-estimate')
    view.saved.inference_expanded = true
    dropRows(view, '--prefill-mode')
    const template = view.switches.find(
      (row) => row.name === '--reasoning-effort',
    )!
    const override = view.switch_overrides.find(
      (row) => row.name === '--reasoning-effort',
    )!
    view.switches.push({
      ...template,
      name: '--prefill-mode',
      label: 'Prefill mode',
      group: 'Speed',
      audience: 'user',
      gui_control: true,
      default_text: 'normal',
      values: {
        ...template.values,
        default: { kind: 'string', text: 'normal' },
      },
      allowed: {
        kind: 'object',
        properties: {
          type: { kind: 'string', text: 'enum' },
          values: {
            kind: 'array',
            items: [
              { kind: 'string', text: 'normal' },
              { kind: 'string', text: 'fast' },
            ],
          },
        },
      },
    })
    view.switch_overrides.push({ ...override, name: '--prefill-mode' })
    if (available !== undefined)
      view.saved_estimate.memory_estimate!.capabilities = {
        prefill_mode_fast: {
          available,
          reason: available ? '' : 'Not available for this model.',
        },
      }
    return view
  }
  mountCorePage('/', build(false))
  await expect
    .element(
      page.getByRole('combobox', { name: 'Prefill mode (experimental)' }),
    )
    .toBeDisabled()
  // The row says it in a short phrase (behind the ⓘ when its cell is narrow); the
  // CLI's reason is in the About popup.
  await expect
    .element(page.getByText('Unsupported model', { exact: true }))
    .toBeInTheDocument()
  await page
    .getByRole('button', { name: 'About Prefill mode (experimental)' })
    .click()
  await expect
    .element(page.getByText('Not available for this model.'))
    .toBeVisible()
})

test.each([[true], [undefined]])(
  'Prefill mode stays on when the CLI says %s about fast mode',
  async (available) => {
    {
      const view = coreView('stopped-saved-estimate')
      view.saved.inference_expanded = true
      dropRows(view, '--prefill-mode')
      const template = view.switches.find(
        (row) => row.name === '--reasoning-effort',
      )!
      const override = view.switch_overrides.find(
        (row) => row.name === '--reasoning-effort',
      )!
      view.switches.push({
        ...template,
        name: '--prefill-mode',
        label: 'Prefill mode',
        group: 'Speed',
        audience: 'user',
        gui_control: true,
        default_text: 'normal',
        values: {
          ...template.values,
          default: { kind: 'string', text: 'normal' },
        },
        allowed: {
          kind: 'object',
          properties: {
            type: { kind: 'string', text: 'enum' },
            values: {
              kind: 'array',
              items: [
                { kind: 'string', text: 'normal' },
                { kind: 'string', text: 'fast' },
              ],
            },
          },
        },
      })
      view.switch_overrides.push({ ...override, name: '--prefill-mode' })
      if (available !== undefined)
        view.saved_estimate.memory_estimate!.capabilities = {
          prefill_mode_fast: { available, reason: '' },
        }
      mountCorePage('/', view)
      await expect
        .element(
          page.getByRole('combobox', { name: 'Prefill mode (experimental)' }),
        )
        .toBeEnabled()
    }
  },
)

test('Server has no context cache section when the CLI does not report the option', async () => {
  const view = coreView('stopped-saved-estimate')
  view.switches = view.switches.filter((row) => row.name !== '--kv-cache-type')
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('link', { name: 'Connect', exact: true }).first())
    .toBeVisible()
  expect(
    document.querySelectorAll('[aria-label="Context cache"]'),
  ).toHaveLength(0)
})

test('the failed start banner names In-engine abliteration, never the CLI name', async () => {
  const view = coreView('switch-override-rejected')
  view.failure!.switch = '--refusal-direction'
  mountCorePage('/', view)
  const box = page.getByRole('region', { name: 'Rejected override' })
  await expect
    .element(box)
    .toHaveTextContent(
      'The server did not start because of the In-engine abliteration (Qwen3.8 Flash Next) setting you changed.',
    )
  await expect
    .element(
      box.getByRole('button', {
        name: 'Reset In-engine abliteration (Qwen3.8 Flash Next)',
      }),
    )
    .toBeEnabled()
  expect(document.body.textContent).not.toContain('Refusal direction')
})

test('In-engine abliteration is off and disabled with the CLI reason, whatever was saved', async () => {
  const view = coreView('stopped-saved-estimate')
  view.switch_overrides.find(
    (row) => row.name === '--refusal-direction',
  )!.saved = 'on'
  // Asked for separately: serve switches saved on the page stop the estimate.
  view.saved_estimate.capabilities = {
    refusal_direction: {
      available: false,
      reason: 'This model is already abliterated.',
    },
  }
  mountCorePage('/', view)
  const select = page.getByRole('combobox', {
    name: 'In-engine abliteration (Qwen3.8 Flash Next)',
  })
  await expect.element(select).toBeDisabled()
  await expect.element(select).toHaveDisplayValue('Off')
  await expect
    .element(page.getByText('Already abliterated', { exact: true }))
    .toBeVisible()
  await page
    .getByRole('button', {
      name: 'About In-engine abliteration (Qwen3.8 Flash Next)',
    })
    .click()
  await expect
    .element(page.getByText('This model is already abliterated.'))
    .toBeVisible()
  // The remembered On cannot be used here, and it can be cleared: Reset stages
  // the clear and Save sends it.
  await page
    .getByRole('button', {
      name: 'About In-engine abliteration (Qwen3.8 Flash Next)',
    })
    .click()
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  await page
    .getByRole('button', {
      name: 'Reset In-engine abliteration (Qwen3.8 Flash Next) to default',
    })
    .click()
  await page.getByRole('button', { name: 'Save server setup' }).click()
  expect(patch).toHaveBeenCalledWith(
    expect.objectContaining({ clear: ['--refusal-direction'] }),
  )
})

test('every model row describes the model in the picker, saved or not', async () => {
  const view = coreView('stopped-saved-estimate')
  const huihui = view.prepared_models[0]!.manifest_path!
  const plain = 'S:\\bubo-walk\\E21\\Import\\Qwen3.8-27B-UD-Q8_K_XL.gguf'
  // A package is saved and says it is already abliterated.
  view.saved.manifest_path = huihui
  view.saved.model_path = undefined
  view.saved_estimate.capabilities = {
    refusal_direction: {
      available: false,
      reason: 'This model is already abliterated.',
    },
  }
  for (const name of ['--mtp-model', '--mmproj']) {
    view.switches.push({
      ...view.switches.find((row) => row.name === '--reasoning-effort')!,
      name,
      label: name,
      gui_control: true,
    })
  }
  // The plain file in the picker can use the setting.
  const estimate = vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: {
        ...view.saved_estimate,
        capabilities: { refusal_direction: { available: true } },
      },
      fit: view.fit,
    }),
  )
  mountCorePage('/', view)
  const picker = page.getByRole('combobox', { name: 'Model', exact: true })
  const select = page.getByRole('combobox', {
    name: 'In-engine abliteration (Qwen3.8 Flash Next)',
  })
  const files = page.getByRole('button', { name: 'Browse for MTP draft file' })
  await expect.element(select).toBeDisabled()
  await expect
    .element(page.getByText('Already abliterated', { exact: true }))
    .toBeVisible()
  expect(files.query()).toBeNull()
  await picker.selectOptions(plain)
  await vi.waitFor(() => expect(estimate).toHaveBeenCalled())
  await expect.element(select).toBeEnabled()
  await expect
    .element(page.getByText('Already abliterated', { exact: true }))
    .not.toBeInTheDocument()
  await expect.element(files).toBeVisible()
  // One line for the pair of rows, not one under each.
  await expect
    .element(page.getByText('Choosing a file saves the model you picked.'))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Browse for mmproj file' }))
    .toBeVisible()
  // Back to the saved package: its facts again.
  await picker.selectOptions(huihui)
  await expect.element(select).toBeDisabled()
  await expect
    .element(page.getByText('Already abliterated', { exact: true }))
    .toBeVisible()
  expect(files.query()).toBeNull()
})

test('a helper file chosen for a model picked and not saved saves that model first', async () => {
  const view = coreView('stopped-saved-estimate')
  const huihui = view.prepared_models[0]!.manifest_path!
  const plain = 'S:\\bubo-walk\\E21\\Import\\Qwen3.8-27B-UD-Q8_K_XL.gguf'
  view.saved.manifest_path = huihui
  view.saved.model_path = undefined
  view.switches.push({
    ...view.switches.find((row) => row.name === '--reasoning-effort')!,
    name: '--mtp-model',
    label: '--mtp-model',
    gui_control: true,
  })
  const calls: string[] = []
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockImplementation(async (request) => {
      calls.push(JSON.stringify(request))
      return bridgeReply({})
    })
  vi.spyOn(actions, 'BrowseSideFile').mockImplementation(async () => {
    calls.push('browse')
    return { cancelled: false, file: 'C:\\side\\mtp.gguf' } as never
  })
  mountCorePage('/', view)
  await page
    .getByRole('combobox', { name: 'Model', exact: true })
    .selectOptions(plain)
  await page.getByRole('button', { name: 'Browse for MTP draft file' }).click()
  await vi.waitFor(() => expect(patch).toHaveBeenCalledTimes(2))
  expect(calls[0]).toBe(
    JSON.stringify({ set: { model_path: plain }, clear: ['manifest_path'] }),
  )
  expect(calls[1]).toBe('browse')
})

test('the restart notice uses the GUI words for a setting it presents and the CLI words for the rest', async () => {
  const view = coreView('stopped-saved-estimate')
  view.serving = true
  view.restart_needed = [
    {
      key: '--kv-cache-type',
      label: 'Smaller context cache (Q8)',
      running: 'f16',
      next: 'q8_0',
    },
    {
      key: '--refusal-direction',
      label: 'In-engine abliteration (Qwen3.8 Flash Next)',
      running: 'off',
      next: 'on',
    },
    { key: 'sessions', label: 'Sessions', running: '1', next: '3' },
  ]
  mountCorePage('/', view)
  await expect.element(page.getByText('KV cache: f16 → Q8')).toBeVisible()
  await expect
    .element(
      page.getByText('In-engine abliteration (Qwen3.8 Flash Next): Off → On'),
    )
    .toBeVisible()
  await expect.element(page.getByText('Sessions: 1 → 3')).toBeVisible()
  expect(document.body.textContent).not.toContain('Refusal direction')
  expect(document.body.textContent).not.toContain('Smaller context cache')
})

test('a saved model the scan has not listed is named by its file, not by the active model', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.model_path = 'S:\\models\\Unlisted-Model-00001-of-00002.gguf'
  view.saved.manifest_path = undefined
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('combobox', { name: 'Model' }))
    .toHaveDisplayValue('Unlisted-Model-00001-of-00002.gguf')
})

test('KV cache shows f16, disabled, with the CLI reason when the model cannot use Q8', async () => {
  const view = coreView('stopped-saved-estimate')
  view.switch_overrides.find((row) => row.name === '--kv-cache-type')!.saved =
    'q8_0'
  view.saved_estimate.capabilities = {
    kv_cache_q8: {
      available: false,
      reason: 'Q8 is for Qwen3.8 Flash Next only.',
    },
  }
  mountCorePage('/', view)
  const select = page.getByRole('combobox', { name: 'KV cache' })
  await expect.element(select).toBeDisabled()
  await expect.element(select).toHaveDisplayValue('f16')
  await expect
    .element(page.getByText('Unsupported model', { exact: true }))
    .toBeVisible()
  await page.getByRole('button', { name: 'About KV cache' }).click()
  await expect
    .element(page.getByText('Q8 is for Qwen3.8 Flash Next only.'))
    .toBeVisible()
  // The remembered Q8 cannot be used here; it can still be cleared.
  await page.getByRole('button', { name: 'About KV cache' }).click()
  await expect
    .element(page.getByRole('button', { name: 'Reset KV cache to default' }))
    .toBeEnabled()
})
