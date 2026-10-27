import { bridgeReply } from '@/components/bridge-test-reply'
import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { activeModelName } from '@/lib/format'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('Starting exposes the Stop action allowed by the View', async () => {
  const view = coreView('starting')
  expect(view.actions.stop.allowed).toBe(true)
  const stop = vi.spyOn(actions, 'Stop').mockResolvedValue({})
  const start = vi.spyOn(actions, 'Start')
  mountCorePage('/', view)
  await page.getByRole('button', { name: 'Stop server', exact: true }).click()
  expect(stop).toHaveBeenCalledOnce()
  expect(start).not.toHaveBeenCalled()
})

test('pre-start setup shares the saved CLI estimate for fit, presets and initial inputs', async () => {
  const view = coreView('stopped-saved-estimate')
  expect(view.memory_estimate).toBeNull()
  const memory = view.saved_estimate.memory_estimate!
  const estimate = vi.spyOn(actions, 'Estimate')
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Memory fit'))
    .toHaveTextContent('Fits')
  const preset = memory.recommended_context!.by_sessions.find(
    (row) => row.sessions === memory.sessions,
  )!
  await expect
    .element(
      page.getByRole('button', {
        name: `Use ${preset.context_per_session.toLocaleString()} tokens`,
        exact: true,
      }),
    )
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('spinbutton', { name: 'Context per session' }))
    .toHaveValue(memory.context_tokens.toLocaleString())
  await expect
    .element(page.getByRole('spinbutton', { name: 'Sessions', exact: true }))
    .toHaveValue(memory.sessions.toLocaleString())
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  expect(estimate).not.toHaveBeenCalled()
})

test('saved estimate arrival hydrates a pristine first-start form', async () => {
  const view = coreView('stopped-saved-estimate')
  const saved = view.saved_estimate
  view.saved_estimate = {}
  const { replaceView } = mountCorePage('/', view)
  replaceView({ ...view, saved_estimate: saved })
  await expect
    .element(page.getByRole('spinbutton', { name: 'Context per session' }))
    .toHaveValue(saved.memory_estimate!.context_tokens.toLocaleString())
})

test.each(['start', 'stop', 'restart'] as const)(
  'the %s control renders only the View permission',
  async (action) => {
    const view = coreView()
    if (action === 'start') {
      view.state = 'Stopped'
      view.instance = undefined
    }
    const label =
      action === 'start'
        ? 'Start server'
        : action === 'stop'
          ? 'Stop server'
          : 'Restart server'
    view.actions[action] = {
      allowed: false,
      reason_code: 'fixture_permission',
      reason: `Fixture ${action} permission denied.`,
    }
    const { replaceView } = mountCorePage('/', view)
    const button = page.getByRole('button', { name: label, exact: true })
    await expect.element(button).toBeDisabled()
    replaceView({
      ...view,
      actions: {
        ...view.actions,
        [action]: { allowed: true, reason_code: '', reason: '' },
      },
    })
    await expect.element(button).toBeEnabled()
  },
)

test('Server renders the CLI model, context and sessions', async () => {
  const view = coreView()
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByLabelText('Model in use'))
    .toHaveTextContent(activeModelName(view) || view.instance!.served_model!)
  await expect
    .element(page.getByRole('combobox', { name: 'Model for next start' }))
    .toBeVisible()
  await expect
    .element(page.getByLabelText('Context per session'))
    .toHaveValue(
      Number(
        view.settings.find((row) => row.name === '--context')!.effective.text,
      ).toLocaleString(),
    )
  await expect
    .element(page.getByLabelText('Sessions', { exact: true }).first())
    .toHaveValue(
      Number(
        view.settings.find((row) => row.name === '--sessions')!.effective.text,
      ).toLocaleString(),
    )
})

test('only supplied GPU fields are displayed', async () => {
  const view = coreView('ready-busy')
  view.gpu = { use_pct: 0, chip_power_w: 73 }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('figure', { name: 'Decode tok/s' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'Prefill tok/s' }))
    .toBeVisible()
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('figure', { name: 'Chip power (CPU, GPU, NPU)' }))
    .toBeInTheDocument()
  await expect
    .element(page.getByText('GPU temperature', { exact: true }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByText('GPU memory', { exact: true }))
    .not.toBeInTheDocument()
})

test('Stop and Restart call their typed actions and a failed action stays visible', async () => {
  const stop = vi.spyOn(actions, 'Stop').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'fixture_failure',
        remedy: 'Fixture stop remedy.',
        component: '',
        stage: '',
        retryable: true,
      },
    }),
  )
  const restart = vi
    .spyOn(actions, 'Restart')
    .mockResolvedValue(bridgeReply({}))
  const view = coreView()
  view.memory_estimate = { ...view.memory_estimate!, fits: true }
  view.saved_estimate = { memory_estimate: view.memory_estimate }
  view.actions.start = { allowed: true, reason_code: '', reason: '' }
  view.actions.restart = { allowed: true, reason_code: '', reason: '' }
  mountCorePage('/', view)
  await page.getByRole('button', { name: 'Stop server', exact: true }).click()
  expect(stop).toHaveBeenCalledOnce()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_failure: Fixture stop remedy.')
  await page
    .getByRole('button', { name: 'Restart server', exact: true })
    .click()
  expect(restart).toHaveBeenCalledOnce()
})

test('drafts are validated and saved through a partial patch; CLI rejection stays in place', async () => {
  const view = coreView()
  view.saved.manifest_path = view.prepared_models[0]!.manifest_path
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue(
    bridgeReply({
      error: { code: 'fixture_rejected', remedy: 'Check the server setup.' },
    }),
  )
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: {
        subject: 'fixture',
        subject_kind: 'file',
        estimate_state: 'estimated',
        fit_verified: false,
        fit_inputs: {},
        notes: [],
        memory_estimate: view.memory_estimate,
      },
    }),
  )
  mountCorePage('/', view)
  const input = page.getByRole('spinbutton', { name: 'Sessions', exact: true })
  await input.fill(
    String(view.memory_estimate!.recommended_context!.max_sessions + 1),
  )
  await page.getByRole('button', { name: 'Save server setup' }).click()
  await expect.element(page.getByRole('alert').first()).toBeVisible()
  expect(save).not.toHaveBeenCalled()
  await input.fill('3')
  await expect
    .element(page.getByText('Changes will be used after restart.'))
    .toBeVisible()
  await page.getByRole('button', { name: 'Save server setup' }).click()
  const detail = 'fixture_rejected: Check the server setup.'
  await expect
    .element(page.getByRole('alert').filter({ hasText: detail }))
    .toHaveTextContent(detail)
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { sessions: 3 },
    clear: [],
  })
})

test('Restart checks admission after releasing the running allocation even for a changed saved setup', async () => {
  const view = coreView()
  view.saved.manifest_path = view.prepared_models[0]!.manifest_path
  view.restart_needed = [
    { key: 'sessions', label: 'Sessions', running: '1', next: '3' },
  ]
  const restart = vi.spyOn(actions, 'Restart').mockResolvedValue({})
  const estimate = vi.spyOn(actions, 'Estimate')
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Restart server', exact: true }))
    .toBeEnabled()
  await page
    .getByRole('button', { name: 'Restart server', exact: true })
    .click()
  expect(restart).toHaveBeenCalledExactlyOnceWith()
  await expect
    .element(page.getByRole('button', { name: 'Stop server', exact: true }))
    .toBeEnabled()
  expect(estimate).not.toHaveBeenCalled()
})

test('missing endpoints display state; supplied addresses are copied without construction', async () => {
  const view = coreView()
  view.endpoints = undefined
  view.connection = undefined
  const { replaceView } = mountCorePage('/', view)
  // No addresses: no section that says so.
  await expect
    .element(page.getByRole('region', { name: 'Addresses', exact: true }))
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByRole('button', { name: 'Copy OpenAI address', exact: true }),
    )
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByRole('button', { name: 'Copy Anthropic address', exact: true }),
    )
    .not.toBeInTheDocument()
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const endpoints = {
    openai_base_url: 'https://example.test/openai-base',
    anthropic_base_url: 'https://example.test/anthropic-base',
  }
  replaceView({ ...view, endpoints })
  await page
    .getByRole('region', { name: 'Addresses', exact: true })
    .getByRole('button', { name: 'Copy OpenAI address' })
    .click()
  expect(copy).toHaveBeenCalledExactlyOnceWith(endpoints.openai_base_url)
  await page
    .getByRole('region', { name: 'Addresses', exact: true })
    .getByRole('button', { name: 'Copy Anthropic address' })
    .click()
  expect(copy).toHaveBeenLastCalledWith(endpoints.anthropic_base_url)
})

test('stall retains CLI request rates and offers recovery', async () => {
  const view = coreView('stalled')
  const restart = vi
    .spyOn(actions, 'Restart')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('figure', { name: 'Decode tok/s', exact: true }))
    .toHaveTextContent(
      view.requests
        .findLast((row) => row.outcome === 'completed')!
        .decode_tps!.toFixed(1) + ' tok/s',
    )
  await expect
    .element(page.getByRole('figure', { name: 'Prefill tok/s', exact: true }))
    .toHaveTextContent(
      view.requests
        .findLast((row) => row.outcome === 'completed')!
        .prefill_tps!.toFixed(1) + ' tok/s',
    )
  // The sidebar has Logs; the status row has no second link to it.
  await expect
    .element(
      page.getByRole('main').getByRole('link', { name: 'Logs', exact: true }),
    )
    .not.toBeInTheDocument()
  await page
    .getByRole('button', { name: 'Restart server', exact: true })
    .click()
  expect(restart).toHaveBeenCalledOnce()
})

test('history survives page navigation in Query cache and does not mix server instances', async () => {
  const view = coreView()
  const { client, replaceView } = mountCorePage('/', view)
  await expect
    .poll(() => client.getQueriesData({ queryKey: ['gpu-history'] }).length)
    .toBe(1)
  const identity = view.instance!.instance_id
  await expect
    .poll(
      () =>
        client.getQueryData<unknown[]>(['gpu-history', identity])?.length ?? 0,
    )
    .toBeGreaterThanOrEqual(1)
  await page
    .getByRole('navigation')
    .getByRole('link', { name: 'Models', exact: true })
    .click()
  await page
    .getByRole('navigation')
    .getByRole('link', { name: 'Server', exact: true })
    .click()
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  expect(
    client.getQueryData<unknown[]>(['gpu-history', identity])!.length,
  ).toBeGreaterThanOrEqual(1)
  replaceView({
    ...view,
    instance: { ...view.instance!, instance_id: 'synthetic-next-instance' },
  })
  await expect
    .poll(
      () =>
        client.getQueryData<unknown[]>([
          'gpu-history',
          'synthetic-next-instance',
        ])?.length,
    )
    .toBe(1)
})

test('missing registry limits and runtime readiness never offer a successful start', async () => {
  const view = coreView('owner-memory-fit')
  // The engine files are missing: a setup state, not a failure.
  view.phase = 'needs_engine'
  view.status_label = 'Setup needed: download the engine files'
  view.actions.start = {
    allowed: false,
    reason_code: 'runtime_missing',
    reason: 'Download or select the runtime before starting.',
  }
  view.actions.restart = view.actions.start
  view.settings = []
  view.switches = []
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Server setup settings have not been reported yet.'),
    )
    .toBeVisible()
  // Start is not offered while the engine is missing; one button fixes it.
  await expect
    .element(page.getByRole('button', { name: 'Start server' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Download the engine files' }))
    .toBeVisible()
})

test('a stopped, prepared server starts through the typed action', async () => {
  const view = coreView()
  view.state = 'Stopped'
  view.instance = undefined
  view.memory_estimate = { ...view.memory_estimate!, fits: true }
  view.saved_estimate = { memory_estimate: view.memory_estimate }
  view.actions.start = { allowed: true, reason_code: '', reason: '' }
  view.actions.restart = { allowed: true, reason_code: '', reason: '' }
  const start = vi.spyOn(actions, 'Start').mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  await page.getByRole('button', { name: 'Start server', exact: true }).click()
  expect(start).toHaveBeenCalledExactlyOnceWith()
})

test('a display name is never sent to the estimate binding as a catalogue ID', async () => {
  const estimate = vi.spyOn(actions, 'Estimate')
  const view = coreView()
  view.current_model = undefined
  mountCorePage('/', view)
  await page
    .getByRole('spinbutton', { name: 'Sessions', exact: true })
    .fill('3')
  const detail = 'Select an installed model to check memory fit.'
  await expect
    .element(page.getByRole('alert').filter({ hasText: detail }))
    .toHaveTextContent(detail)
  expect(estimate).not.toHaveBeenCalled()
})

test('the active catalogue identity drives estimates even when its served name differs', async () => {
  const view = coreView()
  // The package the page lists (the models list); the running-server status
  // behind this scenario was recorded under another root.
  const model = view.models!.prepared_models[0]!
  view.active_model = {
    catalogue_id: model.id,
    manifest_path: model.manifest_path,
  }
  view.instance!.served_model = 'Custom fixture alias'
  const estimate = vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: { memory_estimate: { ...view.memory_estimate!, fits: true } },
    }),
  )
  const save = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  await page
    .getByRole('spinbutton', { name: 'Sessions', exact: true })
    .fill('3')
  // E19: a package is estimated by its folder, wherever the folder is.
  await expect
    .poll(() => estimate.mock.calls.at(-1)?.[0])
    .toBe(model.manifest_path!.replace(/[\\/][^\\/]*$/, ''))
  await page.getByRole('button', { name: 'Save server setup' }).click()
  expect(save).toHaveBeenLastCalledWith({ set: { sessions: 3 }, clear: [] })
  await page.getByRole('button', { name: 'Reset to default' }).click()
  expect(save).toHaveBeenLastCalledWith({
    set: {},
    clear: ['context', 'sessions', 'model_path', 'manifest_path'],
  })
})
