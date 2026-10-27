import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from './core-page-test-support'
import { bridgeReply } from './bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('preparing recording is consistent and only reports the runtime done', async () => {
  const view = coreView('preparing')
  expect(view.runtime.readiness.model).toBe(false)
  expect(view.prepared).toBe(false)
  mountCorePage('/onboarding', view)
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('Engine files · Done')
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('(default model) · In progress')
})

test('status controls stay in the first viewport and dirty setup permits save-and-restart while Stop remains available', async () => {
  const view = coreView()
  view.actions.restart = { allowed: true, reason_code: '', reason: '' }
  const stop = vi.spyOn(actions, 'Stop').mockResolvedValue(bridgeReply({}))
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: { memory_estimate: view.memory_estimate } }),
  )
  mountCorePage('/', view)
  const status = page.getByLabelText('Server status')
  const restart = status.getByRole('button', {
    name: 'Restart server',
    exact: true,
  })
  await expect.element(restart).toBeEnabled()
  expect(status.element().getBoundingClientRect().bottom).toBeLessThan(
    document.documentElement.clientHeight,
  )
  expect(
    document.querySelector('form')!.querySelectorAll('button').length,
  ).toBeGreaterThan(0)
  expect(document.querySelector('form')!.textContent).not.toContain(
    'Stop server',
  )
  await page
    .getByRole('spinbutton', { name: 'Sessions', exact: true })
    .fill('2')
  await expect.element(restart).toBeEnabled()
  expect(status.element().textContent).not.toContain(
    'Save server setup before starting or restarting.',
  )
  await status.getByRole('button', { name: 'Stop server', exact: true }).click()
  expect(stop).toHaveBeenCalledOnce()
})

test('picker shows the CLI prepared default and saved model takes precedence', async () => {
  const view = coreView('stopped-saved-estimate')
  const { replaceView } = mountCorePage('/', view)
  const picker = page.getByRole('combobox', { name: 'Model', exact: true })
  await expect.element(picker).toHaveValue(view.prepared_default!)
  const saved = structuredClone(view)
  saved.saved.manifest_path = saved.prepared_models[0]!.manifest_path
  replaceView(saved)
  await expect.element(picker).toHaveValue(saved.saved.manifest_path!)
})

test('latest completed request owns the large rates and failed rows cannot replace it', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  view.requests.push({
    ...request,
    id: 'fixture-failed',
    outcome: 'failed',
    decode_tps: 9999,
    prefill_tps: 9999,
  })
  mountCorePage('/', view)
  const reported = page.getByRole('region', { name: 'Reported request rates' })
  await expect
    .element(reported)
    .toHaveTextContent(
      new Date(request.time).toLocaleTimeString(undefined, { hour12: false }),
    )
  expect(reported.element().textContent).not.toContain(request.id)
  await expect
    .element(reported.getByRole('figure', { name: 'Decode tok/s' }))
    .toHaveTextContent(request.decode_tps!.toFixed(1))
  await expect
    .element(reported.getByRole('figure', { name: 'Prefill tok/s' }))
    .toHaveTextContent(request.prefill_tps!.toFixed(1))
  expect(reported.element().querySelectorAll('.text-3xl')).toHaveLength(
    request.ttft_ms == null ? 2 : 3,
  )
  expect(reported.element().textContent).not.toContain('9999')
})

test('no completed request stays missing rather than promoting a wall rate', async () => {
  const view = coreView('ready-busy')
  view.requests = []
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Reported request rates' }))
    .toHaveTextContent('No requests yet.')
  // No placeholder figures: the sentence above is the whole statement.
  await expect
    .element(page.getByRole('figure', { name: 'Decode tok/s', exact: true }))
    .not.toBeInTheDocument()
})

test('the Models page decides on typed fields, never on loader prose', async () => {
  const view = coreView()
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Discover models' }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain(
    'Not loadable in this version',
  )
  await expect
    .element(page.getByRole('region', { name: 'Discover models' }))
    .toHaveTextContent(
      "Bubo's server holds the memory now. Stop it to switch models.",
    )
})

test('port keeps the CLI text control, saved scalar and canonical preference write route', async () => {
  const view = coreView()
  const row = view.switches.find((row) => row.name === '--port')!
  const state = view.switch_overrides.find((entry) => entry.name === row.name)!
  view.saved.port = 19001
  state.saved = String(view.saved.port)
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  const input = page.getByRole('textbox', { name: row.label! })
  await expect.element(input).toHaveValue(state.saved)
  expect(input.element().closest('[data-setting]')?.textContent).toContain(
    `Default: ${row.default_text}`,
  )
  await input.fill('19002')
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: row.name, value: '19002' }],
  })
})

test('equal context has no redundant Use action', async () => {
  const view = coreView('stopped-saved-estimate')
  const memory = view.saved_estimate.memory_estimate!
  const preset = memory.recommended_context!.by_sessions.find(
    (row) => row.sessions === memory.sessions,
  )!
  view.saved.context = preset.context_per_session
  mountCorePage('/', view)
  await expect
    .element(
      page.getByRole('button', {
        name: `Use ${preset.context_per_session.toLocaleString()} tokens`,
      }),
    )
    .not.toBeInTheDocument()
})

test('GPU timing with neither age nor window is omitted', async () => {
  const view = coreView('ready-busy')
  view.gpu = { use_pct: 73 }
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('region', { name: 'GPU', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('GPU sample:', { exact: false }))
    .not.toBeInTheDocument()
})

test('Server has one state and address control set while other pages retain the header summary', async () => {
  const view = coreView()
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Addresses', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByLabelText('Server addresses'))
    .not.toBeInTheDocument()
  expect(
    page
      .getByRole('button', { name: 'Copy OpenAI address', exact: true })
      .elements(),
  ).toHaveLength(1)
  expect(page.getByRole('status').elements()).toHaveLength(1)
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click()
  await expect.element(page.getByLabelText('Server addresses')).toBeVisible()
  await expect
    .element(page.getByRole('status').first())
    .toHaveTextContent('Running')
})
