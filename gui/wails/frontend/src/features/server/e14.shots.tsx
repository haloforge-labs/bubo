import { expect, inject, test } from 'vitest'
import { commands, page, userEvent } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  // Park the pointer on the page title: no button wears a hover shade.
  await userEvent.hover(page.getByRole('heading', { level: 1 }).first())
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

test('E14: a catalogue model that is not the default is installed and selectable', async () => {
  const view = coreView('stopped-saved-estimate')
  const other = view.models!.catalogue.find(
    (entry) => entry.default_status !== 'default',
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
    },
  ]
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Installed models' }))
    .toHaveTextContent(other.display_name)
  await shot('models', 'e14-catalogue-model-installed')
})

test('E14: a lifecycle warning of the server', async () => {
  const view = coreView('ready-idle')
  view.server_warnings = [
    { code: 'queue_pressure', remedy: 'Wait for a free session.' },
  ]
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Other GPU work is slowing the server.', { exact: true }),
    )
    .toBeVisible()
  await shot('server', 'e14-server-warning')
})

// A prepared setup (the engine is ready), as a real refusal or a missing
// runtime would leave it.
function failedView(
  failure: NonNullable<ReturnType<typeof coreView>['failure']>,
) {
  const view = coreView('stopped-saved-estimate')
  view.phase = 'failed'
  view.status_label = 'Stopped after an error'
  view.failure = failure
  view.reason = failure.code
  return view
}

test('E14: a missing Visual C++ runtime', async () => {
  const view = failedView({
    code: 'vcredist_missing',
    source: 'supervisor',
    time: '',
    remedy: '',
    forced: false,
  })
  // The CLI cannot run, so nothing is known about the engine or the memory.
  view.actions.start = {
    allowed: false,
    reason_code: 'readiness_unavailable',
    reason: 'The CLI has not reported readiness.',
  }
  view.fit = { ...view.fit, class: 'unavailable', sentence: 'Checking memory…' }
  mountCorePage('/', view)
  await expect
    .element(page.getByText('Bubo needs the Microsoft Visual C++ runtime.'))
    .toBeVisible()
  await shot('server', 'e14-vcruntime-missing')
})

test('E14: the memory guard refused the start', async () => {
  const view = failedView({
    code: 'memory_insufficient',
    source: 'cli',
    time: '',
    remedy: 'Lower --context or --sessions, or free memory.',
    forced: false,
  })
  const estimate = coreView('memory-smaller-context').memory_estimate!
  view.saved.sessions = estimate.sessions
  view.saved.context = estimate.context_tokens
  view.failure!.memory = {
    needed_bytes: estimate.gpu!.need.bytes,
    available_bytes: estimate.gpu!.available!.bytes,
    sessions: estimate.sessions,
    context: estimate.context_tokens,
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByText(/The server didn't start: /))
    .toBeVisible()
  await shot('server', 'e14-memory-insufficient')
})

test('E14: the first call waits and the page says so', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('stopped-saved-estimate')
  view.startup_phase = 'starting_cli'
  view.initialized = false
  view.prepared = false
  view.phase = 'checking'
  view.status_label = 'Starting for the first time…'
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('main'))
    .toHaveTextContent('Starting for the first time…')
  await document.fonts.ready
  await shot('server', 'e14-first-start')
})

test('E14: Requests names the outcome and cause of a request that did not complete', () => {
  const view = fixture('ready-busy')
  const template = view.requests[0]!
  // A request that never ran has no tokens, no timing and no cache.
  const never = {
    slot: undefined,
    prompt_tokens: undefined,
    completion_tokens: undefined,
    ttft_ms: undefined,
    prefill_tps: undefined,
    decode_tps: undefined,
    cached_tokens: undefined,
    cache_kind: undefined,
    finish_reason: undefined,
  }
  view.requests = [
    {
      ...template,
      id: 'r43',
      time: new Date(Date.parse(template.time) - 4_000).toISOString(),
      outcome: 'failed',
      status: 500,
      error_code: 'device_lost',
      ...never,
    },
    {
      ...template,
      id: 'r42',
      time: new Date(Date.parse(template.time) - 19_000).toISOString(),
      outcome: 'refused',
      status: 503,
      error_code: 'insufficient_memory',
      refusal: {
        code: 'insufficient_memory',
        needed_bytes: 8589934592,
        free_bytes: 2147483648,
      },
      ...never,
    },
    {
      ...template,
      id: 'r41',
      time: new Date(Date.parse(template.time) - 41_000).toISOString(),
    },
  ]
  return capturePage({ feature: 'requests', scenario: 'e14-failed-rows', view })
})
