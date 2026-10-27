import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

// UX10 M1/M2: a failure owns its fix. The status row offers Start only.
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

test('a refused start on a prepared setup shows its cause and Start, never an engine download', async () => {
  const view = failedView({
    code: 'memory_insufficient',
    source: 'cli',
    time: '',
    remedy: 'Lower --context or --sessions, or free memory.',
    detail: 'the CLI text',
    memory: {
      needed_bytes: 50784518144,
      available_bytes: 33503973376,
      sessions: view_sessions(),
      context: 65536,
    },
    forced: false,
  })
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent("The server didn't start: ")
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  expect(
    page.getByRole('button', { name: /engine files/ }).elements(),
  ).toHaveLength(0)
  // The refusal row carries the numbers; no "Fits." line beside it.
  expect(document.body.textContent).not.toContain('Fits.')
})

function view_sessions() {
  return 2
}

test('a CLI that cannot run makes no readiness claim: the runtime row only', async () => {
  const view = failedView({
    code: 'vcredist_missing',
    source: 'supervisor',
    time: '',
    remedy: '',
    forced: false,
  })
  // What the supervisor reports when every CLI call exits 0xC0000135.
  view.actions.start = {
    allowed: false,
    reason_code: 'readiness_unavailable',
    reason: 'The CLI has not reported readiness.',
  }
  // No estimate exists without a CLI.
  view.fit = { ...view.fit, class: 'unavailable', sentence: 'Checking memory…' }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Bubo needs the Microsoft Visual C++ runtime.')
  await expect
    .element(page.getByRole('button', { name: 'Download the runtime' }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('engine files')
  // Nothing to set up while the CLI cannot run (as on cli-missing).
  await expect
    .element(page.getByRole('region', { name: 'Server setup' }))
    .not.toBeInTheDocument()
  expect(
    page.getByRole('button', { name: /Download/ }).elements(),
  ).toHaveLength(1)
})

// A package with no measured memory table has no fit verdict: Start works,
// one plain line says memory is checked at start, and there is no "Fits.".
test('a saved model with no fit verdict is startable and says memory is checked at start', async () => {
  mountCorePage('/', coreView('saved-no-verdict'))
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  await expect
    .element(
      page.getByText('Not checked: you can still start the server.', {
        exact: true,
      }),
    )
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Fits.')
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
})

// UX12: the Sampling line goes with the setup form while the CLI cannot run.
test('the runtime screen hides the Sampling line with the setup form', async () => {
  const view = failedView({
    code: 'vcredist_missing',
    source: 'supervisor',
    time: '',
    remedy: '',
    forced: false,
  })
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Bubo needs the Microsoft Visual C++ runtime.')
  expect(document.body.textContent).not.toContain('Sampling controls')
})

// A pause is cleared at shutdown: a stopped server shows no NPU line.
test('the NPU pause shows only while the server runs', async () => {
  const stopped = coreView('npu-prefill-faulted')
  stopped.serving = false
  stopped.phase = 'stopped'
  stopped.status_label = 'Stopped'
  mountCorePage('/', stopped)
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('region', { name: 'NPU prefill status' }))
    .not.toBeInTheDocument()
})
