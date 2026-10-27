import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from './core-page-test-support'
import { appendSample } from '@/lib/gpu-history'

afterEach(() => vi.restoreAllMocks())

test('first-run completion starts Bubo, shows Starting, then copies its running address and stops', async () => {
  const view = coreView('stopped-saved-estimate')
  const start = vi.spyOn(actions, 'Start').mockResolvedValue({})
  const stop = vi.spyOn(actions, 'Stop').mockResolvedValue({})
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const { replaceView } = mountCorePage('/onboarding', view)
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('(default model) · Done')
  expect(
    page
      .getByRole('button', { name: 'Start server', exact: true })
      .element()
      .getBoundingClientRect().bottom,
  ).toBeLessThan(document.documentElement.clientHeight)
  await page.getByRole('button', { name: 'Start server', exact: true }).click()
  expect(start).toHaveBeenCalledExactlyOnceWith()
  await page
    .getByRole('link', { name: 'Show server setup and addresses' })
    .click()
  replaceView(coreView('starting'))
  await expect
    .element(page.getByRole('button', { name: 'Stop server', exact: true }))
    .toBeEnabled()
  const ready = coreView('ready-idle')
  ready.endpoints = {
    openai_base_url: 'https://example.test/server',
    anthropic_base_url: '',
  }
  replaceView(ready)
  await page
    .getByRole('button', { name: 'Copy OpenAI address', exact: true })
    .click()
  expect(copy).toHaveBeenCalledWith(ready.endpoints.openai_base_url)
  await page.getByRole('button', { name: 'Stop server', exact: true }).click()
  expect(stop).toHaveBeenCalledExactlyOnceWith()
  replaceView(view)
  await expect
    .element(page.getByRole('button', { name: 'Stop server', exact: true }))
    .not.toBeInTheDocument()
})

test('Check again refreshes authoritative model checks and exposes typed failures', async () => {
  const refresh = vi.spyOn(actions, 'RefreshModels').mockResolvedValue({
    error: {
      code: 'fixture_refresh',
      component: '',
      stage: '',
      retryable: false,
      remedy: 'Fixture retry failed.',
    },
  })
  mountCorePage('/', coreView('memory-gpu-headroom'))
  await page.getByRole('button', { name: 'Check again', exact: true }).click()
  expect(refresh).toHaveBeenCalledExactlyOnceWith()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture retry failed.')
})

test('stopped state hides inapplicable actions; override has one recovery box', async () => {
  const { replaceView } = mountCorePage('/', coreView('memory-gpu-headroom'))
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  for (const name of ['Stop server', 'Restart server'])
    await expect
      .element(page.getByRole('button', { name, exact: true }))
      .not.toBeInTheDocument()
  replaceView(coreView('switch-override-rejected'))
  const box = page.getByRole('region', { name: 'Rejected override' })
  await expect
    .element(box)
    .toHaveTextContent('Reasoning length setting you changed')
  await expect
    .element(box.getByRole('button', { name: 'Reset Reasoning length' }))
    .toBeEnabled()
  await expect
    .element(box.getByRole('link', { name: 'Open setting' }))
    .toHaveAttribute('href', '/#server-setup')
  await expect
    .element(page.getByLabelText('Memory fit'))
    .not.toBeInTheDocument()
  expect(document.querySelectorAll('[role="alert"]')).toHaveLength(1)
  await expect
    .element(box.getByRole('alert').getByText('Error', { exact: true }))
    .toBeVisible()
})

test('missing-runtime recovery stays inside its error and does not show command instructions', async () => {
  const prepare = vi.spyOn(actions, 'Prepare').mockResolvedValue({})
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
  mountCorePage('/', view)
  // A missing engine is a state: one button, no red box.
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  await page
    .getByRole('button', { name: 'Download the engine files', exact: true })
    .click()
  expect(prepare).toHaveBeenCalledExactlyOnceWith()
  await expect
    .element(
      page.getByText('runtime_missing: Run bubo prepare runtime.', {
        exact: true,
      }),
    )
    .not.toBeInTheDocument()
})

test('GPU history records exactly the View GPU observation without request or wall rates', () => {
  const view = coreView('ready-busy')
  const last = appendSample([], view, 123).at(-1)!
  expect(last.gpu).toEqual(view.gpu)
  expect(last).not.toHaveProperty('decode')
  expect(last).not.toHaveProperty('prefill')
})

test.each(['preparation', 'model'] as const)(
  '%s download keeps the supervisor ordinal of the current file under Details',
  async (kind) => {
    const view = coreView(
      kind === 'preparation' ? 'preparing' : 'models-download-in-progress',
    )
    // Go's whole-job number is what makes the current file a detail.
    const overall = { bytes_done: 3, bytes_total: 12 }
    if (kind === 'preparation') {
      view.preparation_file = { index: 2, total: 8 }
      view.progress!.overall = overall
    } else {
      view.models!.download!.file_position = { index: 2, total: 8 }
      view.models!.download!.overall = overall
    }
    mountCorePage(kind === 'preparation' ? '/onboarding' : '/models', view)
    await expect.element(page.getByText(/^File 2 of 8: /)).toBeInTheDocument()
    await expect.element(page.getByRole('progressbar')).toBeVisible()
  },
)
