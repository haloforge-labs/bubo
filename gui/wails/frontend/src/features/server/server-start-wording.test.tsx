import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

// Exercise the rendered Server header, including the interval before GetView
// reports Starting and old-feed flags retained while the model is loading.
afterEach(() => vi.restoreAllMocks())

for (const stale of [false, true]) {
  test(`starting header asks to wait with stale=${stale}`, async () => {
    const view = coreView('starting')
    view.stale = stale
    mountCorePage('/', view)
    await expect
      .element(page.getByLabelText('Server status').getByRole('status'))
      .toHaveTextContent(`${view.status_label} · Please wait`)
    expect(document.body.textContent).not.toContain('Status is stale')
    expect(document.body.textContent).not.toContain('Not updating')
  })
}

test('loading suppresses old-feed warnings until the server is running', async () => {
  const view = coreView('starting')
  view.stale = true
  view.status_label = 'Loading the model…'
  view.status_feed = {
    state: 'not_updating',
    sample_time: '',
    age_seconds: 12,
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Server status').getByRole('status'))
    .toHaveTextContent('Loading the model… · Please wait')
  expect(document.body.textContent).not.toContain('not updating')
  expect(document.body.textContent).not.toContain('Status is stale')
})

test.each(['stopped', 'failed-with-remedy'])(
  '%s never shows a stale-feed header',
  async (scenario) => {
    const view = coreView(scenario)
    view.stale = true
    mountCorePage('/', view)
    await expect
      .element(page.getByLabelText('Server status').getByRole('status'))
      .toHaveTextContent(view.status_label)
    expect(document.body.textContent).not.toContain('Status is stale')
    expect(document.body.textContent).not.toContain('Not updating')
  },
)

test('a running server with a lost feed says Not updating', async () => {
  const view = coreView('ready-idle')
  view.stale = true
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Server status').getByRole('status'))
    .toHaveTextContent('Running · Not updating')
  expect(document.body.textContent).not.toContain('Status is stale')
})

test('a fresh ready server has no waiting or stale suffix', async () => {
  const view = coreView('ready-idle')
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Server status').getByRole('status'))
    .toHaveTextContent('Running')
  expect(document.body.textContent).not.toContain('Please wait')
  expect(document.body.textContent).not.toContain('Not updating')
})

test.each(['Start', 'Restart'] as const)(
  '%s asks to wait immediately before a fresh View arrives',
  async (action) => {
    const view = coreView(
      action === 'Start' ? 'stopped-saved-estimate' : 'ready-idle',
    )
    view.stale = true
    view.actions[action === 'Start' ? 'start' : 'restart'] = {
      allowed: true,
      reason_code: '',
      reason: '',
    }
    let complete!: (value: Awaited<ReturnType<typeof actions.Start>>) => void
    vi.spyOn(actions, action).mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve
        }),
    )
    mountCorePage('/', view)
    try {
      await page
        .getByRole('button', { name: `${action} server`, exact: true })
        .click()
      await expect
        .element(page.getByLabelText('Server status').getByRole('status'))
        .toHaveTextContent(`${view.status_label} · Please wait`)
      expect(document.body.textContent).not.toContain('Status is stale')
      expect(document.body.textContent).not.toContain('Not updating')
    } finally {
      complete?.({})
    }
  },
)
