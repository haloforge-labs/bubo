import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { renderView } from '@/components/test-view'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { viewKey } from '@/lib/bindings'
import RequestsPage from './index'
import {
  firstCompletionView,
  liveRequestView,
} from './live-request-test-support'
import { replaySlots } from './progress-scenario'

let cleanup = () => {}
afterEach(() => cleanup())

test('recorded writing stays live and every active session has its own phase and counters', async () => {
  const view = liveRequestView()
  view.slots.push({ ...replaySlots('cached')[0]!, id: 1 })
  cleanup = renderView(<RequestsPage />, view).cleanup
  const live = page.getByRole('region', { name: 'Live requests', exact: true })
  await expect.element(live).toBeVisible()
  await expect.element(live).toHaveTextContent('Writing')
  await expect.element(live).toHaveTextContent('14')
  await expect.element(live).toHaveTextContent('54.0')
  await expect.element(live).toHaveTextContent('Reading')
  await expect.element(live).toHaveTextContent('1,345 / 1,373')
  await expect.element(live).toHaveTextContent('1,345 cached')
  await expect.element(live).toHaveTextContent('97%')
  await expect.element(live).toHaveTextContent('Observed elapsed')
  await expect
    .element(live.getByRole('table', { name: 'Active sessions' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .toHaveTextContent('60')
})

test('an unknown live rate remains a dash and prefill has a live chart', async () => {
  const view = liveRequestView('reading')
  Object.assign(view.slots[0]!, { prefill_tps: undefined })
  cleanup = renderView(<RequestsPage />, view).cleanup
  const live = page.getByRole('region', { name: 'Live requests', exact: true })
  await expect.element(live).toBeVisible()
  expect(live.element().textContent).toMatch(/—|-/)
  expect(live.element().textContent).not.toContain('0.0 tok/s')
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .toBeVisible()
})

test('the first completion immediately gets request charts while its live chart remains briefly', async () => {
  const mounted = renderView(<RequestsPage />, liveRequestView())
  cleanup = mounted.cleanup
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .toBeVisible()
  mounted.client.setQueryData(viewKey, firstCompletionView())
  await expect
    .element(
      page.getByRole('figure', {
        name: 'Decode per completed request',
        exact: true,
      }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .toBeVisible()
  const later = firstCompletionView()
  later.status_feed!.sample_time = '2026-10-26T23:15:10Z'
  Object.assign(later, { live_samples: [] })
  mounted.client.setQueryData(viewKey, later)
  await expect
    .element(page.getByRole('figure', { name: /Live request rates/ }).first())
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByRole('figure', {
        name: 'Decode per completed request',
        exact: true,
      }),
    )
    .toBeVisible()
})

test.each([
  ['writing', 'Writing · 54.0 tok/s'],
  ['reading', 'Reading prompt · 97%'],
  ['idle', 'Idle'],
] as const)(
  'running Server header reports %s from the same live facts',
  async (phase, expected) => {
    const view =
      phase === 'idle' ? coreView('ready-idle') : liveRequestView(phase)
    mountCorePage('/', view)
    await expect
      .element(page.getByLabelText('Server status'))
      .toHaveTextContent(expected)
  },
)
