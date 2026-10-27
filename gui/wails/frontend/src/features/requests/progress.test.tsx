import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { renderView } from '@/components/test-view'
import { viewKey } from '@/lib/bindings'
import RequestsPage from './index'
import facts from './progress-cli-facts.json'
import { replaySlots, promptReadingView } from './progress-scenario'

let cleanup = () => {}
afterEach(() => cleanup())

test('a live prompt reports only the server counts, including cached tokens', async () => {
  const view = promptReadingView()
  const mounted = renderView(<RequestsPage />, view)
  cleanup = mounted.cleanup
  const live = page.getByRole('region', { name: 'Live requests' })
  await expect
    .element(live.getByRole('columnheader', { name: 'Progress' }))
    .toBeVisible()
  await expect.element(live).toHaveTextContent('1,345 / 1,373')
  await expect.element(live).toHaveTextContent('1,345 cached')
  await expect
    .element(live.getByRole('progressbar'))
    .toHaveAttribute('aria-valuenow', '1345')
  await expect
    .element(live.getByRole('progressbar'))
    .toHaveAttribute('aria-valuemax', '1373')
  expect(
    getComputedStyle(live.getByRole('progressbar').element().firstElementChild!)
      .backgroundColor,
  ).toBe('rgb(36, 92, 100)')
  expect(live.element().textContent).not.toContain('ETA')
  // Repeated observations do not advance the bar on a timer.
  mounted.client.setQueryData(viewKey, structuredClone(view))
  await expect
    .element(live.getByRole('progressbar'))
    .toHaveAttribute('aria-valuenow', '1345')
  const done = structuredClone(view)
  done.slots = replaySlots('writing')
  mounted.client.setQueryData(viewKey, done)
  await expect.element(live).toHaveTextContent('Writing')
  await expect.element(live).toHaveTextContent('14')
})

test('missing progress remains unavailable and a reported zero remains zero', async () => {
  const view = promptReadingView()
  view.slots = replaySlots('unknown')
  cleanup = renderView(<RequestsPage />, view).cleanup
  const live = page.getByRole('region', { name: 'Live requests' })
  await expect.element(live).toHaveTextContent('Progress unavailable')
  expect(live.element().querySelector('[role="progressbar"]')).toBeNull()
})

test('a cold prompt starts at zero and stale slots vanish after disconnect', async () => {
  const view = promptReadingView()
  view.slots = replaySlots('cold')
  const mounted = renderView(<RequestsPage />, view)
  cleanup = mounted.cleanup
  const live = page.getByRole('region', { name: 'Live requests' })
  await expect.element(live).toHaveTextContent('0 / 493')
  await expect
    .element(live.getByRole('progressbar'))
    .toHaveAttribute('aria-valuenow', '0')
  mounted.client.setQueryData(viewKey, { ...view, serving: false })
  await expect.element(live).not.toBeInTheDocument()
})

test('the shared CLI replay keeps coarse prompt counts unchanged between chunks', async () => {
  const view = promptReadingView()
  const mounted = renderView(<RequestsPage />, view)
  cleanup = mounted.cleanup
  const live = page.getByRole('region', { name: 'Live requests' })
  await expect
    .element(live.getByRole('progressbar'))
    .toHaveAttribute('aria-valuenow', '1345')
  for (const frame of facts.replay.filter((frame) =>
    frame.name.startsWith('long'),
  )) {
    for (let poll = 0; poll < frame.polls; poll++) {
      mounted.client.setQueryData(viewKey, {
        ...view,
        slots: replaySlots(frame.name),
      })
      const slot = frame.slots[0]!
      await expect
        .element(live.getByRole('progressbar'))
        .toHaveAttribute(
          'aria-valuenow',
          String(slot.cached_prompt_tokens! + slot.prefilled_tokens!),
        )
      await expect
        .element(live.getByRole('progressbar'))
        .toHaveAttribute('aria-valuemax', String(slot.prompt_tokens))
    }
  }
})
