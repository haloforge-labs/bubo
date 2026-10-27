import { afterEach, expect, test } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { renderView } from '@/components/test-view'
import RequestsPage from './index'
import { realRequestsView } from './v12-scenario'
import { viewKey } from '@/lib/bindings'

let cleanup: (() => void) | undefined
afterEach(() => cleanup?.())

test('real completed requests have metric points and factual hover and keyboard details', async () => {
  cleanup = renderView(<RequestsPage />, realRequestsView()).cleanup
  const prefill = page.getByRole('figure', {
    name: 'Prefill per completed request',
  })
  const point = prefill.getByRole('button', { name: /^r8 ·/ })
  await expect.element(point).toBeVisible()
  await userEvent.hover(point)
  const details = page.getByRole('tooltip')
  await expect
    .element(details)
    .toHaveTextContent('r8 · 12 s ago · Prefill 1,392 tok/s')
  await expect.element(details).toHaveTextContent('2,606 prompt tokens')
  await expect.element(details).toHaveTextContent('(0 cached)')
  await expect.element(details).toHaveTextContent('1.9 s to first token')
  await expect.element(details).toHaveTextContent('cache miss')
  await userEvent.unhover(point)
  const cached = prefill.getByRole('button', { name: /^r10 ·/ })
  ;(cached.element() as SVGElement).focus()
  await expect.element(details).not.toHaveTextContent('mostly cached')
  await expect.element(details).toHaveTextContent('(1,345 cached)')
  expect(cached.element().getAttribute('fill')).toBe('var(--brand)')
  expect(document.activeElement === cached.element()).toBe(true)
  await userEvent.keyboard('{Escape}')
  await expect.element(details).not.toBeInTheDocument()
  await expect
    .element(
      page.getByRole('figure', {
        name: 'Time to first token per completed request',
      }),
    )
    .toBeVisible()
  expect(document.querySelectorAll('[data-request-id="r4"]')).toHaveLength(0)
  await expect
    .element(page.getByRole('region', { name: 'Requests table' }))
    .toHaveTextContent('r4')
})

test('one reported completed value appears on each applicable chart', async () => {
  const view = realRequestsView()
  view.requests = view.requests.slice(0, 1)
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .element(page.getByRole('region', { name: 'Request performance' }))
    .toHaveTextContent('Each point is one completed request.')
  await expect
    .poll(
      () =>
        page
          .getByRole('region', { name: 'Request performance' })
          .element()
          .querySelectorAll('.recharts-wrapper').length,
    )
    .toBe(3)
})

test('all returned requests remain visible in oldest-first order beyond five minutes', async () => {
  const view = realRequestsView()
  view.requests[0]!.time = '2026-10-25T23:00:00Z'
  view.requests.reverse()
  cleanup = renderView(<RequestsPage />, view).cleanup
  const chart = page.getByRole('figure', {
    name: 'Decode per completed request',
  })
  const oldest = chart.getByRole('button', { name: /^r3 ·/ })
  const newest = chart.getByRole('button', { name: /^r8 ·/ })
  await expect.element(oldest).toBeVisible()
  await expect.element(newest).toBeVisible()
  expect(oldest.element().getBoundingClientRect().left).toBeLessThan(
    newest.element().getBoundingClientRect().left,
  )
  await userEvent.hover(oldest)
  await expect.element(page.getByRole('tooltip')).toBeVisible()
  await userEvent.unhover(chart)
  await expect.element(page.getByRole('tooltip')).not.toBeInTheDocument()
})

test('missing measurements have accurate metric-specific empty text', async () => {
  const view = realRequestsView()
  view.requests.forEach((row) => {
    row.decode_tps = undefined
    row.prefill_tps = undefined
    row.ttft_ms = undefined
  })
  cleanup = renderView(<RequestsPage />, view).cleanup
  const region = page.getByRole('region', { name: 'Request performance' })
  await expect
    .element(region)
    .toHaveTextContent('No decode rates reported for completed requests.')
  await expect
    .element(region)
    .toHaveTextContent('No prefill rates reported for completed requests.')
  await expect
    .element(region)
    .toHaveTextContent(
      'No time to first token reported for completed requests.',
    )
})

test('missing completion age uses the recorded completion timeline without inventing an age', async () => {
  const view = realRequestsView()
  view.last_completed_request!.age_ms = undefined
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .element(page.getByRole('region', { name: 'Request performance' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('region', { name: 'Request performance' }))
    .toHaveTextContent('Times are relative to the latest completion.')
  await expect
    .element(
      page.getByRole('figure', {
        name: 'Decode per completed request',
        exact: true,
      }),
    )
    .toBeVisible()
  const point = page
    .getByRole('figure', { name: 'Decode per completed request', exact: true })
    .getByRole('button')
    .first()
  expect(point.element().getAttribute('aria-label')).toContain(
    'before latest completion',
  )
  expect(point.element().getAttribute('aria-label')).not.toContain('s ago')
})

test('keyboard focus and current details survive a new View observation', async () => {
  const view = realRequestsView()
  const mounted = renderView(<RequestsPage />, view)
  cleanup = mounted.cleanup
  const point = page
    .getByRole('figure', { name: 'Prefill per completed request' })
    .getByRole('button', { name: /^r10 ·/ })
  await expect.element(point).toBeVisible()
  ;(point.element() as SVGElement).focus()
  await expect
    .element(page.getByRole('tooltip'))
    .toHaveTextContent('r10 · 4 s ago')
  const next = structuredClone(view)
  next.last_completed_request!.age_ms = 2000
  mounted.client.setQueryData(viewKey, next)
  await expect
    .poll(() => document.activeElement?.getAttribute('data-request-id'))
    .toBe('r10')
  await expect
    .element(page.getByRole('tooltip'))
    .toHaveTextContent('r10 · 6 s ago')
})
