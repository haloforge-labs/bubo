import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import RequestsPage from './index'

let cleanup = () => {}
afterEach(() => cleanup())

// E22 fix 10 (UX15 N15-04).
//
// Failure list, written before the code:
//  10a. The Requests note always ends "a dash means no value", also when no
//       cell of the table holds a dash.
//       -> the clause shows only when a dash is in the table.
//  10b. With one absent value in a row the clause is there (it explains a dash
//       the reader can see).
//  10c. With no rows there is no table and no dash clause.
const clause = 'a dash means no value'

function complete(view: ReturnType<typeof fixture>) {
  view.requests = view.requests.map((row, index) => ({
    ...row,
    slot: row.slot ?? 0,
    ttft_ms: row.ttft_ms ?? 100,
    decode_tps: row.decode_tps ?? 50,
    prefill_tps: row.prefill_tps ?? 200,
    prompt_tokens: row.prompt_tokens ?? 10,
    completion_tokens: row.completion_tokens ?? 10,
    cache_kind: row.cache_kind || 'none',
    id: row.id || `r${index}`,
  }))
  return view
}

test('10a: a table with every value shows no dash clause', async () => {
  const view = complete(fixture('ready-busy'))
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .poll(() => document.querySelectorAll('tbody tr').length)
    .toBeGreaterThan(0)
  expect(document.querySelector('tbody')!.textContent).not.toContain('—')
  expect(document.body.textContent).not.toContain(clause)
  await expect
    .element(page.getByText(/Rates are for that one request\./))
    .toBeVisible()
})

test('10b: a dash in the table brings its explanation', async () => {
  const view = complete(fixture('ready-busy'))
  view.requests[0]!.ttft_ms = undefined
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .poll(() => document.querySelectorAll('tbody tr').length)
    .toBeGreaterThan(0)
  expect(document.querySelector('tbody')!.textContent).toContain('—')
  await expect.element(page.getByText(new RegExp(clause))).toBeVisible()
})

test('10c: no rows, no dash clause', async () => {
  const view = fixture('ready-idle')
  view.requests = []
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect.element(page.getByText('No requests yet.')).toBeVisible()
  expect(document.body.textContent).not.toContain(clause)
})
