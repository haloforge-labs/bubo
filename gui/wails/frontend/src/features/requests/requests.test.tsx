import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import RequestsPage from './index'

let cleanup = () => {}
afterEach(() => cleanup())
test('request metrics use the recorded rows and sorting preserves one line per request', async () => {
  const view = fixture('ready-busy')
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .element(page.getByRole('button', { name: /TTFT/ }))
    .toBeInTheDocument()
  await expect
    .element(
      page.getByRole('columnheader', { name: /Prefill/ }).getByRole('button'),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: /^Cache/ }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain(
    'Scroll sideways for all columns',
  )
  await expect
    .poll(() => document.querySelectorAll('tbody tr').length)
    .toBe(view.requests.length)
  for (const row of view.requests)
    expect(document.querySelector('tbody')?.textContent).toContain(
      row.finish_reason || row.outcome,
    )
  await page.getByRole('button', { name: /Prompt/ }).click()
  await expect
    .element(page.getByRole('columnheader', { name: /Prompt/ }))
    .toHaveAttribute('aria-sort', 'descending')
  await page.getByRole('button', { name: /Prompt/ }).click()
  await expect
    .element(page.getByRole('columnheader', { name: /Prompt/ }))
    .toHaveAttribute('aria-sort', 'ascending')
  const prompts = Array.from(document.querySelectorAll('tbody tr'), (row) =>
    Number(row.querySelectorAll('td')[2]?.textContent),
  )
  expect(prompts).toEqual(
    view.requests
      .map((row) => row.prompt_tokens)
      .sort((a, b) => (a ?? 0) - (b ?? 0)),
  )
})
test('absent values show a dash while reported zeroes remain zero', async () => {
  const view = fixture('ready-busy')
  const template = view.requests[0]!
  view.requests = [
    {
      ...template,
      prompt_tokens: 0,
      completion_tokens: 0,
      ttft_ms: undefined,
      decode_tps: undefined,
    },
  ]
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect.poll(() => document.querySelectorAll('tbody tr').length).toBe(1)
  const values = Array.from(
    document.querySelectorAll('tbody td'),
    (cell) => cell.textContent,
  )
  expect(values[2]).toBe('0')
  expect(values[3]).toBe('0')
  expect(values[4]).toBe('—')
  expect(values[5]).toBe('—')
})
test('empty request history stays empty', async () => {
  const view = fixture('ready-idle')
  view.requests = []
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect.element(page.getByText('No requests yet.')).toBeVisible()
})

test('request rates align at one decimal and TTFT at whole milliseconds', async () => {
  const view = fixture('ready-busy')
  view.requests = [
    {
      ...view.requests[0]!,
      ttft_ms: 12.76,
      decode_tps: 43,
      prefill_tps: 53.437,
    },
  ]
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect
    .element(page.getByRole('button', { name: /^Decode \(tok\/s\)/ }))
    .toBeVisible()
  const cells = document.querySelectorAll('tbody td')
  expect(cells[4]!.textContent).toBe('13')
  expect(cells[5]!.textContent).toBe('43.0')
  expect(cells[6]!.textContent).toBe('53.4')
  for (const index of [4, 5, 6]) {
    expect(getComputedStyle(cells[index]!).textAlign).toBe('right')
    expect(getComputedStyle(cells[index]!).fontFamily).toContain('Go Mono')
  }
  expect(
    page
      .getByRole('region', { name: 'Requests table' })
      .element()
      .classList.contains('border'),
  ).toBe(false)
})

test('a request that did not complete shows its outcome and cause, never the finish reason of its text', async () => {
  const view = fixture('ready-busy')
  const template = view.requests[0]!
  view.requests = [
    {
      ...template,
      id: 'req-failed',
      outcome: 'failed',
      finish_reason: 'stop',
      status: 500,
      error_code: 'device_lost',
    },
    {
      ...template,
      id: 'req-refused',
      outcome: 'refused',
      finish_reason: 'length',
      status: 503,
      error_code: 'insufficient_memory',
      refusal: {
        code: 'insufficient_memory',
        needed_bytes: 8589934592,
        free_bytes: 2147483648,
      },
    },
    {
      ...template,
      id: 'req-permanent',
      outcome: 'refused',
      status: 507,
      error_code: 'insufficient_memory_permanent',
      refusal: {
        code: 'insufficient_memory_permanent',
        needed_bytes: 137438953472,
        free_bytes: 2147483648,
      },
    },
    {
      ...template,
      id: 'req-pressure',
      outcome: 'refused',
      finish_reason: 'stop',
      status: 503,
      error_code: 'memory_pressure',
      refusal: {
        code: 'memory_pressure',
        needed_bytes: undefined,
        free_bytes: undefined,
      },
    },
    {
      ...template,
      id: 'req-model',
      outcome: 'failed',
      finish_reason: 'stop',
      status: 404,
      error_code: 'model_not_found',
    },
    {
      ...template,
      id: 'req-context',
      outcome: 'failed',
      status: 400,
      error_code: 'context_length_exceeded',
    },
    {
      ...template,
      id: 'req-busy',
      outcome: 'failed',
      status: 429,
      error_code: 'queue_full',
    },
    {
      ...template,
      id: 'req-key',
      outcome: 'failed',
      status: 401,
      error_code: 'unauthorized',
    },
    {
      ...template,
      id: 'req-bad',
      outcome: 'failed',
      status: 422,
      error_code: 'invalid_tool',
    },
    {
      ...template,
      id: 'req-5xx',
      outcome: 'failed',
      status: 502,
      error_code: 'backend_error',
    },
    {
      ...template,
      id: 'req-gone',
      outcome: 'disconnected',
      finish_reason: 'cancelled',
      status: undefined,
      error_code: undefined,
    },
    {
      ...template,
      id: 'req-done',
      outcome: 'completed',
      finish_reason: 'stop',
      status: 200,
      error_code: undefined,
    },
  ]
  cleanup = renderView(<RequestsPage />, view).cleanup
  await expect.poll(() => document.querySelectorAll('tbody tr').length).toBe(12)
  const rows = Object.fromEntries(
    Array.from(document.querySelectorAll('tbody tr'), (row) => [
      row.lastElementChild?.textContent,
      row.querySelectorAll('td')[8]?.textContent ?? '',
    ]),
  )
  // Exact text: no finish reason ("stop", "length") over a row that did not complete.
  expect(rows['req-failed']).toBe('Failed: the GPU stopped responding.')
  expect(rows['req-refused']).toBe(
    'Refused: needs 8.0 GiB of memory, 2.0 GiB free. Try again when more memory is free.',
  )
  expect(rows['req-permanent']).toBe(
    'Refused: needs 128.0 GiB of memory, more than this computer can ever provide.',
  )
  expect(rows['req-pressure']).toBe('Stopped: Windows ran low on memory.')
  expect(rows['req-model']).toBe(
    "Failed: the client asked for a model name this server doesn't serve.",
  )
  expect(rows['req-context']).toBe(
    'Failed: the prompt is longer than the context.',
  )
  expect(rows['req-busy']).toBe('Failed: the server was busy.')
  expect(rows['req-key']).toBe('Failed: the API key was missing or wrong.')
  expect(rows['req-bad']).toBe(
    'Failed: the request was not valid (invalid_tool).',
  )
  expect(rows['req-5xx']).toBe(
    'Failed: the server could not answer (backend_error).',
  )
  expect(rows['req-gone']).toBe('Stopped: the client disconnected.')
  expect(rows['req-done']).toBe('stop')
  // The raw status and code stay available in the cell's title.
  const refusedCell = Array.from(document.querySelectorAll('tbody tr'))
    .find((row) => row.lastElementChild?.textContent === 'req-refused')!
    .querySelectorAll('td')[8]!
    .querySelector('span')!
  expect(refusedCell.getAttribute('title')).toBe(
    'HTTP 503 · insufficient_memory',
  )
})
