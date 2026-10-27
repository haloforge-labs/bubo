import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

const count = (text: string, part: string) => text.split(part).length - 1

test('a model download is no red error: status, one whole-job line and Cancel only', async () => {
  const view = coreView('models-download-in-progress')
  view.progress!.overall = {
    bytes_done: Math.round(31.2 * 1024 ** 3),
    bytes_total: Math.round(108.6 * 1024 ** 3),
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByText('Downloading model', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('31.2 GiB of 108.6 GiB · 29 %', { exact: false }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Cancel download' }))
    .toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  for (const name of ['Start server', 'Restart server'])
    await expect
      .element(page.getByRole('button', { name, exact: true }))
      .not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain(
    'Another Bubo action is still running.',
  )
  // The current file is Details, not the headline.
  expect(document.body.textContent).toContain('Details')
})

test('a server started outside Bubo is plain text with no disabled controls', async () => {
  mountCorePage('/', coreView('external-server'))
  // The status label says it once; no second sentence repeats it.
  await expect
    .element(page.getByText('Running (started outside Bubo)', { exact: true }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain(
    'This server was started outside Bubo.',
  )
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  for (const name of ['Stop server', 'Restart server'])
    await expect
      .element(page.getByRole('button', { name, exact: true }))
      .not.toBeInTheDocument()
})

test('a stalled server makes Restart the primary action and says nothing about memory', async () => {
  mountCorePage('/', coreView('stalled'))
  const restart = page.getByRole('button', {
    name: 'Restart server',
    exact: true,
  })
  await expect.element(restart).toBeVisible()
  expect(restart.element().className).toContain('bg-brand')
  expect(
    page.getByRole('button', { name: 'Stop server', exact: true }).element()
      .className,
  ).not.toContain('bg-brand')
  expect(document.body.textContent).not.toContain(
    "Bubo can't check memory while its own server is using it.",
  )
})

test('a running server shows no memory-check sentence and no absence placeholders', async () => {
  const view = coreView('ready-idle')
  view.requests = []
  view.last_completed_request = undefined
  mountCorePage('/', view)
  await expect
    .element(page.getByText('Running', { exact: true }).first())
    .toBeVisible()
  const text = document.body.textContent!
  expect(text).not.toContain("Bubo can't check memory")
  expect(text).not.toContain('Server log path was not reported')
  expect(text).not.toContain('— tok/s')
})

test('two completed requests retain the last request figures without Server charts', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  view.requests = [
    {
      ...request,
      id: 'ux8-earlier',
      time: new Date(Date.parse(request.time) - 60_000).toISOString(),
      decode_tps: 44,
      prefill_tps: 250,
    },
    request,
  ]
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Recent performance' }))
    .not.toBeInTheDocument()
  const text = document.body.textContent!
  expect(text).not.toContain('average, all sessions')
  expect(text).not.toContain('wall-clock average')
  expect(document.querySelectorAll('.recharts-line-curve')).toHaveLength(0)
})

test('saved but unchecked setup is one plain line with no red box and no Restart', async () => {
  mountCorePage('/', coreView('saved-fit-unchecked'))
  await expect
    .element(
      page.getByText('Not checked: you can still start the server.', {
        exact: true,
      }),
    )
    .toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('Checking memory…')
  await expect
    .element(page.getByRole('button', { name: 'Restart', exact: false }))
    .not.toBeInTheDocument()
})

test('no fit at any size says it once and offers no Check again', async () => {
  mountCorePage('/', coreView('no-fit'))
  await expect
    .element(page.getByRole('region', { name: 'Memory fit' }))
    .toBeVisible()
  const text = document.body.textContent!
  expect(count(text, "Doesn't fit at any context")).toBe(1)
  await expect
    .element(page.getByRole('button', { name: 'Check again', exact: true }))
    .not.toBeInTheDocument()
})

test('a smaller context that fits is offered once, as the button', async () => {
  const view = coreView('memory-smaller-context')
  mountCorePage('/', view)
  const label = `Use ${view.fit.suggested_context!.toLocaleString()} tokens`
  await expect
    .element(page.getByRole('button', { name: label, exact: true }))
    .toBeVisible()
  expect(count(document.body.textContent!, label)).toBe(1)
})
