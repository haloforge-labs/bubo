import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import {
  coreView,
  mountCorePage,
  withEarlierRequest,
} from '@/components/core-page-test-support'
import { checkFailureSentence } from '@/lib/gui-errors'

afterEach(() => vi.restoreAllMocks())

test('a failed update check names the check, not a download', () => {
  expect(checkFailureSentence({ code: 'network', remedy: '' })).toBe(
    "Bubo can't check for updates: it couldn't reach the update server.",
  )
  expect(checkFailureSentence({ code: 'url_refused', remedy: '' })).toBe(
    "Bubo can't check for updates: the update address could not be verified.",
  )
})

for (const name of ['starting', 'failed-with-remedy'])
  test(`${name}: no memory line where no check can finish`, async () => {
    mountCorePage('/', coreView(name))
    await expect
      .element(page.getByRole('heading', { name: 'Server', exact: true }))
      .toBeVisible()
    expect(document.body.textContent).not.toContain('Checking memory…')
  })

test('saved but unchecked setup shows the registry defaults, not blank boxes', async () => {
  mountCorePage('/', coreView('saved-fit-unchecked'))
  await expect
    .element(page.getByRole('spinbutton', { name: 'Context per session' }))
    .toHaveValue('200,000')
  await expect
    .element(page.getByRole('spinbutton', { name: 'Sessions' }))
    .toHaveValue('1')
})

test('a rejected override is a failure and is red; Stop is never a solid red button', async () => {
  mountCorePage('/', coreView('switch-override-rejected'))
  const box = page.getByRole('region', { name: 'Rejected override' })
  await expect.element(box).toBeVisible()
  const panel = box.getByRole('alert').element()
  expect(getComputedStyle(panel).borderLeftWidth).toBe('4px')
  expect(getComputedStyle(panel).backgroundColor).not.toBe('rgba(0, 0, 0, 0)')
  await expect.element(box.getByText('Error', { exact: true })).toBeVisible()
  const running = coreView('ready-idle')
  mountCorePage('/', running)
  const stop = page.getByRole('button', { name: 'Stop server', exact: true })
  await expect.element(stop).toBeVisible()
  expect(stop.element().className).not.toContain('bg-destructive')
})

test('Server leaves rate and GPU history charts to Requests and Logs', async () => {
  mountCorePage('/', withEarlierRequest(coreView('ready-busy')))
  const charts = page.getByRole('region', { name: 'Recent performance' })
  await expect.element(charts).not.toBeInTheDocument()
})

test('Server has no Addresses section, no NPU line and no Anthropic row until they exist', async () => {
  mountCorePage('/', coreView('stopped-saved-estimate'))
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('region', { name: 'Addresses' }))
    .not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('NPU prefill: unavailable')
  expect(document.body.textContent).not.toContain('Addresses appear when')
})

test('a stopped Server does not repeat its memory verdict or offer Check again for a smaller context', async () => {
  mountCorePage('/', coreView('memory-smaller-context'))
  await expect
    .element(page.getByRole('button', { name: /^Use .* tokens$/ }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Check again', exact: true }))
    .not.toBeInTheDocument()
})
