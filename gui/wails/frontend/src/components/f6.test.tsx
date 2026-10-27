import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from './core-page-test-support'
import { serverAddresses } from '@/lib/format'

afterEach(() => vi.restoreAllMocks())

// The Ready idle recording has a saved estimate that fits, so no context
// button would show for any reason. ready-busy is the recorded running server
// whose saved estimate does not fit, so the missing button proves the rule.
test('recorded running server allows Restart and offers no context change', async () => {
  const view = coreView('ready-busy')
  const restart = vi.spyOn(actions, 'Restart').mockResolvedValue({})
  expect(view.saved_estimate.memory_estimate?.fits).toBe(false)
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: /^Use .* tokens$/ }))
    .not.toBeInTheDocument()
  await page
    .getByRole('button', { name: 'Restart server', exact: true })
    .click()
  expect(restart).toHaveBeenCalledExactlyOnceWith()
})

test('a server that is not serving hides retained address copy buttons', async () => {
  const view = coreView()
  view.serving = false
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('heading', { name: 'Models', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: /Copy .* address/ }))
    .not.toBeInTheDocument()
})

test('header and Server use the same serving connection URL', async () => {
  const view = coreView()
  view.endpoints = undefined
  const address = serverAddresses(view).openai_base_url!
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const { replaceView } = mountCorePage('/models', view)
  await page
    .getByRole('button', { name: 'Copy OpenAI address', exact: true })
    .click()
  expect(copy).toHaveBeenLastCalledWith(address)
  await page.getByRole('link', { name: 'Server', exact: true }).click()
  replaceView(view)
  await expect
    .element(page.getByRole('region', { name: 'Addresses', exact: true }))
    .toHaveTextContent(address)
})
