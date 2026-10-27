import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('Server always offers import and the CLI largest-context row', async () => {
  const view = coreView('stopped-saved-estimate')
  mountCorePage('/', view)
  await expect
    .element(
      page.getByRole('link', {
        name: 'Not listed? Add a model you already have.',
      }),
    )
    .toHaveAttribute('href', '/models#import')
  await expect
    .element(page.getByText('Largest that fits now:', { exact: false }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Use largest context' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('link', { name: 'Connect', exact: true }).last())
    .toHaveAttribute('href', '/connect')
})

test('last request reports full cache hits and TTFT without an invented prefill rate', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  request.prompt_tokens = 17
  request.cached_tokens = 17
  request.prefill_tps = undefined
  request.ttft_ms = 8
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Prefill: all 17 tokens from cache', { exact: true }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'Time to first token' }))
    .toHaveTextContent('8 ms')
})

test('busy port saves the CLI port scalar inline', async () => {
  const view = coreView('stopped-saved-estimate')
  view.failure = {
    code: 'port_occupied',
    remedy: 'Fixture busy address',
    source: 'fixture',
    time: '',
    forced: false,
  }
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await page.getByLabelText('New server port').fill('9099')
  await page.getByRole('button', { name: 'Save port' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: '--port', value: '9099' }],
  })
})

test('Tight opt-in writes only consent and waits for supervisor admission', async () => {
  const view = coreView('memory-gpu-headroom')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const { replaceView } = mountCorePage('/', view)
  const start = page.getByRole('button', { name: 'Start server', exact: true })
  const checkbox = page.getByRole('checkbox', { name: /^Override:/ })
  await expect.element(start).toBeDisabled()
  await checkbox.click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { allow_tight_setup: true },
    clear: [],
  })
  await expect.element(start).toBeDisabled()
  replaceView({
    ...view,
    saved: { ...view.saved, allow_tight_setup: true },
    actions: {
      ...view.actions,
      start: { allowed: true, reason_code: '', reason: '' },
    },
  })
  await expect.element(checkbox).toBeChecked()
  await expect.element(start).toBeEnabled()
  await page.getByLabelText('Context per session').fill('65536')
  await expect.element(checkbox).not.toBeChecked()
  await expect.element(checkbox).toBeDisabled()
  // Start saves and rechecks this draft; the saved Tight setup needs fresh consent.
  await expect.element(start).toBeEnabled()
})

test('Mode belongs in Settings and saves through its CLI-driven preference route', async () => {
  const view = coreView('memory-gpu-headroom')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await page
    .getByRole('combobox', { name: 'Mode', exact: true })
    .selectOptions('1')
  await expect
    .element(page.getByRole('button', { name: 'Save changes', exact: true }))
    .toBeEnabled()
  await page.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [
      {
        name: '--profile',
        value: view.profiles.filter((profile) => profile.gui)[1]!.name,
      },
    ],
  })
})

test('partial cache hits show new and cached counts beside the CLI rate and real served name', async () => {
  const view = coreView('ready-busy')
  const request = view.requests.findLast((row) => row.outcome === 'completed')!
  request.prompt_tokens = 1373
  request.cached_tokens = 1345
  const served = view.models!.catalogue[0]!.id
  view.instance!.served_model = served
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('figure', { name: 'Prefill tok/s' }))
    .toHaveTextContent('28 new + 1,345 cached')
  await expect
    .element(page.getByText('Clients see this model as', { exact: false }))
    .toHaveTextContent(`Clients see this model as ${served}. (bubo also works)`)
})
