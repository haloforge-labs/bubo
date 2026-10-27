import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('Server: cold CLI starting phase', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('stopped-saved-estimate')
  view.startup_phase = 'starting_cli'
  view.initialized = false
  view.prepared = false
  // Go's label for a CLI that has not answered yet (ORACLE row 1).
  view.phase = 'checking'
  view.status_label = 'Checking this computer…'
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('main'))
    .toHaveTextContent('Checking this computer…')
  await document.fonts.ready
  await commands.captureFeature('server', 'cold-cli-starting')
})

for (const name of [
  'ready-idle',
  'ready-busy',
  'owner-memory-fit',
  'stopped-saved-estimate',
  'starting',
  'stalled',
  'failed-with-remedy',
  'external-server',
]) {
  test(`Server: ${name}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    mountCorePage('/', coreView(name))
    await expect
      .element(page.getByRole('heading', { name: 'Server', exact: true }))
      .toBeVisible()
    await document.fonts.ready
    expect(await commands.captureFeature('server', name)).toMatch(/\.png$/)
  })
}

test('Server: synthetic telemetry supplements absent recording fields', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-busy')
  view.rates = {
    ...view.rates!,
    decode_tokens_per_second: 43,
    prefill_tokens_per_second: 607,
  }
  view.gpu = {
    use_pct: 73,
    memory_used_bytes: 70 * 1024 ** 3,
    memory_total_bytes: 96 * 1024 ** 3,
    temperature_c: 61,
    chip_power_w: 83,
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await document.fonts.ready
  expect(
    await commands.captureFeature('server', 'synthetic-telemetry'),
  ).toMatch(/\.png$/)
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  await page
    .getByRole('figure', { name: 'Chip power (CPU, GPU, NPU)', exact: true })
    .element()
    .scrollIntoView()
  expect(
    await commands.captureFeature('settings', 'synthetic-gpu-sessions'),
  ).toMatch(/\.png$/)
})

test('Server: synthetic supplied endpoints', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  view.endpoints = {
    openai_base_url: 'https://example.test/openai-base',
    anthropic_base_url: 'https://example.test/anthropic-base',
  }
  mountCorePage('/', view)
  await expect
    .element(
      page
        .getByRole('region', { name: 'Addresses', exact: true })
        .getByRole('button', { name: 'Copy OpenAI address' }),
    )
    .toBeVisible()
  await document.fonts.ready
  expect(
    await commands.captureFeature('server', 'synthetic-endpoints'),
  ).toMatch(/\.png$/)
})
