import { expect, inject, test } from 'vitest'
import { commands, page, userEvent } from 'vitest/browser'
import { recordedGPU, recordedTransfer } from './recorded-observations'
import {
  coreView,
  mountCorePage,
  withEarlierRequest,
} from './core-page-test-support'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

for (const scenario of ['stopped-saved-estimate', 'memory-gpu-headroom']) {
  test(`F5a first run complete ${scenario}`, async () => {
    mountCorePage('/onboarding', coreView(scenario))
    await expect
      .element(page.getByRole('button', { name: 'Start server', exact: true }))
      .toBeVisible()
    await shot('onboarding', `f5a-complete-${scenario}`)
  })
}

for (const scenario of [
  'owner-memory-fit',
  'switch-override-rejected',
  'ready-idle',
  'ready-busy',
  'starting',
]) {
  test(`F5a server journey ${scenario}`, async () => {
    mountCorePage('/', coreView(scenario))
    await expect
      .element(page.getByRole('heading', { name: 'Server', exact: true }))
      .toBeVisible()
    await shot('server', `f5a-${scenario}`)
  })
}

test('F5a known catalogue file ordinal during a download', async () => {
  const view = coreView('preparing')
  // Explicit View fixture for the supervisor's tested plan/catalogue join.
  view.preparation_file = { index: 4, total: view.preparation[0]!.files.length }
  // Go's whole-job number; without it the line is the only statement.
  view.progress!.overall = recordedTransfer().overall
  mountCorePage('/onboarding', view)
  // The current file is a detail of the one whole-job line.
  await page.getByText('Details', { exact: true }).click()
  await expect.element(page.getByText(/^File 4 of 8: /)).toBeVisible()
  await shot('onboarding', 'f5a-known-file-ordinal')
})

test('F5a GPU history ends at the supplied current observation and tooltips fit the theme', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = withEarlierRequest(coreView('ready-busy'))
  const observed = recordedGPU()
  view.gpu = observed.current
  const { client } = mountCorePage('/', view)
  const samples = observed.samples
  expect(samples.at(-1)?.gpu.chip_power_w).toBe(view.gpu!.chip_power_w)
  client.setQueryData(['gpu-history', view.instance!.instance_id], samples)
  await page.getByRole('link', { name: 'Requests', exact: true }).click()
  const requests = page.getByRole('region', { name: 'Request performance' })
  await expect
    .poll(() => requests.element().querySelectorAll('.recharts-wrapper').length)
    .toBe(3)
  expect(requests.element().textContent).not.toContain('Last 300 s')
  await shot('requests', 'f5a-request-rate-charts')
  await page.getByRole('link', { name: 'Logs', exact: true }).click()
  const charts = page.getByRole('region', { name: 'GPU history' })
  await expect.element(charts).toHaveTextContent('Last 300 s')
  expect(charts.element().textContent?.match(/Last 300 s/g)).toHaveLength(1)
  await expect
    .poll(() => charts.element().querySelectorAll('.recharts-wrapper').length)
    .toBe(4)
  const power = charts.getByRole('figure', {
    name: 'Chip power (CPU, GPU, NPU) history',
    exact: true,
  })
  power.element().scrollIntoView({ block: 'center' })
  await userEvent.hover(
    power.element().querySelector('.recharts-reference-dot circle')!,
  )
  await expect
    .element(page.getByLabelText('Sample details'))
    .toHaveTextContent('48.4 W')
  await shot('logs', 'f5a-themed-tooltip')
})
