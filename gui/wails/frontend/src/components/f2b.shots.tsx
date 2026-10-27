import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedGPU } from './recorded-observations'
import { coreView, mountCorePage } from './core-page-test-support'
import { fixture } from './test-view'
import { capturePage } from './page-capture'
import { scenarios } from '../../tests/scenarios'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

test('F2b saved fit unchecked still permits Start', async () => {
  mountCorePage('/', coreView('saved-fit-unchecked'))
  await expect
    .element(
      page.getByText('Not checked: you can still start the server.', {
        exact: true,
      }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  await shot('server', 'saved-fit-unchecked')
})

// The two page-facts fixtures came from a fake CLI. K6-12 has not established
// that their model identity can occur with the shipped CLI.
for (const scenario of scenarios.filter(
  ({ name }) =>
    name !== 'cli-page-facts-loopback' &&
    name !== 'cli-page-facts-authenticated',
)) {
  test(`F2b all scenarios: ${scenario.name}`, async () => {
    mountCorePage('/', structuredClone(scenario.view))
    await expect
      .element(page.getByRole('heading', { level: 1 }).first())
      .toBeVisible()
    await shot(
      scenario.view.prepared ? 'server' : 'onboarding',
      `scenario-${scenario.name}`,
    )
  })
}

test('F2b GPU sample age and measured window', async () => {
  const view = coreView('ready-busy')
  view.gpu = recordedGPU().current
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  // The sample's age and window are a detail, under a closed disclosure.
  const sample = page.getByText('GPU sample:', { exact: false })
  await expect.element(sample).toBeInTheDocument()
  sample.element().closest('details')!.open = true
  await expect.element(sample).toBeVisible()
  const shell = document.getElementById('shell-capture')!
  shell.style.height = '760px'
  shell.style.overflow = 'auto'
  page
    .getByRole('region', { name: 'GPU', exact: true })
    .element()
    .scrollIntoView({ block: 'start' })
  await shot('settings', 'gpu-sample-window')
})

test('F2b external log path', () => {
  const view = fixture('external-server')
  view.log_path =
    'C:\\bubo\\evidence\\capture-04\\run-sample-20260101-1200\\P1\\server.err'
  return capturePage({
    feature: 'logs',
    scenario: 'external-log-path',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('button', { name: 'Copy log path' }))
        .toBeVisible()
    },
  })
})

test('F2b sampling home and explicit seed meaning', () => {
  const view = fixture()
  view.saved.inference_expanded = true
  return capturePage({
    feature: 'server',
    scenario: 'inference-random-seed',
    view,
    prepare: async () => {
      const seed = page.getByRole('spinbutton', { name: 'Seed', exact: true })
      await expect.element(seed).toHaveValue(-1)
      await expect.element(seed).toHaveAttribute('placeholder', 'Random')
    },
  })
})

test('F2b a 32-bit float shows as the value it stands for', () => {
  const view = fixture()
  view.saved.inference_expanded = true
  return capturePage({
    feature: 'server',
    scenario: 'inference-precision-unreported',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('spinbutton', { name: 'Top-p', exact: true }))
        .toHaveValue(0.95)
    },
  })
})
