import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedGPU, recordedTransfer } from './recorded-observations'
import {
  coreView,
  mountCorePage,
  withEarlierRequest,
} from './core-page-test-support'
import { fixture } from './test-view'
import { capturePage } from './page-capture'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
}

test('F1 onboarding after runtime preparation', async () => {
  mountCorePage('/', coreView('runtime-ready-model-missing'))
  await expect
    .element(page.getByRole('button', { name: 'Download model…' }))
    .toBeVisible()
  await shot('onboarding', 'runtime-ready-download-next')
})

test('F1 runtime failure details disclosure', async () => {
  const view = coreView('not-prepared')
  view.failure = {
    code: 'network',
    remedy: 'Check the connection and retry.',
    source: 'cli',
    time: '',
    forced: false,
  }
  // Ordinary setup has no error; exercise Details only for a real failure.
  mountCorePage('/', view)
  await page.getByText('Details', { exact: true }).first().click()
  await shot('onboarding', 'runtime-error-details')
})

test('F1 observed transfer speed and time remaining', async () => {
  const view = coreView('models-download-in-progress')
  const job = view.models!.download!
  const observed = recordedTransfer()
  Object.assign(job, observed.file)
  job.overall = observed.overall
  const { client } = mountCorePage('/models', view)
  client.setQueryData(
    ['transfer-samples', `model-${job.id}-${job.kind}`],
    observed.samples,
  )
  await expect
    .element(page.getByRole('region', { name: 'Model download progress' }))
    .toHaveTextContent('140.0 MiB/s')
  await shot('models', 'observed-download-speed')
})

test('F1 running telemetry after five observed samples', async () => {
  const view = withEarlierRequest(coreView('ready-busy'))
  view.gpu = recordedGPU().current
  const { client } = mountCorePage('/', view)
  const at = Date.now()
  client.setQueryData(
    ['gpu-history', view.instance!.instance_id],
    Array.from({ length: 5 }, (_, index) => ({
      at: at - (4 - index) * 2000,
      gpu: view.gpu,
    })),
  )
  expect(document.querySelector('.recharts-wrapper')).toBeNull()
  await expect
    .element(page.getByRole('region', { name: 'Reported request rates' }))
    .toHaveTextContent('Last request')
  await expect.element(page.getByText('GPU history')).not.toBeInTheDocument()
  await shot('server', 'five-samples-top')
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  const shell = document.getElementById('shell-capture')!
  page
    .getByRole('figure', { name: 'GPU use', exact: true })
    .element()
    .scrollIntoView({ block: 'start' })
  shell.scrollTop = Math.max(0, shell.scrollTop)
  await shot('settings', 'five-samples-current-gpu')
  await page.getByRole('link', { name: 'Logs', exact: true }).click()
  await expect
    .element(page.getByRole('region', { name: 'GPU history' }))
    .not.toBeInTheDocument()
})

for (const scenario of ['starting', 'stopped-saved-estimate', 'lan-on']) {
  test(`F1 Connect scenario ${scenario}`, () =>
    capturePage({
      feature: 'connect',
      scenario: `scenario-${scenario}`,
      view: fixture(scenario),
    }))
}
