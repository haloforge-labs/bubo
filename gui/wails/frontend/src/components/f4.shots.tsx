import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedGPU, recordedTransfer } from './recorded-observations'
import {
  coreView,
  mountCorePage,
  withEarlierRequest,
} from './core-page-test-support'

async function shot(feature: string, name: string) {
  document.documentElement.dataset.theme = inject('theme')
  await document.fonts.ready
  await commands.captureFeature(feature, name)
}

test('F4 busy request rates and compact changing GPU histories', async () => {
  const view = withEarlierRequest(coreView('ready-busy'))
  const observed = recordedGPU()
  view.gpu = observed.current
  const { client } = mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Reported request rates' }))
    .toHaveTextContent('Last request')
  await shot('server', 'f4-busy-request-rates')
  client.setQueryData(
    ['gpu-history', view.instance!.instance_id],
    observed.samples,
  )
  await page.getByRole('link', { name: 'Requests', exact: true }).click()
  const charts = page.getByRole('region', { name: 'Request performance' })
  for (const metric of ['Decode', 'Prefill'])
    await expect
      .element(
        charts.getByRole('figure', {
          name: metric + ' per completed request',
          exact: true,
        }),
      )
      .toHaveTextContent('(tok/s)')
  await expect
    .poll(() => charts.element().querySelectorAll('.recharts-wrapper').length)
    .toBe(3)
  await shot('requests', 'f4-five-samples-rate-charts')
  await page.getByRole('link', { name: 'Logs', exact: true }).click()
  const gpuHistory = page.getByRole('region', { name: 'GPU history' })
  await expect.element(gpuHistory).toBeVisible()
  await expect
    .poll(
      () => gpuHistory.element().querySelectorAll('.recharts-wrapper').length,
    )
    .toBe(4)
  await shot('logs', 'f4-five-samples-gpu-charts')
  await page.getByRole('link', { name: 'Server', exact: true }).click()
  const sessions = page.getByRole('region', { name: 'Sessions' })
  await expect
    .element(sessions.getByRole('cell', { name: '1', exact: true }))
    .toBeVisible()
  sessions.element().scrollIntoView({ block: 'start' })
  await shot('server', 'f4-session-one-based')
})

test('F4 onboarding shows the transferring component in progress', async () => {
  const view = coreView('preparing')
  // The supervisor adds up the plan's files; this fixture states that total.
  const observed = recordedTransfer()
  view.progress!.overall = observed.overall
  const { client } = mountCorePage('/onboarding', view)
  client.setQueryData(
    ['transfer-samples', `prepare-${view.generation}`],
    observed.samples,
  )
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('(default model) · In progress')
  await expect
    .element(
      page.getByText(
        /103\.7 GiB of 108\.6 GiB · 95 % · 140\.0 MiB\/s · about 36 s left/,
      ),
    )
    .toBeVisible()
  await shot('onboarding', 'f4-model-in-progress')
})

test('F4 running model is in use', async () => {
  const view = coreView()
  const model = view.models!.prepared_models[0]!
  view.active_model = {
    catalogue_id: model.id,
    manifest_path: model.manifest_path,
  }
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('region', { name: 'Installed models' }))
    .toHaveTextContent('In use')
  await shot('models', 'f4-running-model-in-use')
  // The page decides on typed fields: no "Not loadable" list from loader prose.
  expect(document.body.textContent).not.toContain(
    'Not loadable in this version',
  )
})

test.each(['--host', '--port'])(
  'F4 %s preserves the CLI control and current preference scalar',
  async (name) => {
    const view = coreView()
    const row = view.switches.find((row) => row.name === name)!
    view.switches = [row]
    if (name === '--port') {
      view.saved.port = 19001
      view.switch_overrides.find((entry) => entry.name === name)!.saved =
        String(view.saved.port)
    }
    mountCorePage('/settings', view)
    await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
    await expect
      .element(page.getByRole('textbox', { name: row.label! }))
      .toBeVisible()
    await shot('settings', `f4-current-${name.slice(2)}`)
  },
)
