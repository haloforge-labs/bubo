import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { recordedGPU } from '@/components/recorded-observations'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

test('V12 Logs compact GPU histories above log lines', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-busy')
  const observed = recordedGPU()
  view.gpu = observed.current
  const { client } = mountCorePage('/models', view)
  await page.getByRole('link', { name: 'Logs', exact: true }).click()
  client.setQueryData(
    ['gpu-history', view.instance!.instance_id],
    observed.samples,
  )
  await expect
    .element(page.getByRole('region', { name: 'GPU history' }))
    .toBeVisible()
  await expect
    .poll(() => document.querySelectorAll('.recharts-wrapper').length)
    .toBe(4)
  await document.fonts.ready
  expect(await commands.captureFeature('logs', 'v12-gpu-history')).toMatch(
    /\.png$/,
  )
})
