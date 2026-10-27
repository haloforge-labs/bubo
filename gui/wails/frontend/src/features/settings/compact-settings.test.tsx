import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import SettingsPage from './index'

let cleanup = () => {}
afterEach(() => cleanup())
test('Advanced contains Mode, memory and hardware with plain help and a label/help filter', async () => {
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('combobox', { name: 'Mode', exact: true }))
    .toBeVisible()
  await expect
    .element(
      page.getByRole('heading', { name: 'Memory and caches', exact: true }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('heading', { name: 'Hardware', exact: true }))
    .toBeVisible()
  const filter = page.getByRole('textbox', { name: 'Find a setting' })
  await filter.fill('tunes the engine')
  await expect
    .element(page.getByRole('combobox', { name: 'Mode', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('NPU prefill', { exact: true }))
    .not.toBeInTheDocument()
  await filter.fill('')
  expect(document.querySelector('[data-setting="--twin-model"]')).toBeNull()
  expect(
    document.querySelector('[data-setting="--draft-head-model"]'),
  ).toBeNull()
  await page.getByRole('button', { name: 'About Mode', exact: true }).click()
  await expect
    .element(
      page.getByText(
        "How Bubo tunes the engine. Default is recommended; Vanilla turns Bubo's tuning off.",
        { exact: true },
      ),
    )
    .toBeVisible()
})

test('Advanced Hardware retains the reported GPU metrics and NPU status', async () => {
  const view = fixture('npu-prefill-on')
  view.gpu = { use_pct: 0, chip_power_w: 73 }
  cleanup = renderView(<SettingsPage />, view).cleanup
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(
      page.getByRole('region', { name: 'NPU prefill status', exact: true }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'GPU use', exact: true }))
    .toBeVisible()
})
