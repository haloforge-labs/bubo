import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

// The design review's rules, as Settings shows them.

test('a recovery link to a Settings setting opens Advanced, unfolds Debug and shows the row', async () => {
  const view = coreView('stopped-saved-estimate')
  const debug = view.switches.find(
    (row) =>
      row.audience === 'debug' &&
      row.gui_control &&
      row.name.startsWith('--') &&
      !row.name.includes('mtp'),
  )!
  // The memory history takes a hash; the helper's type names only the routes.
  mountCorePage(`/settings#setting:${debug.name}` as '/settings', view)
  await expect
    .element(page.getByRole('tab', { name: 'Advanced', selected: true }))
    .toBeVisible()
  const row = document.querySelector(`[data-setting="${debug.name}"]`)!
  await expect.poll(() => row.closest('details')?.open).toBe(true)
})

test('a setting the CLI redacts shows "Value hidden" and never the marker', async () => {
  const view = coreView('stopped-saved-estimate')
  const row = view.switches.find(
    (item) =>
      item.gui_control &&
      item.audience === 'debug' &&
      item.allowed.properties?.type?.text === 'string' &&
      !view.switch_overrides.find((state) => state.name === item.name)
        ?.write_block,
  )!
  const state = view.switch_overrides.find((item) => item.name === row.name)!
  const name = row.name
  state.effective = { kind: 'string', text: '[hidden]' }
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await page.getByText('Debug', { exact: true }).click()
  await expect
    .element(
      page
        .getByRole('group', { name: row.label ?? name, exact: true })
        .getByText('Value hidden'),
    )
    .toBeVisible()
  expect(document.body.textContent).not.toContain('[hidden]')
})

test('a power window under a second is told in milliseconds', async () => {
  const view = coreView('npu-prefill-on')
  view.npu_prefill_runtime = {
    ...view.npu_prefill_runtime!,
    chip_power: { ...view.npu_prefill_runtime!.chip_power, window_s: 0.25 },
  }
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  const hardware = page.getByRole('region', { name: 'NPU prefill status' })
  await hardware.getByText('Details', { exact: true }).click()
  await expect.element(hardware).toHaveTextContent('250 ms')
})

test('the unsaved-changes dialog names only the two clicks it offers', async () => {
  const view = coreView('stopped-saved-estimate')
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await page.getByRole('switch', { name: 'Check for updates at start' }).click()
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByText('Keep editing to save, or discard your changes.'))
    .toBeVisible()
})

test('Open setting for a Debug setting leaves the App tab, clears a hiding filter and shows the row', async () => {
  const view = coreView('switch-override-rejected')
  const debug = view.switches.find(
    (row) =>
      row.audience === 'debug' &&
      row.gui_control &&
      row.name.startsWith('--') &&
      !row.name.includes('mtp'),
  )!
  view.failure!.switch = debug.name
  mountCorePage('/settings', view)
  // The App tab is chosen first.
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  const open = page
    .getByRole('region', { name: 'Rejected override' })
    .getByRole('link', { name: 'Open setting' })
  await open.click()
  await expect
    .element(page.getByRole('tab', { name: 'Advanced', selected: true }))
    .toBeVisible()
  const row = () => document.querySelector(`[data-setting="${debug.name}"]`)
  await expect.poll(() => row()?.closest('details')?.open).toBe(true)
  // A filter that hides the row is cleared by the same link.
  // The reveal scrolled the row into view; the user scrolls back up to the filters.
  document.querySelector('[aria-label="Find a setting"]')!.scrollIntoView()
  await page.getByLabelText('Find a setting').fill('no such setting')
  await page.getByLabelText('Filter debug settings').fill('no such setting')
  await expect.poll(() => row()).toBeNull()
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  await open.click()
  await expect
    .element(page.getByRole('tab', { name: 'Advanced', selected: true }))
    .toBeVisible()
  await expect
    .element(page.getByLabelText('Filter debug settings'))
    .toHaveValue('')
  await expect.poll(() => row()?.closest('details')?.open).toBe(true)
})
