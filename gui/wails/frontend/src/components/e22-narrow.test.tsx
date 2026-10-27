import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

// vitest.config.ts opens every browser test at this size.
afterEach(async () => {
  await page.viewport(1100, 760)
})

// E22 fix 9 (UX15 T15-09, N15-05, K: N15-02). The app's default window is
// 900 x 640 and its minimum is 560 x 360.
//
// Failure list, written before the code:
//  9a. With a running server and an update on offer, the update chip sits on a
//      row of its own, under the status and the two address buttons, at every
//      window size (its text "Update available · 0.1.2" also wrapped inside it).
//      -> the chip is on the status row (same row as the status text), on one
//      line, at the minimum, the default and the largest window.
//  9b. Settings tabs: at the default window "Advanced" sits alone on a second
//      row, stretched across it.
//      -> no tab row holds a single tab when the tabs wrap, at every size.
const sizes = [
  [560, 360],
  [700, 500],
  [900, 640],
  [1100, 760],
  [1400, 900],
] as const

for (const [width, height] of sizes) {
  test(`9a: the update chip stays on the status row at ${width} x ${height}`, async () => {
    await page.viewport(width, height)
    const view = coreView('ready-idle')
    view.update = {
      ...view.update,
      state: 'available',
      available_version: '0.1.2',
    }
    mountCorePage('/requests' as '/', view)
    await expect
      .element(page.getByRole('link', { name: /Update/ }))
      .toBeVisible()
    const header = document.querySelector('header')!
    const status = header
      .querySelector('[role="status"]')!
      .getBoundingClientRect()
    const chip = page.getByRole('link', { name: /Update/ }).element()
    const box = chip.getBoundingClientRect()
    const middle = (rect: DOMRect) => rect.top + rect.height / 2
    expect(
      Math.abs(middle(box) - middle(status)),
      'chip beside the status',
    ).toBeLessThan(Math.max(status.height, box.height))
    // One line inside the chip.
    expect(box.height).toBeLessThan(
      parseFloat(getComputedStyle(chip).lineHeight || '20') * 2,
    )
    // The addresses are still there, in the header.
    expect(
      header.querySelector('[aria-label="Server addresses"]'),
    ).not.toBeNull()
  })

  test(`9b: no Settings tab is alone on a row at ${width} x ${height}`, async () => {
    await page.viewport(width, height)
    mountCorePage('/settings', coreView('ready-idle'))
    await expect
      .element(page.getByRole('tab', { name: 'Advanced' }))
      .toBeVisible()
    const rows = new Map<number, number>()
    for (const tab of document.querySelectorAll('[role="tab"]')) {
      const top = Math.round(tab.getBoundingClientRect().top)
      rows.set(top, (rows.get(top) ?? 0) + 1)
    }
    if (rows.size > 1)
      expect(
        [...rows.values()].every((count) => count > 1),
        `rows ${[...rows.values()]}`,
      ).toBe(true)
  })
}

test('9a: with no update on offer the addresses stay at the right edge of the header', async () => {
  await page.viewport(900, 640)
  mountCorePage('/requests' as '/', coreView('ready-idle'))
  await expect.element(page.getByText('Copy OpenAI address')).toBeVisible()
  const header = document.querySelector('header')!
  const addresses = header.querySelector('[aria-label="Server addresses"]')!
  expect(
    header.getBoundingClientRect().right -
      addresses.getBoundingClientRect().right,
  ).toBeLessThan(40)
})
