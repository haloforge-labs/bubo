import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { fixture, renderView } from '@/components/test-view'
import { RegistrySection } from './registry-section'

let cleanup = () => {}
afterEach(() => cleanup())

// The Advanced tab and the Server page share one setting row in one auto-fit
// grid. A row's buttons and text must stay inside the row's own wrapper at wide
// and narrow widths, so long setting names never run into the next column, and
// the page never scrolls sideways.
test.each([
  [1100, 760],
  [900, 640],
  [560, 640],
  [360, 800],
])(
  'Advanced rows keep their content inside their cell at %i px',
  async (width, height) => {
    await page.viewport(width, height)
    const view = fixture()
    // Long names, as the CLI publishes some ("Frequency Penalty (token frequency penalty)").
    view.switches = view.switches.map((row) => ({
      ...row,
      label: `${row.label || row.name} (a long explanatory name for this setting)`,
    }))
    cleanup = renderView(
      <RegistrySection view={view} audience="advanced" includeUserRemainder />,
      view,
    ).cleanup
    await expect
      .poll(() => document.querySelectorAll('[data-setting]').length)
      .toBeGreaterThan(20)
    // Rows without a value show a Default button (named Change ...); the case
    // must be on screen.
    expect(
      document.querySelectorAll('button[aria-label^="Change "]').length,
    ).toBeGreaterThan(0)
    const outside: string[] = []
    for (const row of document.querySelectorAll<HTMLElement>(
      '[data-setting]',
    )) {
      // The row is one wrapper (role group) in the group's grid: its control,
      // hint and ⓘ must stay inside it.
      const cell = row.getBoundingClientRect()
      for (const item of row.querySelectorAll<HTMLElement>(
        'button, select, input, p, summary',
      )) {
        const box = item.getBoundingClientRect()
        if (
          box.width &&
          (box.right > cell.right + 1 ||
            box.right > document.documentElement.clientWidth + 1)
        )
          outside.push(
            `${row.dataset.setting}: ${item.tagName} "${item.textContent?.slice(0, 40)}" right ${Math.round(box.right)} > cell ${Math.round(cell.right)}`,
          )
      }
    }
    expect(outside).toEqual([])
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(width + 1)
  },
)

// The Server page uses the same rows. Open its Inference list and every
// "More ..." disclosure so all rows are measured.
test.each([
  [1100, 760],
  [360, 800],
])(
  'Server rows keep their content inside their cell at %i px',
  async (width, height) => {
    await page.viewport(width, height)
    const view = coreView('stopped-saved-estimate')
    view.switches = view.switches.map((row) => ({
      ...row,
      label: `${row.label || row.name} (a long explanatory name for this setting)`,
    }))
    mountCorePage('/', view)
    await page.getByRole('button', { name: 'Inference', exact: true }).click()
    for (const details of document.querySelectorAll('details'))
      details.open = true
    await expect
      .poll(() => document.querySelectorAll('[data-setting]').length)
      .toBeGreaterThan(5)
    const outside: string[] = []
    for (const row of document.querySelectorAll<HTMLElement>(
      '[data-setting]',
    )) {
      // The row is one wrapper (role group) in the group's grid: its control,
      // hint and ⓘ must stay inside it.
      const cell = row.getBoundingClientRect()
      for (const item of row.querySelectorAll<HTMLElement>(
        'button, select, input, p, summary',
      )) {
        const box = item.getBoundingClientRect()
        if (
          box.width &&
          (box.right > cell.right + 1 ||
            box.right > document.documentElement.clientWidth + 1)
        )
          outside.push(
            `${row.dataset.setting}: ${item.tagName} "${item.textContent?.slice(0, 40)}" right ${Math.round(box.right)} > cell ${Math.round(cell.right)}`,
          )
      }
    }
    expect(outside).toEqual([])
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(width + 1)
  },
)
