import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import {
  mountCorePage,
  packagesView,
} from '@/components/core-page-test-support'

// vitest.config.ts opens every browser test at this size.
const configuredViewport = [1100, 760] as const
afterEach(async () => {
  vi.restoreAllMocks()
  await page.viewport(...configuredViewport)
})

// E21w W6. At the walk's window size the Installed table was 839 px wide in a
// 495 px box (overflow-x auto): "Select for Server" was clipped off and Size was
// cut. The long monospace path and the three borderless buttons under it set the
// width.
//
// Ways this could fail, written before the code:
//
//	V1 the table is wider than its box at the app's default window (900 x 640).
//	V2 the same at its minimum window (560 x 360).
//	V3 a row's action ("Select for Server", "Open folder", "Copy folder path",
//	   "Check or repair files") or its Size is cut at the box edge.
//	V4 a long path does not wrap: it stays on one line and widens the table.
//	V5 a fix that fits one size only (a fixed pixel width): checked at a third
//	   size in between, and at the largest the app opens at.
//	V6 the fix hides the actions or the path to make room (every row keeps its
//	   Select for Server, Open folder, Copy folder path and Size text).
//	V7 the longest recorded path hides the Status or Memory fit text.

const LONG =
  'C:\\Users\\fixture-with-a-long-name\\AppData\\Local\\Bubo\\models\\huihui-qwen38-flash-next-ud-q4-k-xl-v1\\'

function installedView() {
  const { view, rows } = packagesView()
  // The recorded package id in a long folder: a stranger's profile folder is
  // often longer than the default.
  rows[0]!.manifest_path = `${LONG}bubo-package.json`
  view.serving = false
  view.current_model = undefined
  view.models!.prepared_models = rows
  view.prepared_models = rows
  return view
}

const sizes: [number, number][] = [
  [900, 640], // the app's default window
  [700, 500],
  [560, 360], // its minimum window
  [1100, 760], // the largest window the shots use
]
for (const [width, height] of sizes) {
  test(`the Installed table fits its box with every action and size visible at ${width} x ${height}`, async () => {
    vi.spyOn(actions, 'Estimate').mockResolvedValue({
      fit: { sentence: 'Fits.' },
    } as never)
    await page.viewport(width, height)
    mountCorePage('/models', installedView())
    const region = page.getByRole('region', { name: 'Installed models' })
    await expect.element(region).toBeVisible()
    const container = region
      .element()
      .querySelector('[data-slot="table-container"]')!
    const box = container.getBoundingClientRect()
    // V1 V2 V5: no sideways scroll.
    expect(container.scrollWidth).toBeLessThanOrEqual(container.clientWidth + 1)
    // V3 V6: every control and the Size text is inside the box.
    const names = [
      'Select for Server',
      'Open folder',
      'Copy folder path',
      'Check or repair files',
    ]
    for (const row of Array.from(container.querySelectorAll('tbody tr'))) {
      for (const name of names) {
        const buttons = Array.from(row.querySelectorAll('button')).filter(
          (button) => button.textContent?.trim() === name,
        )
        if (name === 'Select for Server' || name === 'Open folder')
          expect(buttons.length, `${name} in a row`).toBeGreaterThan(0)
        for (const button of buttons) {
          const r = button.getBoundingClientRect()
          expect(r.width, name).toBeGreaterThan(0)
          expect(r.left, `${name} left`).toBeGreaterThanOrEqual(box.left - 1)
          expect(r.right, `${name} right`).toBeLessThanOrEqual(box.right + 1)
        }
      }
      for (const cell of Array.from(row.querySelectorAll('td'))) {
        const r = cell.getBoundingClientRect()
        expect(r.right, 'cell right').toBeLessThanOrEqual(box.right + 1)
        // The text inside a cell is not cut by it.
        expect(cell.scrollWidth, 'cell content').toBeLessThanOrEqual(
          cell.clientWidth + 1,
        )
      }
    }
    // V4: the long path wraps instead of widening the table.
    const path = Array.from(container.querySelectorAll('code')).find((code) =>
      code.textContent?.startsWith('C:\\Users\\fixture-with-a-long-name'),
    )!
    expect(path.getBoundingClientRect().right).toBeLessThanOrEqual(
      box.right + 1,
    )
    // V7: the Size, Status and Memory fit headers are all drawn in the box.
    for (const head of Array.from(container.querySelectorAll('th'))) {
      expect(head.getBoundingClientRect().right).toBeLessThanOrEqual(
        box.right + 1,
      )
    }
  })
}
