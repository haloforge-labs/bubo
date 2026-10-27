import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import {
  coreView,
  mountCorePage,
  packagesView,
} from '@/components/core-page-test-support'

const configuredViewport = [1100, 760] as const
afterEach(async () => {
  vi.restoreAllMocks()
  await page.viewport(...configuredViewport)
})

// E22 fixes 3 and 4 (UX15 K15-08, T15-04, T15-06, N15-01, T15-07a).
//
// Failure list, written before the code:
//  3a. The Installed table has a Status column that is empty on most rows.
//      -> no Status column; "In use", "Incomplete" and "Downloading" sit under
//      the model name.
//  3b. A row with no memory verdict shows a blank cell, or the whole sentence
//      repeated down the column.
//      -> "—" in the cell, and one note under the table: "— Bubo checks memory
//      for this model when the server starts." Shown once, and only when a "—"
//      is in the table.
//  3c. "Checking memory…" stays on screen after the estimate has settled (two
//      copies showed it beside a blank).
//      -> it shows only while an estimate is in flight.
//  3d. "Select for Server" wraps to three lines in a narrow column.
//      -> one line at every window size.
//  4a. A downloading row can be selected and shows a fit.
//      -> Select is disabled and the row shows no fit text.
const NOTE =
  'Bubo has no memory estimate for these models yet; you can still select them and start the server.'

function settled(sentence: string, cls = 'fits') {
  return vi
    .spyOn(actions, 'Estimate')
    .mockResolvedValue({ fit: { sentence, class: cls } } as never)
}

function installed() {
  return page.getByRole('region', { name: 'Installed models' })
}

test('3a: there is no Status column; the status sits under the name', async () => {
  settled('Fits. 12 GiB of memory would be left.')
  const view = coreView()
  mountCorePage('/models', view)
  await expect.element(installed()).toBeVisible()
  const heads = Array.from(installed().element().querySelectorAll('th')).map(
    (head) => head.textContent,
  )
  expect(heads).not.toContain('Status')
  expect(heads[0]).toBe('Model / Path')
  // The running package: "In use" is inside the first cell of its row.
  const row = installed().getByRole('row').filter({ hasText: 'In use' })
  expect(row.elements()).toHaveLength(1)
  const first = row.element().querySelector('td')!
  expect(first.textContent).toContain('In use')
})

test('3a: an incomplete package says Incomplete under its name', async () => {
  settled('Fits.')
  const { view, rows } = packagesView()
  const partial = { ...rows[1]!, prepared_state: 'partial' }
  view.models!.prepared_models = [rows[0]!, partial]
  view.prepared_models = view.models!.prepared_models
  mountCorePage('/models', view)
  await expect.element(installed()).toBeVisible()
  const row = installed().getByRole('row').filter({ hasText: 'Incomplete' })
  expect(row.elements()).toHaveLength(1)
  expect(row.element().querySelector('td')!.textContent).toContain('Incomplete')
})

test('3b: a row with no verdict shows Not checked and the note stands once under the table', async () => {
  settled(
    'Bubo checks memory for this setup when the server starts.',
    'fit_unknown',
  )
  const { view } = packagesView()
  view.current_model = undefined
  mountCorePage('/models', view)
  await expect.element(installed()).toBeVisible()
  const cells = Array.from(
    installed().element().querySelectorAll('tbody tr'),
  ).map((row) => row.querySelectorAll('td')[2]?.textContent)
  expect(cells).toEqual(['Not checked', 'Not checked'])
  expect(installed().element().textContent).not.toContain(
    'Bubo checks memory for this setup',
  )
  expect(installed().element().textContent!.split(NOTE)).toHaveLength(2)
})

test('3b: the note is absent when every row has a verdict', async () => {
  settled('Fits. 12 GiB of memory would be left.')
  const { view } = packagesView()
  view.current_model = undefined
  mountCorePage('/models', view)
  await expect.element(installed()).toHaveTextContent('Fits. 12 GiB')
  expect(installed().element().textContent).not.toContain('— Bubo checks')
})

test('3c: "Checking memory…" shows while an estimate is in flight and not after', async () => {
  const releases: ((value: unknown) => void)[] = []
  vi.spyOn(actions, 'Estimate').mockImplementation(
    () =>
      new Promise((resolve) => {
        releases.push(resolve)
      }) as never,
  )
  const { view } = packagesView()
  view.current_model = undefined
  mountCorePage('/models', view)
  await expect.element(installed()).toHaveTextContent('Checking memory…')
  for (const release of releases)
    release({ fit: { sentence: 'Fits.', class: 'fits' } })
  await expect.element(installed()).not.toHaveTextContent('Checking memory…')
  await expect.element(installed()).toHaveTextContent('Fits.')
})

for (const [width, height] of [
  [900, 640],
  [560, 360],
  [1100, 760],
] as const) {
  test(`3d: "Select for Server" is one line at ${width} x ${height}`, async () => {
    settled('Fits.')
    await page.viewport(width, height)
    const { view } = packagesView()
    view.current_model = undefined
    mountCorePage('/models', view)
    const buttons = installed().getByRole('button', {
      name: 'Select for Server',
    })
    await expect.element(buttons.first()).toBeVisible()
    const container = installed()
      .element()
      .querySelector('[data-slot="table-container"]')!
    expect(container.scrollWidth).toBeLessThanOrEqual(container.clientWidth + 1)
    for (const button of buttons.elements()) {
      const line = parseFloat(getComputedStyle(button).lineHeight) || 20
      expect(button.getBoundingClientRect().height).toBeLessThan(line * 2)
      expect(getComputedStyle(button).whiteSpace).toBe('nowrap')
    }
  })
}

test('4a: a downloading row cannot be selected and shows no fit', async () => {
  const estimate = settled('Fits. 12 GiB of memory would be left.')
  const view = coreView('models-download-in-progress')
  const row = structuredClone(view.models!.prepared_models[0]!)
  row.prepared_state = 'partial'
  view.models!.prepared_models = [row]
  view.prepared_models = [row]
  view.models!.scanned = []
  view.models!.download = {
    ...view.models!.download!,
    id: row.id,
    state: 'running',
  }
  view.serving = false
  view.owned_server = false
  mountCorePage('/models', view)
  await expect.element(installed()).toBeVisible()
  const line = installed().getByRole('row').filter({ hasText: 'Downloading' })
  expect(line.elements()).toHaveLength(1)
  expect(line.element().querySelector('td')!.textContent).toContain(
    'Downloading',
  )
  await expect
    .element(line.getByRole('button', { name: 'Select for Server' }))
    .toBeDisabled()
  expect(installed().element().textContent).not.toContain('Checking memory')
  expect(installed().element().textContent).not.toContain('Fits.')
  expect(estimate).not.toHaveBeenCalled()
})
