import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { mountCorePage } from '@/components/core-page-test-support'
import { scenarios } from '../../tests/scenarios'

afterEach(() => vi.restoreAllMocks())

// E22 fix 8 (UX15 K15-02, T15-10).
//
// Source check first (Lifecycle in components/server-lifecycle.tsx): Start is
// drawn disabled with NO line when its reason is estimate_unavailable or
// readiness_unavailable while the fit is "unavailable" (the branch returns
// null). The recorded "stopped" scenario (and every update-* scenario built on
// it) is this state: a greyed Start server under "Stopped" and nothing that says
// why.
//
// Failure list, written before the code:
//  8a. "stopped": Start server is disabled and the row has no line under it.
//      -> the line "Checking memory…" (the sentence Go reports for this state)
//      is under it.
//  8b. Sweep: in no scenario is a disabled "Start server" the last thing in its
//      block; a sibling line follows it.
//  8c. The line goes when the estimate arrives: an enabled Start has no line.
test('8a: a greyed Start says why', async () => {
  mountCorePage(
    '/',
    structuredClone(scenarios.find((s) => s.name === 'stopped')!.view),
  )
  const start = page.getByRole('button', { name: 'Start server', exact: true })
  await expect.element(start).toBeDisabled()
  const row = start.element().parentElement!
  expect(row.nextElementSibling?.textContent).toBe('Checking memory…')
})

for (const { name, view } of scenarios) {
  test(`8b: ${name}: a disabled Start has one current reason`, async () => {
    mountCorePage('/', structuredClone(view))
    await expect.element(page.getByRole('heading', { level: 1 })).toBeVisible()
    const starts = page
      .getByRole('button', { name: 'Start server', exact: true })
      .elements() as HTMLButtonElement[]
    for (const start of starts) {
      if (!start.disabled) continue
      if (
        [
          'memory_tight',
          'memory_does_not_fit',
          'memory_does_not_fit_any',
        ].includes(view.actions.start.reason_code)
      ) {
        await expect.element(page.getByLabelText('Memory fit')).toBeVisible()
        await expect
          .element(page.getByLabelText('Memory fit'))
          .toHaveTextContent(view.fit.sentence)
        expect(
          start.parentElement!.nextElementSibling?.textContent ?? '',
        ).not.toContain('Blocked:')
        continue
      }
      const row = start.parentElement!
      expect(
        row.nextElementSibling?.textContent?.trim(),
        'a line under the disabled Start',
      ).toBeTruthy()
    }
  })
}

test('8c: an enabled Start has no reason line', async () => {
  mountCorePage(
    '/',
    structuredClone(scenarios.find((s) => s.name === 'saved-no-verdict')!.view),
  )
  const start = page.getByRole('button', { name: 'Start server', exact: true })
  await expect.element(start).toBeEnabled()
  expect(
    start.element().parentElement!.nextElementSibling?.textContent ?? '',
  ).not.toContain('Checking memory…')
})
