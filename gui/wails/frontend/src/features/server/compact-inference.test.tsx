import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import { useSetupRegistry } from './components/setup-registry'
import type { View } from '@/lib/bindings'

let cleanup = () => {}
afterEach(() => cleanup())
function Inference({ view, width = 800 }: { view: View; width?: number }) {
  const registry = useSetupRegistry(view)
  // The rows follow the width of their container, not of the window: two
  // columns when the container is wide, one when it is narrow.
  return (
    <div className="@container" style={{ width }}>
      <section aria-label="Inference">{registry.render('Inference')}</section>
    </div>
  )
}
test('Inference lays its rows out in two columns on a wide page and hides the remaining rows', async () => {
  const view = fixture()
  cleanup = renderView(<Inference view={view} />, view).cleanup
  await expect
    .element(page.getByRole('group', { name: 'Temperature', exact: true }))
    .toBeVisible()
  const names = [...document.querySelectorAll('[data-setting]')].map((el) =>
    el.getAttribute('data-setting'),
  )
  expect(names.slice(0, 5)).toEqual([
    '--temperature',
    '--top-p',
    '--top-k',
    '--seed',
    '--reasoning-effort',
  ])
  const rects = names
    .slice(0, 3)
    .map((name) =>
      document
        .querySelector(`[data-setting="${name}"]`)!
        .getBoundingClientRect(),
    )
  expect(rects[0]!.top).toBe(rects[1]!.top)
  expect(rects[2]!.top).toBeGreaterThan(rects[0]!.top)
  expect(rects[1]!.left).toBeGreaterThan(rects[0]!.right)
  await expect
    .element(page.getByText('More inference settings', { exact: true }))
    .toBeVisible()
  await expect
    .element(
      page.getByRole('button', { name: 'About Reasoning effort', exact: true }),
    )
    .toBeVisible()
  expect(
    document.querySelector('[aria-label="Inference"]')!.getBoundingClientRect()
      .height,
  ).toBeLessThan(500)
})

test('Inference lays its rows out in one column when its container is narrow', async () => {
  const view = fixture()
  cleanup = renderView(<Inference view={view} width={300} />, view).cleanup
  await expect
    .element(page.getByRole('group', { name: 'Temperature', exact: true }))
    .toBeVisible()
  const rects = ['--temperature', '--top-p', '--top-k'].map((name) =>
    document.querySelector(`[data-setting="${name}"]`)!.getBoundingClientRect(),
  )
  expect(rects[1]!.top).toBeGreaterThan(rects[0]!.top)
  expect(rects[2]!.top).toBeGreaterThan(rects[1]!.top)
  expect(rects[1]!.left).toBe(rects[0]!.left)
})

test('stopped inference shows editable narrow sampling number inputs even before effective values arrive', async () => {
  const view = fixture('memory-default-setup')
  view.settings = []
  view.effective_switches = []
  view.switch_overrides = view.switch_overrides.map((row) => ({
    ...row,
    saved: undefined,
    effective: { kind: 'null' },
  }))
  cleanup = renderView(<Inference view={view} />, view).cleanup
  await expect
    .element(page.getByRole('spinbutton', { name: 'Temperature', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('spinbutton', { name: 'Top-p', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('spinbutton', { name: 'Top-k', exact: true }))
    .toBeVisible()
  await expect
    .element(
      page.getByRole('button', { name: 'Change Temperature', exact: true }),
    )
    .not.toBeInTheDocument()
})
