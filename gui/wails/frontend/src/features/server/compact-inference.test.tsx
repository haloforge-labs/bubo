import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import { useSetupRegistry } from './components/setup-registry'
import type { View } from '@/lib/bindings'

let cleanup = () => {}
afterEach(() => cleanup())
function Inference({ view }: { view: View }) {
  const registry = useSetupRegistry(view)
  return (
    <section aria-label="Inference">{registry.render('Inference')}</section>
  )
}
test('Inference starts with three side by side sampling controls and hides the remaining rows', async () => {
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
  expect(rects[1]!.top).toBe(rects[2]!.top)
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
  ).toBeLessThan(720)
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
    .element(page.getByRole('button', { name: 'Set Temperature', exact: true }))
    .not.toBeInTheDocument()
})
