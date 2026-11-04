import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { applyFont, restoreFont } from '@/lib/font'
import { fixture, renderView } from '@/components/test-view'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import SettingsPage from './index'

let cleanup = () => {}
beforeEach(() => {
  localStorage.removeItem('bubo-font')
  delete document.documentElement.dataset.font
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  localStorage.removeItem('bubo-font')
  delete document.documentElement.dataset.font
})

const family = (node: Element) => getComputedStyle(node).fontFamily

test('Font is Serif by default; Sans serif saves at once and applies without a restart', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  const serif = page.getByRole('button', { name: 'Serif', exact: true })
  const sans = page.getByRole('button', { name: 'Sans serif', exact: true })
  await expect.element(serif).toHaveAttribute('aria-pressed', 'true')
  await expect.element(sans).toHaveAttribute('aria-pressed', 'false')
  await sans.click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { font: 'sans' },
    clear: [],
  })
  expect(document.documentElement.dataset.font).toBe('sans')
  expect(localStorage.getItem('bubo-font')).toBe('sans')
})

test('a saved sans font shows Sans serif as the chosen one', async () => {
  const view = fixture()
  view.saved.font = 'sans'
  cleanup = renderView(<SettingsPage />, view).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Sans serif', exact: true }))
    .toHaveAttribute('aria-pressed', 'true')
})

test('a rejected save puts the old font back', async () => {
  vi.spyOn(actions, 'PatchPreferences').mockRejectedValue({
    code: 'fixture_rejected',
    remedy: 'Review app preferences.',
  })
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  applyFont('serif')
  await page.getByRole('button', { name: 'Sans serif', exact: true }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Review app preferences.')
  expect(document.documentElement.dataset.font).toBe('serif')
})

test('the next start draws in the remembered font before the settings arrive', () => {
  restoreFont()
  expect(document.documentElement.dataset.font).toBe('serif')
  localStorage.setItem('bubo-font', 'sans')
  restoreFont()
  expect(document.documentElement.dataset.font).toBe('sans')
  localStorage.setItem('bubo-font', 'comic')
  restoreFont()
  expect(document.documentElement.dataset.font).toBe('serif')
})

test('the saved font reaches the whole app from the root', async () => {
  const sans = coreView('stopped')
  sans.saved.font = 'sans'
  const { replaceView } = mountCorePage('/', sans)
  await expect.poll(() => document.documentElement.dataset.font).toBe('sans')
  const serif = coreView('stopped')
  serif.saved.font = undefined
  replaceView(serif)
  await expect.poll(() => document.documentElement.dataset.font).toBe('serif')
})

test('one attribute swaps the typeface of the page and Go Mono stays for code', async () => {
  cleanup = renderView(<SettingsPage />, fixture()).cleanup
  await expect
    .element(page.getByRole('group', { name: 'Font', exact: true }))
    .toBeVisible()
  const heading = page
    .getByRole('group', { name: 'Font', exact: true })
    .element()
  const code = document.createElement('code')
  document.body.append(code)
  applyFont('serif')
  expect(family(heading)).toContain('ET Book')
  applyFont('sans')
  expect(family(heading)).toContain('IBM Plex Sans')
  expect(family(heading)).toContain('Segoe UI')
  expect(family(heading)).not.toContain('ET Book')
  expect(family(code)).toContain('Go Mono')
  code.remove()
})
