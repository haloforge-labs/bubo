import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from './core-page-test-support'

function luminance(color: string) {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 1
  const context = canvas.getContext('2d')!
  context.fillStyle = color
  context.fillRect(0, 0, 1, 1)
  const linear = [...context.getImageData(0, 0, 1, 1).data]
    .slice(0, 3)
    .map((part) => {
      const value = part / 255
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
    })
  return linear[0]! * 0.2126 + linear[1]! * 0.7152 + linear[2]! * 0.0722
}

function contrast(a: string, b: string) {
  const values = [luminance(a), luminance(b)]
  return (Math.max(...values) + 0.05) / (Math.min(...values) + 0.05)
}

for (const theme of ['dark', 'light']) {
  test(`${theme}: Server titles, error and primary action share accessible owl coral`, async () => {
    document.documentElement.dataset.theme = theme
    const view = coreView('stopped-saved-estimate')
    view.failure = {
      code: 'protocol_mismatch',
      remedy: '',
      source: 'supervisor',
      time: '',
      forced: false,
    }
    const mounted = mountCorePage('/', view)
    const title = page.getByRole('heading', { name: 'Server', exact: true })
    await expect.element(title).toBeVisible()
    const error = page
      .getByRole('alert')
      .filter({ hasText: 'different versions' })
    await expect.element(error).toBeVisible()
    const label = error.getByText('Error', { exact: true }).element()
    const home = page.getByRole('link', { name: 'Bubo home' }).element()
    const owl = home.querySelector('[aria-hidden="true"]')!
    const sidebar = home.closest('[data-slot="sidebar-inner"]')!
    const coral = theme === 'dark' ? 'rgb(232, 90, 90)' : 'rgb(184, 63, 63)'
    expect(getComputedStyle(title.element()).color).toBe(coral)
    expect(getComputedStyle(label).color).toBe(coral)
    expect(getComputedStyle(owl).backgroundColor).toBe(coral)
    expect(getComputedStyle(home).backgroundColor).toBe(
      getComputedStyle(sidebar).backgroundColor,
    )
    const primary = document.querySelector('button.bg-brand')!
    expect(primary).not.toBeNull()
    expect(getComputedStyle(primary).backgroundColor).toBe(coral)
    const paper = getComputedStyle(document.documentElement).backgroundColor
    const panel = getComputedStyle(error.element())
    expect(contrast(coral, paper)).toBeGreaterThanOrEqual(4.5)
    expect(contrast(coral, panel.backgroundColor)).toBeGreaterThanOrEqual(4.5)
    expect(panel.backgroundColor).not.toBe(paper)
    expect(panel.borderLeftWidth).toBe('4px')
    expect(panel.borderLeftColor).toBe(coral)
    expect(label.querySelector('svg')).not.toBeNull()
    expect(getComputedStyle(home).color).toBe(
      theme === 'dark' ? 'rgb(139, 198, 204)' : 'rgb(36, 92, 100)',
    )
    const message = error.getByText(
      'Bubo and its runtime use different versions.',
    )
    expect(getComputedStyle(message.element()).color).toBe(coral)
    const save = page.getByRole('button', { name: 'Save server setup' })
    await expect.element(save).toBeDisabled()
    expect(getComputedStyle(save.element()).opacity).toBe('0.5')
    mounted.replaceView(coreView('stopped-saved-estimate'))
    const start = page.getByRole('button', {
      name: 'Start server',
      exact: true,
    })
    await expect.element(start).toBeEnabled()
    await expect
      .poll(() => getComputedStyle(start.element()).backgroundColor)
      .toBe(coral)
    expect(
      contrast(coral, getComputedStyle(start.element()).color),
    ).toBeGreaterThanOrEqual(4.5)
  })
}
