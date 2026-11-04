import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from './core-page-test-support'

// The walk-through shots of the font choice: Settings, a setup that does not fit
// (red) and a running server (Stop is the primary button), in both fonts.
const fonts = ['serif', 'sans'] as const
const pages = [
  ['settings', '/settings', 'stopped', 'Font'],
  ['no-fit', '/', 'no-fit', "Doesn't fit at any context"],
  ['running', '/', 'ready-idle', 'Stop server'],
] as const

for (const font of fonts)
  for (const [name, path, scenario, ready] of pages)
    test(`${name} in ${font}`, async () => {
      document.documentElement.dataset.theme = inject('theme')
      const view = coreView(scenario)
      view.saved.font = font
      mountCorePage(path, view)
      await expect
        .element(
          page
            .getByText(ready, { exact: true })
            .or(page.getByRole('heading', { name: ready, exact: true }))
            .first(),
        )
        .toBeVisible()
      await expect.poll(() => document.documentElement.dataset.font).toBe(font)
      await document.fonts.ready
      await commands.captureFeature('font', `${name}-${font}`)
    })
