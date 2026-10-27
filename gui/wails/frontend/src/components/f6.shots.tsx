import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from './core-page-test-support'

test('F6 stopped refusal offers the CLI context for the saved session count', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('memory-smaller-context')
  mountCorePage('/', view)
  const useButton = page.getByRole('button', { name: /^Use .* tokens$/ })
  await expect.element(useButton).toBeVisible()
  await expect
    .element(page.getByRole('region', { name: 'Memory fit', exact: true }))
    .toHaveTextContent(view.fit.sentence)
  await document.fonts.ready
  await commands.captureFeature('server', 'f6-stopped-context-remedy')
})
