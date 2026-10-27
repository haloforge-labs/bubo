import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

for (const feature of ['onboarding', 'models'] as const) {
  for (const name of [
    'onboarding-96-64',
    'onboarding-96-72',
    'onboarding-128-96',
  ]) {
    test(`${feature} memory: ${name}`, async () => {
      document.documentElement.dataset.theme = inject('theme')
      mountCorePage(`/${feature}`, coreView(name))
      await expect
        .element(
          page.getByRole('heading', {
            name: feature === 'onboarding' ? 'Get started' : 'Models',
            exact: true,
          }),
        )
        .toBeVisible()
      await document.fonts.ready
      if (feature === 'models') {
        const shell = document.getElementById('shell-capture')!
        shell.style.height = '760px'
        shell.style.overflow = 'auto'
        page
          .getByRole('region', { name: 'Discover models' })
          .element()
          .scrollIntoView({ block: 'start' })
      }
      expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
    })
  }
}
