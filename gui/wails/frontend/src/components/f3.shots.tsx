import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from './core-page-test-support'

function frameForScroll() {
  const frame = document.getElementById('shell-capture')!
  frame.style.height = '100vh'
  frame.style.overflow = 'auto'
}

for (const name of ['memory-gpu-headroom', 'memory-default-setup']) {
  for (const [feature, route] of [
    ['server', '/'],
    ['onboarding', '/onboarding'],
    ['models', '/models'],
  ] as const) {
    test(`F3 ${feature} ${name}`, async () => {
      document.documentElement.dataset.theme = inject('theme')
      const view = coreView(name)
      mountCorePage(route, view)
      // The Server page says a refusal once, beside Start, with its one fix
      // (no separate Memory fit line); the recorded 8-session answer is one.
      const besideStart =
        feature === 'server' &&
        view.actions.start.reason_code === 'memory_does_not_fit'
      const content =
        feature === 'models'
          ? page.getByRole('region', { name: 'Discover models' })
          : besideStart
            ? page.getByText(view.fit.sentence)
            : page.getByLabelText('Memory fit')
      await expect.element(content).toBeVisible()
      await document.fonts.ready
      frameForScroll()
      content
        .element()
        .scrollIntoView({ block: feature === 'server' ? 'center' : 'start' })
      expect(await commands.captureFeature(feature, name)).toMatch(/\.png$/)
    })
  }
}
