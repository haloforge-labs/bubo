import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { coreView } from '@/components/core-page-test-support'

for (const expanded of [false, true]) {
  test(`Server Inference ${expanded ? 'expanded' : 'collapsed'}`, async () => {
    const view = coreView('stopped-saved-estimate')
    Object.assign(view.saved, { inference_expanded: expanded })
    await capturePage({
      feature: 'server',
      scenario: `inference-${expanded ? 'expanded' : 'collapsed'}`,
      view,
      prepare: async () => {
        const toggle = page.getByRole('button', {
          name: 'Inference',
          exact: true,
        })
        await expect
          .element(toggle)
          .toHaveAttribute('aria-expanded', String(expanded))
        toggle.element().closest('section')!.scrollIntoView({ block: 'start' })
        await document.fonts.ready
        console.log(
          `INFERENCE_${expanded ? 'EXPANDED' : 'COLLAPSED'}_HEIGHT=${Math.round(toggle.element().closest('section')!.getBoundingClientRect().height)}`,
        )
      },
    })
  })
}
