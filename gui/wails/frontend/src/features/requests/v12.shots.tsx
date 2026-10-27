import { expect, test } from 'vitest'
import { commands, page, userEvent } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { realRequestsView } from './v12-scenario'

test('V12 Requests SPEC r3–r11 with r8 prefill tooltip open', () =>
  capturePage({
    feature: 'requests',
    scenario: 'v12-real-requests-table',
    view: realRequestsView(),
    prepare: async () => {
      const point = page
        .getByRole('figure', { name: 'Prefill per completed request' })
        .getByRole('button', { name: /^r8 ·/ })
      await userEvent.hover(point)
      await expect
        .element(page.getByRole('tooltip'))
        .toHaveTextContent('r8 · 12 s ago · Prefill 1,392 tok/s')
      await document.fonts.ready
      expect(
        await commands.captureFeature('requests', 'v12-real-requests-tooltip'),
      ).toMatch(/\.png$/)
      await userEvent.unhover(point)
      const table = page
        .getByRole('region', { name: 'Requests table' })
        .element()
      table.scrollIntoView({ block: 'start' })
      const viewport = table.querySelector('[data-slot="table-container"]')!
      viewport.scrollLeft = viewport.scrollWidth
    },
  }))
