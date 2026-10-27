import { expect, inject, test } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { renderView } from '@/components/test-view'
import RequestsPage from './index'
import recorded from './walk-recorded.json'
import { liveRequestView } from './live-request-test-support'
import { AppShell } from '@/components/app-shell'

for (const width of [1100, 560]) {
  test(`recorded Requests charts at ${width}px and 100 percent`, async () => {
    const theme = inject('theme')
    const output = inject('chartShotsDirectory')
    document.documentElement.dataset.theme = theme
    await page.viewport(width, 900)
    const view = liveRequestView()
    view.requests = recorded
    view.last_completed_request = {
      ...view.last_completed_request!,
      age_ms: 58000,
    }
    const mounted = renderView(
      <AppShell>
        <RequestsPage />
      </AppShell>,
      view,
    )
    try {
      await expect
        .element(
          page.getByRole('figure', {
            name: 'Time to first token per completed request',
          }),
        )
        .toBeVisible()
      await document.fonts.ready
      for (const [name, title] of [
        ['decode', 'Decode per completed request'],
        ['prefill', 'Prefill per completed request'],
        ['ttft', 'Time to first token per completed request'],
        ['live', 'Live request rates · Session 1'],
      ]) {
        const chart = page.getByRole('figure', { name: title, exact: true })
        await expect.element(chart).toBeVisible()
        const root = chart.element()
        expect(root.getBoundingClientRect().width).toBeGreaterThanOrEqual(280)
        await expect
          .poll(
            () =>
              root.querySelectorAll('.recharts-yAxis-tick-labels text').length,
          )
          .toBeGreaterThan(0)
        const bounds = root
          .querySelector('.recharts-surface')!
          .getBoundingClientRect()
        for (const tick of root.querySelectorAll('svg text')) {
          const box = tick.getBoundingClientRect()
          expect(box.left).toBeGreaterThanOrEqual(bounds.left)
          expect(box.right).toBeLessThanOrEqual(bounds.right)
        }
        await chart.screenshot({
          path: `${output}/${name}-${theme}-${width}-linear.png`,
        })
        if (name !== 'live') {
          await userEvent.click(
            chart.getByRole('checkbox', { name: 'Log scale' }),
          )
          await expect.element(chart).toHaveTextContent('expands small values')
          await chart.screenshot({
            path: `${output}/${name}-${theme}-${width}-log.png`,
          })
          const point =
            name === 'ttft'
              ? chart.getByRole('button', { name: /^r19 ·/ })
              : chart.getByRole('button').first()
          ;(point.element() as SVGElement).focus()
          await expect.element(page.getByRole('tooltip')).toBeVisible()
          await chart.screenshot({
            path: `${output}/${name}-${theme}-${width}-tooltip.png`,
          })
          ;(point.element() as SVGElement).blur()
        }
      }
    } finally {
      mounted.cleanup()
    }
  })
}
