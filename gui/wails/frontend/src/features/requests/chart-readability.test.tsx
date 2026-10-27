import { expect, test } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { RequestCharts } from './charts'
import { fixture } from '@/components/test-view'
// Completed HTTP events from the owner's 26 October walkthrough, including r19.
import recorded from './walk-recorded.json'
import { LiveRateChart } from './live-rate-chart'
import { liveRequestView } from './live-request-test-support'

function walkView() {
  const view = fixture('ready-idle')
  view.requests = recorded
  view.last_completed_request = {
    ...view.last_completed_request!,
    age_ms: 113000,
  }
  return view
}

test('TTFT uses seconds and whole milliseconds with one copy of each fact', async () => {
  await render(<RequestCharts view={walkView()} />)
  const chart = page.getByRole('figure', {
    name: 'Time to first token per completed request',
  })
  const long = chart.getByRole('button', { name: /^r19 ·/ })
  await expect.element(long).toBeVisible()
  await userEvent.hover(long)
  const text = page.getByRole('tooltip').element().textContent!
  expect(text).toContain('21.2 s to first token')
  expect(text.match(/to first token/g)).toHaveLength(1)
  expect(text).toContain('28,550 prompt tokens (0 cached)')
  expect(text).not.toContain('21,215')
  expect(chart.element().textContent).toContain('0 s')
  const small = [...chart.element().querySelectorAll('[data-request-id]')].find(
    (p) => /167 ms/.test(p.getAttribute('aria-label')!),
  )
  expect(small).toBeTruthy()
})

test('request order separates every recorded completion and is labeled', async () => {
  await render(<RequestCharts view={walkView()} />)
  const chart = page.getByRole('figure', {
    name: 'Time to first token per completed request',
  })
  await expect.element(chart.getByRole('button').first()).toBeVisible()
  expect(chart.element().textContent).toContain('Request order')
  const dots = [...chart.element().querySelectorAll('circle[data-request-id]')]
  const positions = dots.map((p) => Number(p.getAttribute('cx')))
  expect(new Set(positions).size).toBe(dots.length)
  expect(
    Math.min(...dots.map((p) => Number(p.getAttribute('r')))),
  ).toBeLessThanOrEqual(3)
})

test('log toggle expands small values and retains the outlier', async () => {
  await render(<RequestCharts view={walkView()} />)
  const chart = page.getByRole('figure', {
    name: 'Time to first token per completed request',
  })
  const long = chart.getByRole('button', { name: /^r19 ·/ })
  await expect.element(long).toBeVisible()
  const low = chart.element().querySelector('[data-request-id="r5"]')!
  const before = Number(low.getAttribute('cy'))
  await userEvent.click(chart.getByRole('checkbox', { name: 'Log scale' }))
  await expect
    .poll(() => Number(low.getAttribute('cy')))
    .toBeLessThan(before - 15)
  await expect.element(long).toBeVisible()
})

test('every axis label fits at a 560px window, including large rate values', async () => {
  const view = walkView()
  view.requests[0]!.prefill_tps = 40000
  await render(
    <div style={{ width: 520 }}>
      <RequestCharts view={view} />
    </div>,
  )
  await expect
    .poll(
      () =>
        document.querySelectorAll('.recharts-yAxis-tick-labels text').length,
    )
    .toBeGreaterThan(0)
  for (const svg of document.querySelectorAll<SVGSVGElement>(
    '.recharts-surface',
  )) {
    const bounds = svg.getBoundingClientRect()
    for (const tick of svg.querySelectorAll('text')) {
      const box = tick.getBoundingClientRect()
      expect(box.left).toBeGreaterThanOrEqual(bounds.left)
      expect(box.right).toBeLessThanOrEqual(bounds.right)
    }
  }
})

test('live rate axes include time units and fit large rates at minimum width', async () => {
  const view = liveRequestView()
  const samples = view.live_samples.map((s) => ({
    ...s,
    prefill_tps: 40000,
    phase: 'prefilling',
  }))
  await render(
    <div style={{ width: 400 }}>
      <LiveRateChart
        session={0}
        samples={samples}
        sampleTime={view.status_feed!.sample_time}
        active
      />
    </div>,
  )
  await expect
    .poll(
      () =>
        document.querySelectorAll('.recharts-xAxis-tick-labels text').length,
    )
    .toBeGreaterThan(0)
  expect(
    document.querySelector('.recharts-xAxis-tick-labels')!.textContent,
  ).toContain('s ago')
  const svg = document
    .querySelector('.recharts-surface')!
    .getBoundingClientRect()
  for (const tick of document.querySelectorAll(
    '.recharts-yAxis-tick-labels text',
  )) {
    expect(tick.getBoundingClientRect().left).toBeGreaterThanOrEqual(svg.left)
  }
})
