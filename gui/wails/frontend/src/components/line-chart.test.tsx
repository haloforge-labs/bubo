import { expect, test } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { MetricLineChart } from '@/components/line-chart'
import { RequestPointChart } from './request-point-chart'

const samples = Array.from({ length: 5 }, (_, index) => ({
  secondsAgo: 11 - index * 2,
  value: 70 + index,
}))

for (const theme of ['light', 'dark'])
  test(`${theme} request points stand alone on the time axis: no line joins them and no average is drawn`, async () => {
    document.documentElement.dataset.theme = theme
    await render(
      <RequestPointChart
        label="Decode per completed request"
        metricLabel="Decode"
        unit="tok/s"
        points={[
          { secondsAgo: 10, value: 210 },
          { secondsAgo: 2, value: 190 },
        ]}
      />,
    )
    await expect
      .poll(
        () =>
          document.querySelectorAll('.recharts-reference-dot circle').length,
      )
      .toBe(2)
    expect(document.querySelectorAll('.recharts-yAxis')).toHaveLength(1)
    expect(document.querySelector('.recharts-line-curve')).toBeNull()
    expect(document.body.textContent).not.toContain('average')
    expect(
      document.querySelector('.recharts-yAxis-tick-labels')!.textContent,
    ).toContain('300')
  })

test('charts wait for five samples and evenly position Y marks', async () => {
  const screen = await render(
    <MetricLineChart
      label="GPU samples"
      samples={samples.slice(0, 4)}
      unit="%"
      maximum={100}
    />,
  )
  await expect
    .element(page.getByRole('figure', { name: 'GPU samples' }))
    .toBeVisible()
  expect(document.querySelector('.recharts-line')).toBeNull()
  await screen.rerender(
    <MetricLineChart
      label="GPU samples"
      samples={samples}
      unit="%"
      maximum={100}
    />,
  )
  await expect
    .poll(() => document.querySelectorAll('.recharts-line').length)
    .toBe(1)
  const ticks = Array.from(
    document.querySelectorAll(
      '.recharts-yAxis-tick-labels .recharts-cartesian-axis-tick-value',
    ),
  )
  expect(ticks.map((tick) => tick.textContent)).toEqual([
    '0',
    '25',
    '50',
    '75',
    '100',
  ])
  const heights = Array.from(
    document.querySelectorAll('.recharts-yAxis-tick-lines line'),
    (tick) => Number(tick.getAttribute('y1')),
  )
  const gap = heights[1]! - heights[0]!
  for (let index = 1; index < heights.length; index++)
    expect(heights[index]! - heights[index - 1]!).toBeCloseTo(gap)
  await expect
    .element(page.getByText('GPU samples (%, last 300 s)', { exact: true }))
    .toBeVisible()
  expect(document.querySelectorAll('.recharts-label')).toHaveLength(0)
})

test('sparse step-after history ends at the last observed dot and clips old samples', async () => {
  await render(
    <MetricLineChart
      label="Decode history"
      unit="tok/s"
      samples={[{ secondsAgo: 301, value: 9000 }, ...samples]}
    />,
  )
  await expect
    .poll(
      () => document.querySelectorAll('.recharts-reference-dot circle').length,
    )
    .toBe(1)
  const curve = document.querySelector<SVGPathElement>('.recharts-line-curve')!
  const dot = document.querySelector('.recharts-reference-dot circle')!
  expect(document.querySelectorAll('.recharts-line-dot')).toHaveLength(0)
  const end = curve.getPointAtLength(curve.getTotalLength())
  expect(end.x).toBeCloseTo(Number(dot.getAttribute('cx')), 1)
  expect(end.y).toBeCloseTo(Number(dot.getAttribute('cy')), 1)
  expect(document.querySelectorAll('.recharts-label')).toHaveLength(0)
  const points = [
    ...curve.getAttribute('d')!.matchAll(/[ML]([\d.]+),([\d.]+)/g),
  ].map((match) => [Number(match[1]), Number(match[2])])
  expect(points).toHaveLength(samples.length * 2 - 1)
  for (let index = 1; index < points.length; index++) {
    const previous = points[index - 1]!
    const current = points[index]!
    // Each step is horizontal then vertical; never a sloped interpolation.
    if (index % 2 === 1) expect(current[1]).toBe(previous[1])
    else expect(current[0]).toBe(previous[0])
  }
  expect(
    document.querySelector('.recharts-yAxis-tick-labels')!.textContent,
  ).not.toContain('9,000')
  const nowTick = document.querySelector('.recharts-xAxis-tick-lines line')!
  expect(end.x).toBeLessThan(Number(nowTick.getAttribute('x1')))
})

test('dense histories omit point marks', async () => {
  await render(
    <MetricLineChart
      label="Dense history"
      unit="tok/s"
      samples={Array.from({ length: 20 }, (_, index) => ({
        secondsAgo: 40 - index * 2,
        value: index,
      }))}
    />,
  )
  await expect
    .poll(() => document.querySelectorAll('.recharts-line').length)
    .toBe(1)
  expect(document.querySelector('.recharts-line-dot')).toBeNull()
})

for (const theme of ['light', 'dark']) {
  test(`${theme} tooltip uses theme colors, includes a unit, and stays within the chart at its right edge`, async () => {
    document.documentElement.dataset.theme = theme
    await render(
      <MetricLineChart
        label="Chip power"
        unit="W"
        samples={samples.map((sample) => ({
          ...sample,
          secondsAgo: sample.secondsAgo - 3,
        }))}
      />,
    )
    await expect
      .poll(
        () =>
          document.querySelectorAll('.recharts-reference-dot circle').length,
      )
      .toBe(1)
    await userEvent.hover(
      document.querySelector('.recharts-reference-dot circle')!,
    )
    const tooltip = page.getByLabelText('Sample details')
    await expect.element(tooltip).toHaveTextContent('0 s ago · 74 W')
    const colors = getComputedStyle(tooltip.element())
    expect(colors.color).not.toBe(colors.backgroundColor)
    const box = tooltip.element().getBoundingClientRect()
    const chart = document
      .querySelector('.recharts-wrapper')!
      .getBoundingClientRect()
    expect(box.left).toBeGreaterThanOrEqual(chart.left)
    expect(box.right).toBeLessThanOrEqual(chart.right)
    document.documentElement.removeAttribute('data-theme')
  })
}
