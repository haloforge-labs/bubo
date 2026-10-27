import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

const pausedSentence = 'The NPU is paused, so the GPU handles prompts for now.'
for (const [state, reason, wording] of [
  ['on', null, 'NPU prefill: on'],
  ['tripped', 'power', pausedSentence],
  ['faulted', 'fault', pausedSentence],
] as const) {
  test(`NPU ${state}/${reason} renders CLI state and Debug observations`, async () => {
    const view = coreView('npu-prefill-on')
    view.npu_prefill_runtime = {
      ...view.npu_prefill_runtime!,
      state,
      reason: reason ?? undefined,
    }
    const runtime = view.npu_prefill_runtime
    mountCorePage('/settings', view)
    await page.getByRole('tab', { name: 'Advanced' }).click()
    const hardware = page.getByRole('region', { name: 'NPU prefill status' })
    await expect.element(hardware).toHaveTextContent(wording)
    // Paused says so once: no "NPU prefill: paused" line over the sentence.
    if (state === 'tripped' || state === 'faulted')
      expect(hardware.element().textContent).not.toContain('NPU prefill:')
    await hardware.getByText('Details', { exact: true }).click()
    await expect
      .element(hardware)
      .toHaveTextContent(runtime.hal_faults.toLocaleString())
    await expect
      .element(hardware)
      .toHaveTextContent(`${runtime.chip_power.w_mean!.toLocaleString()} W`)
    await expect.element(hardware).toHaveTextContent(runtime.scheme!.name)
    await expect.element(hardware).toHaveTextContent(runtime.scheme!.guid)
    for (const [label, value] of [
      ['Gate holds', runtime.trips],
      ['HAL faults', runtime.hal_faults],
      ['WHEA events', runtime.whea_events],
      ['Invalid chunks', runtime.invalid_chunks],
    ] as const) {
      expect(
        hardware.getByText(label, { exact: true }).element().nextElementSibling
          ?.textContent,
      ).toBe(value.toLocaleString())
    }
  })
}

test('NPU off is the normal default: no line and no Debug panel', async () => {
  const view = coreView('ready-idle')
  view.npu_prefill_runtime = coreView('npu-prefill-off').npu_prefill_runtime
  expect(view.npu_prefill_runtime!.state).toBe('off')
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('region', { name: 'Hardware', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('region', { name: 'NPU prefill status' }))
    .not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('NPU prefill:')
})

test('an unavailable NPU draws nothing; null power never becomes zero', async () => {
  const view = coreView()
  const { replaceView } = mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  const hardware = page.getByRole('region', { name: 'NPU prefill status' })
  await expect.element(hardware).not.toBeInTheDocument()
  const next = coreView('npu-prefill-on')
  next.npu_prefill_runtime!.chip_power = {
    state: 'unavailable',
    w_mean: undefined,
    window_s: 0,
  }
  next.npu_prefill_runtime!.memory = undefined
  next.npu_prefill_runtime!.scheme = undefined
  replaceView(next)
  await hardware.getByText('Details', { exact: true }).click()
  await expect.element(hardware).toHaveTextContent('unavailable')
  expect(hardware.element().textContent).not.toContain('0 W')
  expect(
    [...hardware.element().querySelectorAll('dd')].filter(
      (cell) => cell.textContent === 'Unavailable',
    ),
  ).toHaveLength(3)
})
