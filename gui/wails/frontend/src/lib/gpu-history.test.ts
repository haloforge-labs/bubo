import { expect, test } from 'vitest'
import type { View } from '@/lib/bindings'
import recorded from '@/scenarios/ready-idle.json'
import { appendSample } from './gpu-history'
const view = recorded as unknown as View
test('history retains the latest 300 observations without fabricating absent metrics', () => {
  const view = {
    ...recorded,
    rates: undefined,
    gpu: undefined,
  } as unknown as View
  let history: ReturnType<typeof appendSample> = []
  for (let at = 1; at <= 305; at++) history = appendSample(history, view, at)
  expect(history).toHaveLength(300)
  expect(history[0]?.at).toBe(6)
  expect(history.at(-1)?.at).toBe(305)
  expect(history.at(-1)?.gpu).toBeUndefined()
  expect(appendSample(history, view, 305)).toEqual(history)
})
test('a real zero and each CLI metric survive independently', () => {
  const sample = appendSample(
    [],
    {
      ...view,
      rates: {
        ...view.rates!,
        decode_tokens_per_second: 0,
        prefill_tokens_per_second: 37,
      },
      gpu: { use_pct: 0, chip_power_w: 73 },
    },
    1,
  )[0]!
  expect(sample.gpu?.use_pct).toBe(0)
  expect(sample.gpu?.chip_power_w).toBe(73)
})
