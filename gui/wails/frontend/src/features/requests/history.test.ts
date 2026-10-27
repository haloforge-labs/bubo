import { expect, test } from 'vitest'
import type { View } from '@/lib/bindings'
import busy from '@/scenarios/ready-busy.json'
import { requestSamples } from './history'
test('completed points use each CLI rate and timestamp with the CLI age, excluding failures and missing values', () => {
  const source = structuredClone(busy as unknown as View)
  const request = source.requests[0]!
  source.requests = [
    {
      ...request,
      time: '2000-01-01T00:00:00Z',
      decode_tps: 0,
      prefill_tps: 300,
    },
    {
      ...request,
      time: '2000-01-01T00:00:10Z',
      decode_tps: 50,
      prefill_tps: undefined,
    },
    {
      ...request,
      time: '2000-01-01T00:00:11Z',
      outcome: 'failed',
      decode_tps: 9999,
    },
  ]
  source.last_completed_request = {
    ...source.last_completed_request!,
    age_ms: 12000,
  }
  expect(
    requestSamples(source, 'decode').map(({ secondsAgo, value }) => ({
      secondsAgo,
      value,
    })),
  ).toEqual([
    { secondsAgo: 22, value: 0 },
    { secondsAgo: 12, value: 50 },
  ])
  expect(
    requestSamples(source, 'prefill').map(({ secondsAgo, value }) => ({
      secondsAgo,
      value,
    })),
  ).toEqual([{ secondsAgo: 22, value: 300 }])
  source.last_completed_request.age_ms = undefined
  expect(
    requestSamples(source, 'decode').map(({ secondsAgo, value }) => ({
      secondsAgo,
      value,
    })),
  ).toEqual([
    { secondsAgo: 10, value: 0 },
    { secondsAgo: 0, value: 50 },
  ])
})
