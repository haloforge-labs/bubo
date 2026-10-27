import { expect, test } from 'vitest'
import type { View } from '@/lib/bindings'
import ready from '@/scenarios/ready-idle.json'
import {
  bytes,
  fileName,
  folderPath,
  number,
  rate,
  percent,
  milliseconds,
  updatePresentation,
} from './format'
import { transferSummary } from '@/components/transfer-progress'

test('byte and number presentation scales units and keeps thousands separators', () => {
  expect(bytes(512)).toBe('512 B')
  expect(bytes(2048)).toBe('2.0 KiB')
  expect(bytes(24 * 1024 ** 2)).toBe('24.0 MiB')
  expect(bytes(24 * 1024 ** 3)).toBe('24.0 GiB')
  expect(number(1234567, 0)).toBe((1234567).toLocaleString())
})

test('path helpers split a file name and its folder', () => {
  expect(fileName('/fixture/file.gguf')).toBe('file.gguf')
  expect(folderPath('C:\\models\\model.gguf')).toBe('C:\\models\\')
})

test('transfer math uses observed elapsed time, leaves one-sample speed unknown, and retains zero', () => {
  expect(transferSummary(0, 100, [])).toBe('0 B of 100 B · 0 %')
  expect(
    transferSummary(25e6, 100e6, [
      { bytes: 0, at: 1000 },
      { bytes: 25e6, at: 2000 },
    ]),
  ).toContain('23.8 MiB/s · 3 s left')
  expect(transferSummary(1e6, 100e6, [{ bytes: 1e6, at: 2000 }])).not.toContain(
    '/s',
  )
})

test('update presentation preserves true failures', () => {
  const update = (structuredClone(ready) as unknown as View).update
  expect(
    updatePresentation({
      ...update,
      state: 'failed',
      error: { ...update.error!, code: 'network' },
    }),
  ).toBe('failed')
})

test('quantity formatters use one precision and retain reported zeroes', () => {
  expect(rate(43)).toBe('43.0')
  expect(rate(53.432)).toBe('53.4')
  expect(milliseconds(12.78)).toBe('13')
  expect(percent(24.6)).toBe('25 %')
  expect(rate(0)).toBe('0.0')
})
