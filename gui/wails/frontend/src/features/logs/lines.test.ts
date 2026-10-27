import { expect, test } from 'vitest'
import { copyLabel, logText, NO_LINES, NO_MATCH, shownLines } from './lines'

// Port of legacy logs-lines.test.mjs.
const lines = [
  'a serve: listening',
  'b serve: device_lost during decode',
  'c Serve: done',
]
test('no filter shows every line', () =>
  expect(logText(lines, '')).toBe(lines.join('\n')))
test('filter is a case insensitive contains', () =>
  expect(logText(lines, 'SERVE: d')).toBe(
    'b serve: device_lost during decode\nc Serve: done',
  ))
test('no match gives the no match message', () =>
  expect(logText(lines, 'zzz')).toBe(NO_MATCH))
test('empty log gives the no lines message with or without a filter', () => {
  expect(logText([], '')).toBe(NO_LINES)
  expect(logText(undefined, 'x')).toBe(NO_LINES)
})
test('copy copies shown lines and says so when filtered', () => {
  expect(shownLines(lines, 'done')).toEqual(['c Serve: done'])
  expect(copyLabel('x')).toBe('Copy shown lines')
  expect(copyLabel('')).toBe('Copy all')
})
