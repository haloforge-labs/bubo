import { expect, test } from 'vitest'
import type { View } from '@/lib/bindings'
import recorded from '@/scenarios/ready-idle.json'
import { contextFits, contextPreset, serverFormSchema } from './context'

const view = recorded as unknown as View
const table = view.memory_estimate!.recommended_context!

// Port of legacy presets.test.js. The reviewed lookup behavior remains;
// slider snapping and automatic context changes are replaced by validated input.
test.each([1, 3, 8])(
  'preset for %i sessions is the exact CLI row',
  (sessions) => {
    expect(contextPreset(table, sessions)).toEqual(
      table.by_sessions.find((row) => row.sessions === sessions),
    )
  },
)
test('missing table or unknown session count has no invented preset', () => {
  expect(contextPreset(undefined, 2)).toBeUndefined()
  expect(contextPreset(table, table.max_sessions + 1)).toBeUndefined()
  expect(contextFits(undefined, 2, 1)).toBeUndefined()
})
test('preset limits follow the independent recorded CLI table', () => {
  for (const row of table.by_sessions) {
    expect(contextFits(table, row.sessions, row.context_per_session)).toBe(
      row.fits,
    )
    expect(
      contextFits(
        table,
        row.sessions,
        row.context_per_session + table.step_tokens,
      ),
    ).toBe(false)
  }
})
test('more sessions never select a larger context', () => {
  for (let index = 1; index < table.by_sessions.length; index++)
    expect(table.by_sessions[index]!.context_per_session).toBeLessThanOrEqual(
      table.by_sessions[index - 1]!.context_per_session,
    )
})
test('typed context and session limits come from allowed and recommended_context', () => {
  const schema = serverFormSchema(view)
  const valid = {
    context: table.minimum_context_tokens,
    sessions: table.by_sessions[0]!.sessions,
    model: '',
  }
  expect(schema.safeParse(valid).success).toBe(true)
  expect(
    schema.safeParse({ ...valid, context: table.context_max_tokens + 1 })
      .success,
  ).toBe(false)
  expect(
    schema.safeParse({ ...valid, context: table.minimum_context_tokens - 1 })
      .success,
  ).toBe(false)
  expect(
    schema.safeParse({ ...valid, sessions: table.max_sessions + 1 }).success,
  ).toBe(false)
  expect(schema.safeParse({ ...valid, sessions: 1.5 }).success).toBe(false)
})
test('missing numeric contract is visibly unsupported', () => {
  expect(() =>
    serverFormSchema({ ...view, settings: [], switches: [] }),
  ).toThrow('no numeric allowed contract')
})
