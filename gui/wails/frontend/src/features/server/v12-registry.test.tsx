import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('one Server save combines core and inference settings and resets atomically', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = true
  const row = view.switch_overrides.find(
    (row) => row.name === '--reasoning-effort',
  )!
  row.saved = 'high'
  const nextContext =
    view.saved_estimate.memory_estimate!.recommended_context!
      .minimum_context_tokens
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: { ...view.saved_estimate }, fit: view.fit }),
  )
  mountCorePage('/', view)
  await page.getByLabelText('Context per session').fill(String(nextContext))
  await page
    .getByRole('spinbutton', { name: 'Temperature', exact: true })
    .fill('0.8')
  await page
    .getByRole('button', {
      name: 'Reset Reasoning effort to default',
      exact: true,
    })
    .click()
  await page
    .getByRole('button', { name: 'Save server setup', exact: true })
    .click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { context: nextContext },
    clear: [row.preference_field!],
    switch_values: [{ name: '--temperature', value: '0.8' }],
  })
})

test('promoted effort map uses CLI default levels and suggested bounds, then resets in the same Save', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = true
  const row = view.switches.find(
    (row) => row.name === 'GUFO_THINK_MARKER_BIAS_BY_EFFORT',
  )!
  expect(row.audience).toBe('user')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('low', { exact: true }))
    .toHaveValue(-2)
  await expect
    .element(page.getByLabelText('low', { exact: true }))
    .toHaveAttribute('min', '-2')
  await expect
    .element(page.getByLabelText('xhigh', { exact: true }))
    .toHaveAttribute('max', '0')
  await page.getByLabelText('low', { exact: true }).fill('-1.5')
  await page.getByRole('button', { name: 'Save server setup' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: row.name, value: 'low:-1.5,medium:-1,xhigh:0' }],
  })
  await page
    .getByRole('button', { name: `Reset ${row.label} to default`, exact: true })
    .click()
  await page.getByRole('button', { name: 'Save server setup' }).click()
  expect(save).toHaveBeenLastCalledWith({ set: {}, clear: [row.name] })
})

test('effort fields merge partial overrides with recorded default levels and bounds', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = true
  const row = view.switches.find(
    (row) => row.name === 'GUFO_THINK_MARKER_BIAS_BY_EFFORT',
  )!
  expect(row.audience).toBe('user')
  view.switch_overrides.find((state) => state.name === row.name)!.saved =
    'low:-1.5'
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('low', { exact: true }))
    .toHaveValue(-1.5)
  await expect
    .element(page.getByLabelText('xhigh', { exact: true }))
    .toHaveValue(0)
  await expect
    .element(page.getByLabelText('xhigh', { exact: true }))
    .toHaveAttribute('min', '-2')
})

test('an exact large default stays text while editing and saving', async () => {
  const view = coreView('stopped-saved-estimate')
  const row = view.switches.find((row) => row.name === 'GUFO_MAX_IMAGE_TOKENS')!
  row.allowed = {
    kind: 'object',
    properties: {
      type: { kind: 'string', text: 'integer' },
      maximum: { kind: 'number', text: '18446744073709551615' },
    },
  }
  const state = view.switch_overrides.find((state) => state.name === row.name)!
  state.saved = '9007199254740993'
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await page.getByRole('textbox', { name: 'Find a setting' }).fill(row.name)
  await expect
    .element(page.getByRole('textbox', { name: row.label! }))
    .toHaveValue('9007199254740993')
  await page.getByRole('textbox', { name: row.label! }).fill('9007199254740995')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [{ name: row.name, value: '9007199254740995' }],
  })
})
