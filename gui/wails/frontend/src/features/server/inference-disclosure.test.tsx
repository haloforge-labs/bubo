import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'

afterEach(() => vi.restoreAllMocks())

test('Inference starts collapsed with current values and remembers the accessible toggle through preferences', async () => {
  const view = coreView('stopped-saved-estimate')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  const toggle = page.getByRole('button', { name: 'Inference', exact: true })
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false')
  const section = toggle.element().closest('section')!
  await expect
    .element(
      section.querySelector<HTMLInputElement>(
        '[data-setting="--temperature"] input',
      ),
    )
    .not.toBeVisible()
  const summary = (section as HTMLElement).innerText
  expect(summary).toContain('Temperature 1.0')
  expect(summary).toContain('Top-p 0.95')
  expect(summary).toContain('Top-k 20')
  expect(summary).toContain('Reasoning medium')
  await toggle.click()
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true')
  await expect
    .element(page.getByRole('spinbutton', { name: 'Temperature', exact: true }))
    .toBeVisible()
  expect(save).toHaveBeenCalledWith({
    set: { inference_expanded: true },
    clear: [],
  })
  await toggle.click()
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false')
  expect(save).toHaveBeenLastCalledWith({
    set: { inference_expanded: false },
    clear: [],
  })
})

test('saved expansion is restored, and collapsing preserves unsaved sampling and Seed edits in the summary', async () => {
  const view = coreView('stopped-saved-estimate')
  Object.assign(view.saved, { inference_expanded: true })
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  const toggle = page.getByRole('button', { name: 'Inference', exact: true })
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true')
  await page
    .getByRole('spinbutton', { name: 'Temperature', exact: true })
    .fill('0.8')
  await page.getByRole('spinbutton', { name: 'Seed', exact: true }).fill('42')
  await toggle.click()
  const section = toggle.element().closest('section')!
  expect(section.textContent).toContain('Temperature 0.8')
  expect(section.textContent).toContain('Seed 42')
  await toggle.click()
  await expect
    .element(page.getByRole('spinbutton', { name: 'Temperature', exact: true }))
    .toHaveValue(0.8)
  await expect
    .element(page.getByRole('spinbutton', { name: 'Seed', exact: true }))
    .toHaveValue(42)
  save.mockClear()
  await page
    .getByRole('button', { name: 'Save server setup', exact: true })
    .click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [
      { name: '--temperature', value: '0.8' },
      { name: '--seed', value: '42' },
    ],
  })
})

test('Seed is fourth in the main grid with a Random hint and the approved About help', async () => {
  const view = coreView('stopped-saved-estimate')
  Object.assign(view.saved, { inference_expanded: true })
  mountCorePage('/', view)
  const seed = page.getByRole('spinbutton', { name: 'Seed', exact: true })
  await expect.element(seed).toBeVisible()
  expect(
    [...seed.element().closest('section')!.querySelectorAll('[data-setting]')]
      .slice(0, 4)
      .map((row) => row.getAttribute('data-setting')),
  ).toEqual(['--temperature', '--top-p', '--top-k', '--seed'])
  const row = seed.element().closest('[data-setting]')!
  expect(row.closest('details')).toBeNull()
  expect(row.textContent).toContain('Random')
  await page.getByRole('button', { name: 'About Seed', exact: true }).click()
  await expect
    .element(
      page.getByText(
        'A fixed seed with the same prompt and settings gives the same answer. A request can still send its own seed.',
        { exact: true },
      ),
    )
    .toBeVisible()
})

test.each(['', '-1'])(
  'Seed %j means random, stays out of the collapsed summary, and saves with the setup',
  async (random) => {
    const view = coreView('stopped-saved-estimate')
    Object.assign(view.saved, { inference_expanded: true })
    const state = view.switch_overrides.find((row) => row.name === '--seed')!
    state.saved = '42'
    const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
    mountCorePage('/', view)
    await page
      .getByRole('spinbutton', { name: 'Seed', exact: true })
      .fill(random)
    const toggle = page.getByRole('button', { name: 'Inference', exact: true })
    await toggle.click()
    const section = toggle.element().closest('section')!
    const visibleText = (section as HTMLElement).innerText
    expect(visibleText).not.toContain('Seed ')
    save.mockClear()
    await page
      .getByRole('button', { name: 'Save server setup', exact: true })
      .click()
    await expect.poll(() => save.mock.calls.length).toBe(1)
    const patch = save.mock.calls[0]![0]
    if (random === '')
      expect(patch.clear).toContain(state.preference_field || '--seed')
    else
      expect(patch.switch_values).toContainEqual({
        name: '--seed',
        value: '-1',
      })
  },
)
