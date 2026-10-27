import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

for (const path of ['/', '/settings'] as const) {
  test(`${path} clears only the CLI-identified rejected override`, async () => {
    const view = coreView('switch-override-rejected')
    const name = view.failure!.switch!
    const label = view.switches.find((row) => row.name === name)!.label
    const clear = vi.spyOn(actions, 'SetSwitchOverrides').mockResolvedValue({})
    mountCorePage(path, view)
    await expect
      .element(page.getByRole('region', { name: 'Rejected override' }))
      .toHaveTextContent(label!)
    await expect
      .element(
        page
          .getByRole('region', { name: 'Rejected override' })
          .getByRole('alert')
          .getByText('Error', { exact: true }),
      )
      .toBeVisible()
    if (path === '/') {
      await expect
        .element(
          page.getByRole('button', {
            name: 'Check or repair Bubo files',
            exact: true,
          }),
        )
        .not.toBeInTheDocument()
      expect(
        page
          .getByRole('button', { name: `Reset ${label}`, exact: true })
          .element().className,
      ).toContain('bg-brand')
    }
    await page
      .getByRole('button', { name: `Reset ${label}`, exact: true })
      .click()
    expect(clear).toHaveBeenCalledExactlyOnceWith({ set: [], clear: [name] })
  })
}

test('absent offending identifier never guesses a switch from the remedy', async () => {
  const view = coreView('switch-override-rejected')
  view.failure!.switch = undefined
  const clear = vi.spyOn(actions, 'SetSwitchOverrides')
  mountCorePage('/settings', view)
  await expect
    .element(page.getByRole('region', { name: 'Rejected override' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('alert').first())
    .toHaveTextContent(view.failure!.remedy)
  expect(clear).not.toHaveBeenCalled()
})

test('rejected clear preserves the CLI error and recovery action', async () => {
  const view = coreView('switch-override-rejected')
  vi.spyOn(actions, 'SetSwitchOverrides').mockResolvedValue({
    error: {
      code: 'fixture_clear_failed',
      remedy: 'Fixture clear remedy.',
      retryable: false,
      component: '',
      stage: '',
    },
  })
  mountCorePage('/', view)
  const region = page.getByRole('region', { name: 'Rejected override' })
  await region.getByRole('button').click()
  await expect
    .element(
      region
        .getByRole('alert')
        .filter({ hasText: 'fixture_clear_failed: Fixture clear remedy.' }),
    )
    .toHaveTextContent('fixture_clear_failed: Fixture clear remedy.')
  expect(region.getByRole('alert').elements()).toHaveLength(2)
  await expect.element(region.getByRole('button')).toBeEnabled()
})
