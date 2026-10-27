import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

afterEach(() => vi.restoreAllMocks())

test('Settings names the CLI rejected override with exact recovery', () =>
  capturePage({
    feature: 'settings',
    scenario: 'rejected-override-recovery',
    view: fixture('switch-override-rejected'),
  }))

test('Settings Debug uses the pending override state supplied by the View', () =>
  capturePage({
    feature: 'settings',
    scenario: 'override-pending',
    view: fixture('switch-overrides-pending'),
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await page.getByRole('button', { name: 'Debug', exact: true }).click()
      await page
        .getByRole('textbox', { name: 'Filter debug settings' })
        .fill('GUFO_LOOP_BREAKER')
      page
        .getByRole('textbox', { name: 'Filter debug settings' })
        .element()
        .scrollIntoView({ block: 'start' })
    },
  }))

test('Settings Advanced write rejection remains next to its control', () => {
  const view = fixture()
  const row = view.switches.find((row) => row.name === '--draft-tokens')!
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'invalid_arguments',
      component: 'settings',
      stage: 'save',
      retryable: false,
      remedy: 'Review this value.',
    },
  })
  return capturePage({
    feature: 'settings',
    scenario: 'override-rejected',
    view,
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await page.getByRole('textbox', { name: 'Find a setting' }).fill(row.name)
      await page.getByRole('textbox', { name: row.label! }).fill('6')
      await page.getByRole('button', { name: 'Save changes' }).click()
      await expect
        .element(page.getByRole('alert'))
        .toHaveTextContent('A server setting needs attention.')
      page.getByRole('alert').element().scrollIntoView({ block: 'center' })
    },
  })
})

test('Settings app rejection remains next to app preferences', () => {
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'invalid_arguments',
      component: 'settings',
      stage: 'save',
      retryable: false,
      remedy: 'Review app preferences.',
    },
  })
  return capturePage({
    feature: 'settings',
    scenario: 'app-write-rejected',
    view: fixture(),
    prepare: async () => {
      await page.getByRole('tab', { name: 'App', exact: true }).click()
      await page
        .getByRole('textbox', { name: 'Runtime folder' })
        .fill('C:/bubo/runtimes/c4pkg-344f41ac/bin')
      await page.getByRole('button', { name: 'Save app settings' }).click()
      await expect
        .element(page.getByRole('alert'))
        .toHaveTextContent('A server setting needs attention.')
    },
  })
})
