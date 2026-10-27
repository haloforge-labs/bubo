import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture } from '@/components/test-view'
import { mountCorePage } from '@/components/core-page-test-support'
import { capturePage } from '@/components/page-capture'

afterEach(() => vi.restoreAllMocks())

test('Settings F5b one draft and one save bar', () => {
  const view = fixture()
  view.switches = view.switches.filter((row) =>
    ['--draft-tokens', '--prefill-chunk'].includes(row.name),
  )
  return capturePage({
    feature: 'settings',
    scenario: 'f5b-two-unsaved-changes',
    view,
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await page
        .getByRole('textbox', { name: 'Longest draft', exact: true })
        .fill('6')
      await page
        .getByRole('spinbutton', {
          name: 'Prompt tokens between generation rounds',
          exact: true,
        })
        .fill('256')
      await expect
        .element(page.getByText('2 unsaved changes', { exact: true }))
        .toBeVisible()
    },
  })
})

test('Settings F5b warns before leaving App edits', () =>
  capturePage({
    feature: 'settings',
    scenario: 'f5b-unsaved-navigation',
    view: fixture(),
    prepare: async () => {
      await page.getByRole('tab', { name: 'App', exact: true }).click()
      await page
        .getByRole('textbox', { name: 'Runtime folder' })
        .fill('C:/bubo/runtimes/c4pkg-344f41ac/bin')
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await expect
        .element(page.getByRole('dialog', { name: 'Unsaved changes' }))
        .toBeVisible()
    },
  }))

test('Settings F5b pending boolean keeps its CLI wire value behind a switch', () => {
  const view = fixture('switch-overrides-pending')
  const entry = view.switch_overrides.find((entry) => entry.saved === '0')!
  view.switches = view.switches.filter((row) => row.name === entry.name)
  return capturePage({
    feature: 'settings',
    scenario: 'f5b-boolean-switch',
    view,
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await page.getByRole('button', { name: 'Debug', exact: true }).click()
      await expect
        .element(page.getByRole('switch', { name: view.switches[0]!.label! }))
        .not.toBeChecked()
    },
  })
})

test('Settings F5b saves all edits with one action', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = fixture()
  view.switches = view.switches.filter((row) => row.name === '--draft-tokens')
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  const { replaceView } = mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  await page
    .getByRole('textbox', { name: 'Longest draft', exact: true })
    .fill('6')
  await page.getByRole('button', { name: 'Save changes' }).click()
  const saved = structuredClone(view)
  saved.switch_overrides.find(
    (entry) => entry.name === saved.switches[0]!.name,
  )!.saved = '6'
  replaceView(saved)
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByText('Set by you: 6', { exact: true }))
    .toBeVisible()
  await document.fonts.ready
  await commands.captureFeature('settings', 'f5b-saved-clean')
})
