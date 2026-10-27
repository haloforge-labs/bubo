import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { bridgeReply } from '@/components/bridge-test-reply'
import { renderView } from '@/components/test-view'
import { RegistrySection } from './registry-section'
import { useView } from '@/lib/bindings'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function RegistryFromView() {
  const view = useView().data
  return view ? <RegistrySection view={view} audience="user" /> : null
}

test('job: edit two settings, keep drafts across polls, save once, become clean', async () => {
  const view = coreView()
  view.switches = view.switches.filter((row) =>
    ['--temperature', '--top-p'].includes(row.name),
  )
  const save = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  const mounted = renderView(<RegistryFromView />, view)
  cleanup = mounted.cleanup
  const button = page.getByRole('button', { name: 'Save changes', exact: true })
  await expect.element(button).not.toBeInTheDocument()
  await page
    .getByRole('textbox', { name: 'Temperature', exact: true })
    .fill('0.61')
  await page
    .getByRole('textbox', {
      name: 'Top-p',
      exact: true,
    })
    .fill('0.91')
  await expect
    .element(page.getByText('2 unsaved changes', { exact: true }))
    .toBeVisible()
  await mounted.client.invalidateQueries()
  await expect
    .element(page.getByRole('textbox', { name: 'Temperature', exact: true }))
    .toHaveValue('0.61')
  await button.click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: {},
    switch_values: [
      { name: '--temperature', value: '0.61' },
      { name: '--top-p', value: '0.91' },
    ],
    clear: [],
  })
  await expect.element(button).not.toBeInTheDocument()
  await expect
    .element(page.getByText('0 unsaved changes', { exact: true }))
    .not.toBeInTheDocument()
})

test('tab and route navigation warn, Keep editing preserves values, discard leaves', async () => {
  mountCorePage('/settings', coreView())
  await page.getByRole('tab', { name: 'App', exact: true }).click()
  const runtime = page.getByRole('textbox', { name: 'Runtime folder' })
  await runtime.fill('D:/fixture/draft')
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  await expect
    .element(page.getByRole('dialog', { name: 'Unsaved changes' }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Keep editing' }).click()
  await expect.element(runtime).toHaveValue('D:/fixture/draft')
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .click()
  await expect
    .element(page.getByRole('dialog', { name: 'Unsaved changes' }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Keep editing' }).click()
  await expect.element(runtime).toHaveValue('D:/fixture/draft')
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  await page.getByRole('button', { name: 'Discard and leave' }).click()
  await expect
    .element(page.getByRole('button', { name: 'Debug', exact: true }))
    .toBeVisible()
  expect(
    page.getByRole('button', { name: 'Save changes', exact: true }).elements(),
  ).toHaveLength(0)
})

test('model preference row routes to Models without a free-text model control', async () => {
  const view = coreView()
  const row = view.switches.find((row) => row.name === '--model')!
  const route = view.switch_overrides.find((entry) => entry.name === row.name)!
  expect(route.write_via).toBe('preference')
  view.switches = [row]
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('link', { name: 'Choose a model' }))
    .toHaveAttribute('href', '/models')
  await expect
    .element(page.getByRole('textbox', { name: row.label! }))
    .not.toBeInTheDocument()
})

// This test starts from an unset Disk cache ("Not set" and a Set button). The
// release exe's Ready idle status reports it as the default text "Off", which
// the page reads as unset (registry.ts effectiveValue).
test('a rejected transaction retains all unsaved fields for correction', async () => {
  const view = coreView('ready-idle')
  view.switches = view.switches.filter((row) =>
    ['--context', '--cache-disk'].includes(row.name),
  )
  const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue(
    bridgeReply({
      error: { code: 'fixture_rejected', remedy: 'Review the cache folder.' },
    }),
  )
  const override = vi.spyOn(actions, 'SetSwitchOverrides')
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page
    .getByRole('spinbutton', { name: 'Context per session' })
    .fill('1234')
  await page.getByRole('button', { name: 'Set Disk cache' }).click()
  await page
    .getByRole('textbox', { name: 'Disk cache', exact: true })
    .fill('D:/fixture/cache')
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: [],
    switch_values: [
      { name: '--cache-disk', value: 'D:/fixture/cache' },
      { name: '--context', value: '1234' },
    ],
  })
  expect(override).not.toHaveBeenCalled()
  await expect
    .element(page.getByText('2 unsaved changes', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Review the cache folder.')
})

test('reset is a draft change and can clear an invalid edit only on Save', async () => {
  const view = coreView()
  const row = view.switches.find((row) => row.name === '--sessions')!
  view.switches = [row]
  view.switch_overrides.find((entry) => entry.name === row.name)!.saved = '2'
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  cleanup = renderView(
    <RegistrySection view={view} audience="user" />,
    view,
  ).cleanup
  await page.getByRole('spinbutton', { name: row.label! }).fill('-1')
  await page
    .getByRole('button', { name: 'Reset to default', exact: true })
    .click()
  expect(patch).not.toHaveBeenCalled()
  await expect
    .element(page.getByText('1 unsaved change', { exact: true }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Save changes' }).click()
  expect(patch).toHaveBeenCalledExactlyOnceWith({
    set: {},
    clear: ['sessions'],
  })
  await expect
    .element(page.getByRole('button', { name: 'Save changes' }))
    .not.toBeInTheDocument()
})
