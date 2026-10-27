import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

for (const tab of ['App', 'Advanced', 'Debug']) {
  test(`Settings ${tab} from recorded ready-idle`, () =>
    capturePage({
      feature: 'settings',
      scenario: `ready-idle-${tab.toLowerCase()}`,
      view: fixture(),
      prepare: async () => {
        if (tab !== 'App')
          await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
        if (tab === 'Debug') {
          await page.getByRole('button', { name: 'Debug', exact: true }).click()
          page
            .getByRole('button', { name: 'Debug', exact: true })
            .element()
            .scrollIntoView({ block: 'start' })
        }
      },
    }))
}

test('Settings Advanced keeps an unmapped user setting reachable', () => {
  const view = fixture()
  // Arrange an existing CLI option in an ungrouped user-facing position.
  const row = view.switches.find((row) => row.name === '--max-connections')!
  row.audience = 'user'
  row.group = 'Additional settings'
  return capturePage({
    feature: 'settings',
    scenario: 'advanced-user-remainder',
    view,
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      await page
        .getByRole('textbox', { name: 'Find a setting' })
        .fill(row.label!)
      await expect
        .element(page.getByText(row.label!, { exact: true }))
        .toBeVisible()
    },
  })
})
