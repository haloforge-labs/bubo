import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

afterEach(() => vi.restoreAllMocks())

test('Connect LAN write rejection leaves network access off', () => {
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'lan_unavailable',
      component: 'settings',
      stage: 'save',
      retryable: false,
      remedy: 'Review network preferences.',
    },
  })
  return capturePage({
    feature: 'connect',
    scenario: 'lan-write-rejected',
    view: fixture(),
    prepare: async () => {
      await page.getByRole('switch', { name: 'Local network access' }).click()
      await expect
        .element(page.getByRole('alert'))
        .toHaveTextContent(
          "Local network access isn't available in this copy of Bubo.",
        )
    },
  })
})
