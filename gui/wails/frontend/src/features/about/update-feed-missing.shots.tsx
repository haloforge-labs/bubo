import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

test('About missing update feed has one plain explanation', async () => {
  const view = fixture('update-check-failed')
  view.update.error = {
    code: 'update_information_missing',
    component: 'update',
    stage: '',
    retryable: true,
    remedy:
      'manifest returned HTTP 404 URL: https://github.com/haloforge-labs/bubo/releases/latest/download/bubo-release.json',
  }
  await capturePage({
    feature: 'about',
    scenario: 'update-feed-missing',
    view,
    prepare: async () => {
      const region = page.getByRole('region', { name: 'About', exact: true })
      await expect
        .element(region.getByRole('alert'))
        .toHaveTextContent(
          'No update information was found. Bubo checks again later.',
        )
      expect(region.getByRole('alert').elements()).toHaveLength(1)
      expect(document.body.textContent).not.toContain(
        'The download could not connect.',
      )
      expect(document.body.textContent).not.toContain('retry the same action')
    },
  })
})
