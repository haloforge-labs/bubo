import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { promptReadingView } from './progress-scenario'

test('Requests reading a cached prompt with reported counts', () =>
  capturePage({
    feature: 'requests',
    scenario: 'prompt-reading',
    view: promptReadingView(),
    prepare: async () => {
      await expect
        .element(page.getByRole('progressbar'))
        .toHaveAttribute('aria-valuenow', '1345')
    },
  }))

test('Requests reading a prompt before request history is available', () => {
  const view = promptReadingView()
  view.requests = []
  return capturePage({
    feature: 'requests',
    scenario: 'prompt-reading-no-history',
    view,
    prepare: async () => {
      await expect.element(page.getByRole('progressbar')).toBeVisible()
      await expect
        .element(page.getByText('No completed requests yet.'))
        .toBeVisible()
      await expect
        .element(page.getByText('No requests yet.', { exact: true }))
        .not.toBeInTheDocument()
      await expect
        .element(page.getByText('0 requests in history', { exact: true }))
        .toBeVisible()
    },
  })
})
