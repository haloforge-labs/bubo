import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

test('About missing version stays prose', () => {
  const view = fixture()
  view.product = undefined
  view.update.current_version = ''
  return capturePage({
    feature: 'about',
    scenario: 'missing-version',
    view,
    prepare: async () => {
      const placeholder = page.getByText('Version not reported', {
        exact: false,
      })
      await expect.element(placeholder).toBeVisible()
      expect(getComputedStyle(placeholder.element()).fontFamily).toContain(
        'ET Book',
      )
    },
  })
})

test('About embeds the generated packaging inventory', () =>
  capturePage({
    feature: 'about',
    scenario: 'generated-notices',
    view: fixture(),
    prepare: async () => {
      await page.getByText('Third-party notices', { exact: true }).click()
      page
        .getByText('Third-party notices', { exact: true })
        .element()
        .scrollIntoView({ block: 'start' })
    },
  }))

for (const state of [
  'none',
  'current',
  'available',
  'progress',
  'ready',
  'failed',
  'check-failed',
  'rollback_pending',
]) {
  const name = `update-${state}`
  test(`About generated ${name}`, () =>
    capturePage({ feature: 'about', scenario: name, view: fixture(name) }))
}

// E22 E-7: the card of an update whose helper stopped before it changed anything
// (code update_interrupted, the sentence of the W1 walk).
test('About update_interrupted', () => {
  const sentence =
    'The update stopped before it changed anything. Your installed version is unchanged.'
  const view = fixture('update-failed')
  view.update.error = {
    code: 'update_interrupted',
    component: 'update',
    stage: '',
    remedy: sentence,
    retryable: true,
  }
  view.update.last_result_notice = sentence
  view.update.check_failed = false
  return capturePage({
    feature: 'about',
    scenario: 'update-interrupted',
    view,
    prepare: async () => {
      await expect
        .element(
          page
            .getByRole('region', { name: 'About', exact: true })
            .getByRole('alert'),
        )
        .toHaveTextContent(sentence)
    },
  })
})
