import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView } from './core-page-test-support'
import { capturePage } from './page-capture'

test('Semantic colors: Server protocol error', async () => {
  const view = coreView('stopped-saved-estimate')
  view.failure = {
    code: 'protocol_mismatch',
    remedy: '',
    source: 'supervisor',
    time: '',
    forced: false,
  }
  await capturePage({
    feature: 'server',
    scenario: 'semantic-colors-protocol-error',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('alert').getByText('Error', { exact: true }))
        .toBeVisible()
      await expect
        .element(page.getByText('Bubo and its runtime use different versions.'))
        .toBeVisible()
    },
  })
})

test('Semantic colors: Server normal', () =>
  capturePage({
    feature: 'server',
    scenario: 'semantic-colors-normal',
    view: coreView('stopped-saved-estimate'),
  }))

test('Semantic colors: Models', () =>
  capturePage({
    feature: 'models',
    scenario: 'semantic-colors-models',
    view: coreView(),
  }))
