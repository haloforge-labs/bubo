import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { mountCorePage } from '@/components/core-page-test-support'

// Input Views come from a recording-backed fake CLI process start through the
// production supervisor and ViewBuilder. Capture the production page in Chromium.
for (const state of ['starting', 'loading', 'ready']) {
  test(`Server start wording: ${state}`, async () => {
    const view = inject('startupViews')?.[state]
    expect(
      view,
      'retain the supervisor start sequence Views first',
    ).toBeDefined()
    if (!view) throw new Error('Missing supervisor startup View.')
    document.documentElement.dataset.theme = inject('theme')
    mountCorePage('/', structuredClone(view))
    await expect
      .element(page.getByRole('heading', { name: 'Server', exact: true }))
      .toBeVisible()
    await expect
      .element(page.getByLabelText('Server status').getByRole('status'))
      .toHaveTextContent(
        `${view.status_label}${state === 'ready' ? '' : ' · Please wait'}`,
      )
    expect(document.body.textContent).not.toContain('Status is stale')
    await document.fonts.ready
    await commands.captureFeature('server', state)
  })
}
