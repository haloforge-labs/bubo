import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import AboutPage from './index'
import { whatsNewRelease } from './whats-new'

let cleanup = () => {}
afterEach(() => cleanup())

// Written before implementation: reuse facts can disappear at the View/UI
// boundary, unknown values can become promises, changed files can be called
// reused, and the release text can be omitted or replaced with supplied prose.
test.each(['available', 'current'] as const)(
  'the %s update card says exactly which parts are not redownloaded',
  async (state) => {
    const view = fixture('update-available')
    view.update.state = state
    view.update.runtime_followup_required = state === 'current'
    expect(view.update.runtime_download_bytes).toBe(0)
    expect(view.update.models_download_bytes).toBe(0)
    cleanup = renderView(<AboutPage />, view).cleanup
    await expect
      .element(
        page.getByText('Engine files: not redownloaded.', { exact: true }),
      )
      .toBeVisible()
    await expect
      .element(page.getByText('Models: not redownloaded.', { exact: true }))
      .toBeVisible()
    await expect
      .element(page.getByText('Download size: 63.1 MiB.', { exact: true }))
      .toBeVisible()
  },
)

test.each([undefined, 1014229359])(
  'an unknown or changed engine has no reuse claim (%s)',
  async (size) => {
    const view = fixture('update-available')
    Object.assign(view.update, {
      runtime_download_bytes: size,
      models_download_bytes: undefined,
    })
    cleanup = renderView(<AboutPage />, view).cleanup
    await expect
      .element(page.getByRole('button', { name: /^Download update/ }))
      .toBeVisible()
    expect(document.body.textContent).not.toContain(
      'Engine files: not redownloaded.',
    )
    expect(document.body.textContent).not.toContain('Models: not redownloaded.')
  },
)

test('About ships the release text for this build', async () => {
  const view = fixture('update-current')
  // The fixture is a recorded 0.1.1 build; the bundled text is for the release being made.
  view.product = { ...view.product!, version: whatsNewRelease! }
  view.update.current_version = whatsNewRelease!
  cleanup = renderView(<AboutPage />, view).cleanup
  await expect
    .element(
      page.getByRole('heading', {
        name: `What's new in ${whatsNewRelease}`,
        exact: true,
      }),
    )
    .toBeVisible()
  const section = page.getByRole('region', {
    name: `What's new in ${whatsNewRelease}`,
  })
  await expect.element(section.getByRole('listitem').first()).toBeVisible()
  // The release text is a numbered list with its details nested under an item.
  const list = section.element().querySelector('ol')
  expect(list).not.toBeNull()
  expect(list!.children.length).toBeGreaterThanOrEqual(3)
  expect(section.element().textContent).not.toMatch(/(^|\s)1\. /)
  expect(document.body.textContent).not.toContain('Fixture release notes')
})
