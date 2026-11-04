import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { whatsNewRelease } from '@/features/about/whats-new'
import { fixture, renderView } from '@/components/test-view'
import { actions } from '@/lib/bindings'
import { WhatsNewDialog } from './whats-new-dialog'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// The fixture is a ready install of the version the bundled text describes.
function thisRelease(lastSeen?: string, setupDone = true) {
  const view = fixture('ready-idle')
  view.product = { ...view.product!, version: whatsNewRelease! }
  view.update.current_version = whatsNewRelease!
  view.runtime.readiness.runtime = setupDone
  view.runtime.readiness.model = setupDone
  view.saved.last_seen_version = lastSeen
  return view
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 100))
const seenPatch = { set: { last_seen_version: whatsNewRelease }, clear: [] }

test('the release text shows once after an update, then not again', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(<WhatsNewDialog />, thisRelease('0.0.1')).cleanup
  await expect
    .element(
      page.getByRole('dialog', { name: `What's new in ${whatsNewRelease}` }),
    )
    .toBeVisible()
  expect(save).not.toHaveBeenCalled()
  await page.getByRole('button', { name: 'OK', exact: true }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith(seenPatch)
  await expect.poll(() => page.getByRole('dialog').query()).toBeNull()

  // The next start sees the saved version.
  cleanup()
  cleanup = renderView(<WhatsNewDialog />, thisRelease(whatsNewRelease)).cleanup
  await settle()
  expect(page.getByRole('dialog').query()).toBeNull()
  expect(save).toHaveBeenCalledTimes(1)
})

test('an install from before versions were recorded counts as an update', async () => {
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(<WhatsNewDialog />, thisRelease()).cleanup
  await expect
    .element(
      page.getByRole('dialog', { name: `What's new in ${whatsNewRelease}` }),
    )
    .toBeVisible()
})

test('a new installation records the version and shows nothing', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(
    <WhatsNewDialog />,
    thisRelease(undefined, false),
  ).cleanup
  await expect.poll(() => save.mock.calls.length).toBe(1)
  expect(save).toHaveBeenCalledWith(seenPatch)
  expect(page.getByRole('dialog').query()).toBeNull()
})

test('a build whose text is for another version shows nothing', async () => {
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  const view = thisRelease('0.0.1')
  view.product = { ...view.product!, version: '0.0.1' }
  view.update.current_version = '0.0.1'
  cleanup = renderView(<WhatsNewDialog />, view).cleanup
  await settle()
  expect(page.getByRole('dialog').query()).toBeNull()
  expect(save).not.toHaveBeenCalled()
})
