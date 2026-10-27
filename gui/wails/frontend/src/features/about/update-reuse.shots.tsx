import { expect, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'
import release from '@/lib/release-test-data'

// The real reviewed 0.1.1 package over the published 0.1.0 package. Only the
// offer state is arranged; no synthetic versions, bytes, IDs or notes.
test('Update reuse published 0.1.0 to reviewed 0.1.1 update', () => {
  const view = fixture('update-available')
  view.product = {
    name: 'bubo',
    version: '0.1.0',
    revision: '3ff74dce',
    engine_version: '0.10.0',
  }
  view.update.current_version = '0.1.0'
  view.update.available_version = release.version
  const asset = release.assets[0]
  if (!asset) throw new Error('The recorded release package asset is missing.')
  view.update.size_bytes = asset.size
  view.update.notes = release.notes.summary
  return capturePage({
    feature: 'about',
    scenario: 'update-reuse-update',
    view,
    prepare: async () => {
      await commands.captureFeature('about', 'update-reuse-offer')
      await page.getByText('Update to 0.1.1', { exact: true }).click()
      await expect
        .element(page.getByText(release.notes.summary, { exact: true }))
        .toBeVisible()
    },
  })
})

test('Release text installed 0.1.1 release text', () =>
  capturePage({
    feature: 'about',
    scenario: 'update-reuse-whats-new',
    view: fixture('update-current'),
    prepare: async () => {
      const section = page.getByRole('region', { name: "What's new in 0.1.1" })
      if (section.elements().length)
        section.element().scrollIntoView({ block: 'start' })
    },
  }))
