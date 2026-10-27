import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { capturePage } from '@/components/page-capture'

test('review Cyber-Tiel card shows its abliterated base and licence once', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  const entry = view.models!.catalogue.find((row) =>
    row.id.startsWith('cyber-tiel'),
  )!
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: `About ${entry.display_name}`, exact: true })
    .click()
  const card = page.getByRole('dialog')
  await expect.element(card).toHaveTextContent(entry.about)
  expect(
    card.element().textContent!.split(entry.licence.prompt_line),
  ).toHaveLength(2)
  await document.fonts.ready
  await commands.captureFeature('models', 'review-cyber-card')
})

test('V12 Models Import header, published rows and CLI lower-memory model info', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-busy')
  const entry = view.models!.catalogue.find((entry) =>
    entry.id.includes('lowmemory'),
  )!
  expect(entry.about).toContain('GSQ-RCO by ISTA DAS Lab')
  expect(entry.about_links).toEqual([
    'https://arxiv.org/abs/2604.18556',
    'https://arxiv.org/abs/2605.00649',
  ])
  mountCorePage('/models', view)
  await expect
    .element(page.getByRole('button', { name: 'Import', exact: true }))
    .toBeVisible()
  await document.fonts.ready
  expect(await commands.captureFeature('models', 'v12-import-header')).toMatch(
    /\.png$/,
  )
  page
    .getByRole('region', { name: 'Discover models' })
    .element()
    .scrollIntoView({ block: 'start' })
  expect(await commands.captureFeature('models', 'v12-catalogue-list')).toMatch(
    /\.png$/,
  )
  page
    .getByRole('row')
    .filter({ hasText: 'DiffusionGemma 26B' })
    .element()
    .scrollIntoView({ block: 'end' })
  expect(await commands.captureFeature('models', 'v12-catalogue-tail')).toMatch(
    /\.png$/,
  )
  const info = page.getByRole('button', {
    name: `About ${entry.display_name}`,
    exact: true,
  })
  info.element().closest('tr')!.scrollIntoView({ block: 'center' })
  expect(entry.display_name).toBe(
    'Qwen3.8 Flash Next · HaloForge Labs lower-memory quant (GSQ-RCO)',
  )
  expect(view.models!.catalogue.every((row) => row.download_enabled)).toBe(true)
  await page
    .getByRole('button', { name: `About ${entry.display_name}`, exact: true })
    .click()
  await expect
    .element(page.getByRole('dialog', { name: entry.display_name }))
    .toBeVisible()
  await expect
    .element(page.getByText(entry.about, { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('link', { name: 'arXiv 2604.18556' }))
    .toBeVisible()
  document.getElementById('shell-capture')!.scrollTop = 0
  document.scrollingElement!.scrollTop = 0
  expect(await commands.captureFeature('models', 'v12-model-info')).toMatch(
    /\.png$/,
  )
})

for (const state of ['running', 'failed', 'cancelled']) {
  test(`V12 plain download state ${state}`, () => {
    const view = coreView('models-download-in-progress')
    const job = view.models!.download!
    job.state = state
    job.phase = 'verify'
    job.stalled = state === 'running'
    job.stall_seconds = state === 'running' ? 12 : undefined
    return capturePage({
      feature: 'models',
      scenario: `v12-download-${state}`,
      view,
      prepare: async () => {
        if (state === 'running')
          await expect
            .element(page.getByText('Download stalled for 12 seconds.'))
            .toBeVisible()
        else
          await expect
            .element(
              page.getByRole('heading', {
                name: new RegExp(`Download ${state}`),
              }),
            )
            .toBeVisible()
      },
    })
  })
}
