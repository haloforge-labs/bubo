import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'

afterEach(() => vi.restoreAllMocks())

function shell() {
  const shell = document.getElementById('shell-capture')!
  shell.style.height = '760px'
  shell.style.overflow = 'auto'
}
async function capture(name: string, feature = 'server-v12') {
  await document.fonts.ready
  await commands.captureFeature(feature, name)
}

for (const [name, fixture] of [
  ['fits', 'stopped-saved-estimate'],
  ['tight', 'memory-gpu-headroom'],
  ['does-not-fit', 'memory-smaller-context'],
  ['does-not-fit-any-context', 'no-fit'],
] as const) {
  test(`V12 memory: ${name}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    const view = coreView(fixture)
    const patch = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
    const { replaceView } = mountCorePage('/', view)
    const panel = page.getByRole('region', { name: 'Memory fit' })
    await expect.element(panel).toBeVisible()
    shell()
    panel.element().scrollIntoView({ block: 'start' })
    await capture(name === 'tight' ? 'tight-unticked' : name)
    if (name === 'tight') {
      await page.getByRole('checkbox', { name: /^Override:/ }).click()
      expect(patch).toHaveBeenCalledExactlyOnceWith({
        set: { allow_tight_setup: true },
        clear: [],
      })
      replaceView({
        ...view,
        saved: { ...view.saved, allow_tight_setup: true },
        actions: {
          ...view.actions,
          start: { allowed: true, reason_code: '', reason: '' },
        },
      })
      await expect
        .element(page.getByRole('checkbox', { name: /^Override:/ }))
        .toBeChecked()
      panel.element().scrollIntoView({ block: 'start' })
      await capture('tight-ticked')
    }
  })
}

test('V12 Server sections and current figures', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-busy')
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Server setup' }))
    .toBeVisible()
  shell()
  await capture('setup-status')
  for (const name of ['Network', 'Addresses']) {
    const section = page.getByRole('region', { name, exact: true })
    await expect.element(section).toBeVisible()
    section.element().scrollIntoView({ block: 'start' })
    await capture(name.toLowerCase())
  }
  const inference = page.getByRole('button', { name: 'Inference', exact: true })
  await inference.click()
  await expect.element(inference).toHaveAttribute('aria-expanded', 'true')
  inference.element().scrollIntoView({ block: 'start' })
  await capture('inference')
  const effort = document.querySelector<HTMLElement>(
    '[data-setting="--reasoning-effort"]',
  )!
  await expect.element(effort).toBeVisible()
  effort.scrollIntoView({ block: 'start' })
  await capture('rethinking-card')
  await inference.click()
  page
    .getByRole('region', { name: 'Reported request rates' })
    .element()
    .scrollIntoView({ block: 'start' })
  await capture('last-request')
  page
    .getByRole('region', { name: 'Model in use' })
    .element()
    .scrollIntoView({ block: 'start' })
  await capture('served-name')
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
  for (const name of ['Hardware', 'Memory and caches']) {
    const section = page.getByRole('region', { name, exact: true })
    await expect.element(section).toBeVisible()
    section.element().scrollIntoView({ block: 'start' })
    await capture(name.toLowerCase().replaceAll(' ', '-'), 'settings')
  }
  const cache = document.querySelector<HTMLElement>(
    '[data-setting="--cache-disk"]',
  )!
  await expect.element(cache).toBeVisible()
  cache.scrollIntoView({ block: 'start' })
  await capture('disk-cache-card', 'settings')
})

test('V12 measured cached request with the CLI catalogue served ID', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-busy')

  view.requests = [
    {
      id: 'r5',
      time: '2026-10-26T04:00:08Z',
      outcome: 'completed',
      status: 200,
      finish_reason: 'stop',
      prompt_tokens: 17,
      cached_tokens: 17,
      ttft_ms: 8,
      decode_tps: 45.5,
      cache_kind: 'memory',
    },
  ]
  view.instance!.served_model = view.models!.catalogue[0]!.id
  mountCorePage('/', view)
  await expect
    .element(
      page.getByText('Prefill: all 17 tokens from cache', { exact: true }),
    )
    .toBeVisible()
  shell()
  page
    .getByRole('region', { name: 'Model in use' })
    .element()
    .scrollIntoView({ block: 'start' })
  await capture('synthetic-cache-hit-served-name')
})

test('V12 promoted effort map and busy-port recovery', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('stopped-saved-estimate')
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  view.failure = {
    code: 'port_occupied',
    remedy: '',
    source: 'supervisor',
    time: '',
    forced: false,
  }
  mountCorePage('/', view)
  await expect.element(page.getByLabelText('New server port')).toBeVisible()
  shell()
  await capture('busy-port')
  await page.getByRole('button', { name: 'Inference', exact: true }).click()
  document
    .querySelector('[data-setting="GUFO_THINK_MARKER_BIAS_BY_EFFORT"]')!
    .scrollIntoView({ block: 'start' })
  await capture('effort-map')
})
