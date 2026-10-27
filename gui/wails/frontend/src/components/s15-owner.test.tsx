import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions, type View } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import { bytes } from '@/lib/format'
import {
  coreView,
  mountCorePage,
  recordedRuntimePlan,
  lowerMemoryPlanFixture,
} from '@/components/core-page-test-support'

afterEach(async () => {
  vi.restoreAllMocks()
  await page.viewport(1100, 760)
})

test('a clean profile waits for engine files before offering a model-file check', async () => {
  const view = coreView('not-prepared')
  view.runtime.readiness.runtime = false
  Object.assign(view.models!, {
    status: 'not_reported',
    reason_code: 'runtime_missing',
    reason: 'Fixture engine files are missing.',
    catalogue: [],
  })
  mountCorePage('/onboarding', view)
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('Model · After step 1')
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  await expect
    .element(
      page.getByRole('button', { name: 'Check for model files', exact: true }),
    )
    .not.toBeInTheDocument()
})

test('an unread catalogue explains the failure and offers a model-file check on Get started', async () => {
  const view = coreView('not-prepared')
  view.runtime.readiness.runtime = true
  Object.assign(view.models!, {
    status: 'not_reported',
    reason_code: 'query_timeout',
    reason: 'Fixture list timeout detail.',
    catalogue: [],
  })
  const refresh = vi
    .spyOn(actions, 'RefreshModels')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/onboarding', view)
  await expect.element(page.getByRole('alert')).toBeVisible()
  await expect
    .element(
      page.getByRole('button', { name: 'Check for model files', exact: true }),
    )
    .toBeVisible()
  await page
    .getByRole('button', { name: 'Check for model files', exact: true })
    .click()
  expect(refresh).toHaveBeenCalledOnce()
  await page.getByText('Details', { exact: true }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture list timeout detail.')
})

function shortDiskView() {
  const view = coreView('not-prepared')
  const runtime = recordedRuntimePlan()
  view.preparation = [runtime]
  view.models!.disk.drive = 'C:'
  view.models!.disk.free_bytes = runtime.volumes[0]!.free_bytes
  view.download_folder.change = { allowed: true, reason: '', reason_code: '' }
  return view
}

function spaceSentence(view: View) {
  const model = view.models!.catalogue.find(
    (row) => row.default_status === 'default',
  )!
  const engine = view.preparation
    .flatMap((plan) => plan.files)
    .reduce((sum, file) => sum + file.size, 0)
  return `Needs ${bytes(model.download_bytes + engine)} free; C: has ${bytes(view.models!.disk.free_bytes!)}.`
}
const roomyBytes = lowerMemoryPlanFixture.plan.volumes[0]!.free_bytes

test('short first-run disk space offers the shared picker before downloading and refreshes after a choice', async () => {
  const view = shortDiskView()
  const pick = vi.spyOn(actions, 'BrowseDownloadFolder').mockResolvedValue(
    bridgeReply({
      cancelled: false,
      folder: 'D:\\Bubo',
      added: true,
      message: '',
    }),
  )
  const refresh = vi
    .spyOn(actions, 'RefreshModels')
    .mockResolvedValue(bridgeReply({}))
  const { replaceView } = mountCorePage('/onboarding', view)
  await expect.element(page.getByText(spaceSentence(view))).toBeVisible()
  await expect
    .element(
      page.getByRole('link', {
        name: 'Change the download folder in Settings.',
      }),
    )
    .toHaveAttribute('href', '/settings#download-folder')
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent('This choice applies to the model files.')
  await expect
    .element(page.getByLabelText('Setup steps'))
    .not.toHaveTextContent('engine installation stays in its current folder')
  await page.getByRole('button', { name: 'Choose download folder…' }).click()
  expect(pick).toHaveBeenCalledOnce()
  // BrowseDownloadFolder already refreshes the disk/list through its saved
  // preference write; do not enqueue a duplicate frontend refresh.
  expect(refresh).not.toHaveBeenCalled()
  const roomy = structuredClone(view)
  roomy.models!.disk.free_bytes = roomyBytes
  replaceView(roomy)
  await expect
    .element(page.getByRole('button', { name: 'Choose download folder…' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByText(spaceSentence(view)))
    .not.toBeInTheDocument()
})

test.each(['downloading_model', 'preparing_engine'] as const)(
  'first-run disk warning waits until %s finishes',
  async (phase) => {
    const view = shortDiskView()
    const idlePhase = view.phase
    view.phase = phase
    const { replaceView } = mountCorePage('/onboarding', view)
    await expect
      .element(page.getByLabelText('Setup steps'))
      .toHaveTextContent('In progress')
    await expect
      .element(page.getByText(spaceSentence(view)))
      .not.toBeInTheDocument()
    await expect
      .element(page.getByRole('button', { name: 'Choose download folder…' }))
      .not.toBeInTheDocument()

    const idle = structuredClone(view)
    idle.phase = idlePhase
    replaceView(idle)
    await expect.element(page.getByText(spaceSentence(view))).toBeVisible()
    await expect
      .element(page.getByRole('button', { name: 'Choose download folder…' }))
      .toBeVisible()
  },
)

test.each(['enough', 'unknown'] as const)(
  'first-run disk warning is absent when space is %s',
  async (space) => {
    const view = shortDiskView()
    view.models!.disk.free_bytes = space === 'enough' ? roomyBytes : undefined
    mountCorePage('/onboarding', view)
    await expect
      .element(page.getByRole('button', { name: 'Choose download folder…' }))
      .not.toBeInTheDocument()
    await expect
      .element(
        page.getByRole('link', {
          name: 'Change the download folder in Settings.',
        }),
      )
      .not.toBeInTheDocument()
  },
)

test('first-run space includes the reported engine download size', async () => {
  const view = shortDiskView()
  mountCorePage('/onboarding', view)
  await expect.element(page.getByText(spaceSentence(view))).toBeVisible()
})

test('Settings link opens the model download folder section', async () => {
  mountCorePage('/onboarding', shortDiskView())
  await page
    .getByRole('link', { name: 'Change the download folder in Settings.' })
    .click()
  await expect
    .element(page.getByRole('region', { name: 'Model download folder' }))
    .toBeVisible()
})

function runtimeView(sameFolder = false) {
  const view = coreView('preparing')
  const plan: View['preparation'][number] = recordedRuntimePlan()
  if (sameFolder) plan.paths.runtime = plan.paths.cache!
  view.preparation = [plan]
  view.approval = {
    plan_digest: plan.plan_digest,
    question: '',
    components: ['runtime'],
  }
  return view
}

test('runtime consent uses plain file labels, CLI sizes, named destinations, and download wording without states', async () => {
  const view = runtimeView()
  mountCorePage('/onboarding', view)
  const dialog = page.getByRole('dialog')
  await expect
    .element(dialog)
    .toHaveTextContent(
      `To download: ${bytes(view.preparation[0]!.files.reduce((sum, file) => sum + file.size, 0))}`,
    )
  await expect.element(dialog).not.toHaveTextContent('Planned files:')
  await page
    .getByText(`Files (${view.preparation[0]!.files.length})`, { exact: true })
    .click()
  const files = page.getByRole('list', { name: 'Download files' })
  for (const label of [
    'AMD engine core',
    'AMD engine libraries',
    'AMD GPU kernels (gfx1151)',
  ])
    await expect.element(files).toHaveTextContent(label)
  for (const file of view.preparation[0]!.files)
    await expect.element(files).toHaveTextContent(bytes(file.size))
  await expect.element(files).not.toHaveTextContent('.whl')
  await page.getByText('Engine folder', { exact: true }).click()
  await expect
    .element(dialog)
    .toHaveTextContent(`Downloads to ${view.preparation[0]!.paths.cache}`)
  await expect
    .element(dialog)
    .toHaveTextContent(`Installs to ${view.preparation[0]!.paths.runtime}`)
  const licences = page
    .getByText(`Read licence texts (${view.preparation[0]!.licences.length})`, {
      exact: true,
    })
    .element()
    .closest('details')!
  expect(licences.open).toBe(false)
})

test('runtime consent lists a shared download/install folder once', async () => {
  const view = runtimeView(true)
  mountCorePage('/onboarding', view)
  await page.getByText('Engine folder', { exact: true }).click()
  const details = page
    .getByText('Engine folder', { exact: true })
    .element()
    .closest('details')!
  expect(details.querySelectorAll('code')).toHaveLength(1)
})

test('runtime dialog confines long paths to a scrolling body with visible fixed actions at width 600', async () => {
  await page.viewport(600, 480)
  mountCorePage('/onboarding', runtimeView())
  await page.getByText('Engine folder', { exact: true }).click()
  const dialog = page.getByRole('dialog').element() as HTMLElement
  await expect.poll(() => dialog.scrollWidth <= dialog.clientWidth).toBe(true)
  const body = dialog.querySelector<HTMLElement>('[data-download-body]')!
  expect(body).not.toBeNull()
  expect(body.scrollHeight).toBeGreaterThan(body.clientHeight)
  expect(body.scrollWidth).toBeLessThanOrEqual(body.clientWidth)
  const footer = dialog.querySelector<HTMLElement>(
    '[data-slot="dialog-footer"]',
  )!
  const footerTop = footer.getBoundingClientRect().top
  body.scrollTop = body.scrollHeight
  expect(footer.getBoundingClientRect().top).toBe(footerTop)
  for (const name of ['Not now', 'Download engine files']) {
    const rect = page
      .getByRole('button', { name, exact: true })
      .element()
      .getBoundingClientRect()
    const bounds = dialog.getBoundingClientRect()
    expect(rect.left).toBeGreaterThanOrEqual(bounds.left)
    expect(rect.right).toBeLessThanOrEqual(bounds.right)
    expect(rect.bottom).toBeLessThanOrEqual(bounds.bottom)
    expect(
      body.contains(page.getByRole('button', { name, exact: true }).element()),
    ).toBe(false)
  }
})

test('sidebar uses the shipped owl mask and preserves the home label', async () => {
  mountCorePage('/onboarding', coreView('not-prepared'))
  await expect
    .element(page.getByRole('link', { name: 'Bubo home', exact: true }))
    .toBeVisible()
  const home = page
    .getByRole('link', { name: 'Bubo home', exact: true })
    .element()
  const owl = home.querySelector('[aria-hidden="true"]')!
  expect(owl).not.toBeNull()
  expect(getComputedStyle(owl).maskImage).toContain('/icons/owl.svg')
  // size-4, like the previous sidebar SVG, is one rem (18 px in this app).
  const oldIconSize = parseFloat(
    getComputedStyle(document.documentElement).fontSize,
  )
  expect(owl.getBoundingClientRect().width).toBe(oldIconSize)
  expect(owl.getBoundingClientRect().height).toBe(oldIconSize)
})
