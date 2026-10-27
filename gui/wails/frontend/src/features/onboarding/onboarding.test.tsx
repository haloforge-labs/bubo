import { bridgeReply } from '@/components/bridge-test-reply'
import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bytes } from '@/components/model-facts'
import {
  coreView,
  mountCorePage,
  recordedPlan,
  recordedRuntimePlan,
} from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('missing runtime offers Prepare first and reveals Download only after runtime readiness', async () => {
  const view = coreView('not-prepared')
  const { replaceView } = mountCorePage('/', view)
  await expect
    .element(
      page.getByRole('button', {
        name: 'Download the engine files',
        exact: true,
      }),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Download model…' }))
    .not.toBeInTheDocument()
  const ready = structuredClone(view)
  ready.runtime.readiness.runtime = true
  replaceView(ready)
  await expect
    .element(page.getByRole('button', { name: 'Download model…' }))
    .toBeVisible()
  await expect
    .element(
      page.getByText('runtime_missing: Run bubo prepare runtime.', {
        exact: true,
      }),
    )
    .not.toBeInTheDocument()
})

test('first run uses the recommended catalogue entry and shares the model download flow', async () => {
  const view = coreView('not-prepared')
  view.runtime.readiness.runtime = true
  view.actions.download_default = { allowed: true, reason_code: '', reason: '' }
  const model = view.models!.catalogue.find(
    (entry) =>
      entry.role === 'recommended' && entry.default_status === 'default',
  )!
  const plan = recordedPlan() as NonNullable<
    Awaited<ReturnType<typeof actions.CheckModel>>['plan']
  >
  const download = vi
    .spyOn(actions, 'Download')
    .mockResolvedValue(bridgeReply({ plan }))
  const confirm = vi
    .spyOn(actions, 'ConfirmDownload')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('heading', { name: 'Get started', exact: true }))
    .toBeVisible()
  await expect
    .element(
      page.getByText(
        `Bubo ${view.product!.version} is optimized for Qwen3.8 Flash Next on Strix Halo computers running Windows.`,
        { exact: false },
      ),
    )
    .toBeVisible()
  await expect
    .element(page.getByLabelText('Setup steps'))
    .toHaveTextContent(`${model.display_name} (default model)`)
  await page.getByRole('button', { name: 'Download model…' }).click()
  expect(download).toHaveBeenCalledExactlyOnceWith(model.id)
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Download', exact: true })
    .click()
  expect(confirm).toHaveBeenCalledExactlyOnceWith(model.id)
})

test('requirements link and disk size come from the catalogue', async () => {
  const view = coreView('not-prepared')
  const model = view.models!.catalogue.find(
    (entry) =>
      entry.role === 'recommended' && entry.default_status === 'default',
  )!
  model.download_bytes = 123456789
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('link', { name: 'install guide' }))
    .toHaveAttribute('href', 'https://haloforgelabs.com/docs/')
  expect(
    document.querySelector('[aria-label="Setup steps"]')?.textContent,
  ).toContain(bytes(model.download_bytes))
  expect(
    page.getByRole('region', { name: 'System requirements' }).element()
      .textContent,
  ).not.toContain('memory')
})

test('a preparing download shows the whole job, with the current file under Details', async () => {
  const view = coreView('preparing')
  const overall = view.progress!.overall!
  expect(overall.bytes_total).toBeGreaterThan(view.progress!.bytes_total!)
  mountCorePage('/', view)
  await expect
    .element(page.getByLabelText('Preparation progress'))
    .toBeVisible()
  await expect
    .element(page.getByRole('main'))
    .toHaveTextContent(bytes(overall.bytes_total))
  await page.getByText('Details', { exact: true }).click()
  await expect
    .element(page.getByRole('main'))
    .toHaveTextContent(
      `File ${view.preparation_file!.index} of ${view.preparation_file!.total}: ${bytes(view.progress!.bytes_done)} of ${bytes(view.progress!.bytes_total!)}`,
    )
})

test('prepare progress displays bytes and supports cancellation', async () => {
  const cancel = vi.spyOn(actions, 'Cancel').mockResolvedValue(bridgeReply({}))
  const view = coreView('preparing')
  // The fixture carries Go's whole-job number; this test is the case where Go
  // gave none (an older flow).
  view.progress!.overall = undefined
  const progress = view.progress!
  mountCorePage('/', view)
  // Go gave no whole-job number: the text line only, with no per-file bar.
  await expect
    .element(page.getByLabelText('Preparation progress'))
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByText(
        `Current file: ${bytes(progress.bytes_done)} of ${bytes(progress.bytes_total!)} · ${Math.round((progress.bytes_done / progress.bytes_total!) * 100)} %`,
      ),
    )
    .toBeVisible()
  const cancelButton = page.getByRole('button', { name: 'Cancel download' })
  expect(
    cancelButton.element().getBoundingClientRect().bottom,
  ).toBeLessThanOrEqual(document.documentElement.clientHeight)
  await cancelButton.click()
  expect(cancel).toHaveBeenCalledOnce()
})

test('prepare checks files before showing any approval and retains a typed failure', async () => {
  const prepare = vi.spyOn(actions, 'Prepare').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'unsupported',
        remedy: 'Fixture runtime is unavailable.',
        retryable: false,
        component: '',
        stage: '',
      },
    }),
  )
  mountCorePage('/', coreView('not-prepared'))
  await page.getByRole('button', { name: 'Download the engine files' }).click()
  expect(prepare).toHaveBeenCalledOnce()
  await expect
    .element(page.getByRole('alert').last())
    .toHaveTextContent('unsupported: Fixture runtime is unavailable.')
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
})

test('a preparation approval shows plan, destination and licence before accepting', async () => {
  const view = coreView('preparing')
  const plan = recordedPlan()
  view.approval = {
    question: 'Fixture approval question?',
    components: ['model'],
    plan_digest: plan.plan_digest,
  }
  const approve = vi
    .spyOn(actions, 'Approve')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  const dialog = page.getByRole('dialog', {
    name: `Download ${view.models!.catalogue[0]!.display_name}`,
  })
  await expect.element(dialog).toHaveTextContent(Object.values(plan.paths)[0]!)
  await expect
    .element(dialog.getByRole('link', { name: plan.licence.name }))
    .toHaveAttribute('href', plan.licence.url)
  expect(approve).not.toHaveBeenCalled()
  await dialog.getByRole('button', { name: 'Download and install' }).click()
  expect(approve).toHaveBeenCalledOnce()
})

test('first-start engine consent uses each plan licence text without a catalogue', async () => {
  const view = coreView('preparing')
  const runtime = recordedRuntimePlan()
  view.models!.catalogue = []
  view.phase = 'needs_engine'
  view.status_label = 'Setup needed: download the engine files'
  view.preparation = [runtime]
  view.approval = {
    question: '',
    components: ['runtime'],
    plan_digest: runtime.plan_digest,
  }
  const approve = vi
    .spyOn(actions, 'Approve')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/', view)
  const dialog = page.getByRole('dialog', {
    name: 'Download the engine files',
  })
  await expect.element(dialog).toBeVisible()
  expect(dialog.element().textContent).not.toContain('Model files')
  await expect
    .element(page.getByText('Another Bubo action is still running.'))
    .not.toBeInTheDocument()
  await expect
    .element(
      dialog.getByText('The download licence link is not reported by Bubo.'),
    )
    .not.toBeInTheDocument()
  await expect
    .element(
      dialog.getByText(`Read licence texts (${runtime.licences.length})`),
    )
    .toBeInTheDocument()
  await dialog
    .getByText(`Read licence texts (${runtime.licences.length})`)
    .click()
  const licence = runtime.licences[0]!
  const text = dialog
    .getByRole('heading', { name: licence.path, exact: true })
    .element()
    .closest('section')!
    .querySelector('pre')!.textContent!
  expect(text.replaceAll('\r\n', '\n')).toBe(
    licence.text.replaceAll('\r\n', '\n'),
  )
  await dialog.getByRole('button', { name: 'Download engine files' }).click()
  expect(approve).toHaveBeenCalledOnce()
})

test('prepared transition selects Server automatically', async () => {
  const { replaceView } = mountCorePage('/', coreView('not-prepared'))
  await expect
    .element(page.getByRole('heading', { name: 'Get started', exact: true }))
    .toBeVisible()
  replaceView(coreView())
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByRole('heading', { name: 'Get started', exact: true }))
    .not.toBeInTheDocument()
})

test('ordinary setup is neutral and has one primary action', async () => {
  mountCorePage('/', coreView('not-prepared'))
  await expect
    .element(page.getByRole('heading', { name: 'Get started', exact: true }))
    .toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  const steps = page.getByLabelText('Setup steps')
  await expect.element(steps).toHaveTextContent('1 Engine files · Next')
  await expect
    .element(steps)
    .toHaveTextContent('(default model) · After step 1')
  expect(
    document.querySelectorAll('[aria-label="Get started"] button.bg-brand'),
  ).toHaveLength(1)
})
