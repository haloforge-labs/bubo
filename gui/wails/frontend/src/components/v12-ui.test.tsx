import { shortByShown } from '@/components/model-facts'
import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import {
  coreView,
  mountCorePage,
  recordedPlan,
  recordedRuntimePlan,
  combinedShortSpaceView,
} from './core-page-test-support'
import { fixture, renderView } from './test-view'
import { CopyText } from './copy-text'
import { ModelDownloadProgress } from './model-download'
import ConnectPage from '@/features/connect'
import AboutPage from '@/features/about'
import { GUIError } from './gui-error'
import { bridgeReply } from './bridge-test-reply'
import { CurrentFileDetails } from './transfer-progress'
import { Button } from './ui/button'
import { UpdateBanner } from './update-banner'
import { Preparation } from './preparation'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('combined short space has plain, actionable drive guidance with no alert inside a paragraph', async () => {
  const view = combinedShortSpaceView()
  const combined = view
  cleanup = renderView(<Preparation view={view} />, view).cleanup
  const guidance = page.getByText(
    `Free up space on C: ${shortByShown(combined.preparation.reduce((sum, plan) => sum + plan.volumes[0]!.bytes_to_write!, 0) + Math.max(...combined.preparation.map((plan) => plan.volumes[0]!.safety_margin_bytes!)), Math.min(...combined.preparation.map((plan) => plan.volumes[0]!.free_bytes)))} more is needed. Then try again.`,
    { exact: true },
  )
  await expect.element(guidance).toBeVisible()
  expect(guidance.element().tagName).toBe('P')
  expect(document.querySelector('p [role="alert"]')).toBeNull()
  expect(page.getByRole('dialog').getByRole('alert').elements()).toHaveLength(0)
})

test('a stalled download is a muted warning rather than an error', async () => {
  const view = fixture('models-download-in-progress')
  view.models!.download!.stalled = true
  view.models!.download!.stall_seconds = 12
  cleanup = renderView(
    <ModelDownloadProgress view={view} retry={() => {}} />,
    view,
  ).cleanup
  await expect
    .element(page.getByText('Download stalled for 12 seconds.'))
    .toBeVisible()
  expect(document.querySelector('[role="alert"]')).toBeNull()
})

test.each(['failed', 'cancelled'])(
  'a %s download has a plain heading',
  async (state) => {
    const view = fixture('models-download-in-progress')
    view.models!.download!.state = state
    view.models!.download!.phase = 'verify'
    cleanup = renderView(
      <ModelDownloadProgress view={view} retry={() => {}} />,
      view,
    ).cleanup
    await expect
      .element(
        page.getByRole('heading', { name: new RegExp(`Download ${state}`) }),
      )
      .toBeVisible()
  },
)

test('stopped Connect explains what will appear and links to Server without Start', async () => {
  cleanup = renderView(<ConnectPage />, fixture('stopped')).cleanup
  await expect
    .element(
      page.getByText(
        'No server is running, so there is nothing to connect to yet.',
      ),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('link', { name: 'Server page' }))
    .toHaveAttribute('href', '/')
  expect(document.body.textContent).not.toContain('appear here once it runs')
  await expect
    .element(page.getByRole('button', { name: 'Start server' }))
    .not.toBeInTheDocument()
})

test('icon-only copy has a tooltip and label, copies exact text, and resets after two seconds', async () => {
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(
    <CopyText iconOnly label="Copy version" text="Bubo fixture" />,
    fixture(),
  ).cleanup
  const button = page.getByRole('button', { name: 'Copy version' })
  await expect.element(button).toHaveTextContent('')
  await button.hover()
  await expect
    .element(page.getByRole('tooltip'))
    .toHaveTextContent('Copy version')
  await button.click()
  expect(copy).toHaveBeenCalledExactlyOnceWith('Bubo fixture')
  await expect.element(page.getByRole('status')).toHaveTextContent('Copied.')
  expect(button.element().querySelector('.lucide-check')).not.toBeNull()
  await expect
    .poll(() => button.element().querySelector('.lucide-copy'), {
      timeout: 3500,
    })
    .not.toBeNull()
  await expect.element(page.getByRole('status')).not.toBeInTheDocument()
})

test.each(['/', '/onboarding'] as const)(
  'setup sidebar at %s uses Get started, product version, and external footer links',
  async (path) => {
    const view = coreView('not-prepared')
    view.product = {
      name: 'Bubo',
      version: 'fixture-version',
      revision: 'fixture-revision',
      engine_version: '',
    }
    const external = vi.spyOn(actions, 'OpenExternal').mockResolvedValue({})
    mountCorePage(path, view)
    await expect
      .element(
        page
          .getByRole('navigation', { name: 'Main navigation' })
          .getByRole('link', { name: 'Get started', exact: true }),
      )
      .toHaveAttribute('aria-current', 'page')
    await expect
      .element(page.getByText('Halowin Bubo fixture-version'))
      .toBeVisible()
    await expect.element(page.getByText('by HaloForge Labs')).toBeVisible()
    const sidebar = page.getByLabelText('Sidebar', { exact: true })
    await sidebar.getByRole('link', { name: 'Website', exact: true }).click()
    expect(external).toHaveBeenLastCalledWith('https://haloforgelabs.com')
    await sidebar.getByRole('link', { name: 'GitHub', exact: true }).click()
    expect(external).toHaveBeenLastCalledWith(
      'https://github.com/haloforge-labs/bubo',
    )
  },
)

test('setup model choice offers all three published options and downloads the selected ID', async () => {
  const view = coreView('not-prepared')
  view.runtime.readiness.runtime = true
  view.actions.download_default = { allowed: true, reason_code: '', reason: '' }
  const recommended = view.models!.catalogue.find(
    (row) => row.role === 'recommended' && row.default_status === 'default',
  )!
  const family = view.models!.catalogue.filter(
    (row) => row.architecture === recommended.architecture,
  )
  const original = family.find(
    (row) => row.id !== recommended.id && row.download_enabled,
  )!
  const disabled = family.find(
    (row) => row.id !== recommended.id && row.id !== original.id,
  )!
  const download = vi
    .spyOn(actions, 'Download')
    .mockResolvedValue(bridgeReply({ plan: recordedPlan() }))
  mountCorePage('/onboarding', view)
  await expect
    .element(
      page.getByText(
        /optimized for Qwen3.8 Flash Next on Strix Halo computers running Windows/,
      ),
    )
    .toBeVisible()
  expect(document.body.textContent).not.toContain(recommended.id)
  const choice = page.getByRole('combobox', { name: 'Model to download' })
  await expect.element(choice).toHaveValue(recommended.id)
  expect(choice.getByRole('option').elements()).toHaveLength(family.length)
  await expect
    .element(
      choice.getByRole('option', {
        name: new RegExp(
          (disabled.selection_label || disabled.display_name).replace(
            /[.*+?^${}()|[\]\\]/g,
            '\\$&',
          ),
        ),
      }),
    )
    .toBeEnabled()
  expect(
    family.every((entry) => entry.download_enabled && !entry.disabled_reason),
  ).toBe(true)
  await choice.selectOptions(original.id)
  await page.getByRole('button', { name: 'Download model…' }).click()
  expect(download).toHaveBeenCalledExactlyOnceWith(original.id)
})

test('existing-model link navigates to the Import section', async () => {
  mountCorePage('/onboarding', coreView('not-prepared'))
  await expect
    .element(page.getByRole('link', { name: 'Models page', exact: true }))
    .toHaveAttribute('href', '/models')
  const link = page.getByRole('link', { name: 'I already have a model' })
  await expect.element(link).toHaveAttribute('href', '/models#import')
  await link.click()
  await expect
    .element(page.getByRole('region', { name: 'Import models' }))
    .toHaveAttribute('id', 'import')
})

test('About puts icon-only copy beside the installed version and opens release notes', async () => {
  const view = fixture('update-available')
  const external = vi.spyOn(actions, 'OpenExternal').mockResolvedValue({})
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<AboutPage />, view).cleanup
  const button = page.getByRole('button', { name: 'Copy version' })
  await expect.element(button).toHaveTextContent('')
  expect(button.element().parentElement?.parentElement?.textContent).toContain(
    view.product!.version,
  )
  await button.click()
  expect(copy).toHaveBeenCalledExactlyOnceWith(
    `Bubo ${view.product!.version} · ${view.product!.revision}`,
  )
  await page.getByRole('link', { name: 'Release notes', exact: true }).click()
  expect(external).toHaveBeenCalledExactlyOnceWith(
    'https://github.com/haloforge-labs/bubo/releases/tag/v' +
      view.product!.version,
  )
  expect(document.querySelector('details')?.textContent).toContain(
    view.update.notes,
  )
})

test('GUI errors carry an icon, an Error label and the Bubo release recovery', async () => {
  const external = vi.spyOn(actions, 'OpenExternal').mockResolvedValue({})
  cleanup = renderView(
    <GUIError
      error={{ code: 'cli_missing', remedy: 'Fixture missing CLI.' }}
    />,
    fixture(),
  ).cleanup
  const alert = page.getByRole('alert')
  await expect.element(alert.getByText('Error', { exact: true })).toBeVisible()
  expect(alert.element().querySelector('svg')).not.toBeNull()
  expect(
    parseFloat(getComputedStyle(alert.element()).borderLeftWidth),
  ).toBeGreaterThanOrEqual(3)
  expect(getComputedStyle(alert.element()).backgroundColor).not.toBe(
    'rgba(0, 0, 0, 0)',
  )
  await alert.getByRole('button', { name: 'Open download page' }).click()
  expect(external).toHaveBeenLastCalledWith(
    'https://github.com/haloforge-labs/bubo/releases',
  )
})

test('model download stays inside step two with one model Cancel and no stale engine progress', async () => {
  const view = coreView('models-download-in-progress')
  view.prepared = false
  view.phase = 'downloading_model'
  view.runtime.readiness.runtime = true
  view.runtime.readiness.model = false
  view.preparation = [recordedRuntimePlan()]
  view.progress = {
    component: 'runtime',
    phase: 'download',
    file_id: view.preparation[0]!.files[2]!.file_id,
    bytes_done: view.preparation[0]!.files[2]!.size,
    bytes_total: view.preparation[0]!.files[2]!.size,
    resume_offset: 0,
    verification_method: '',
    overall: { bytes_done: 1014229359, bytes_total: 1014229359 },
  }
  const cancel = vi.spyOn(actions, 'CancelDownload').mockResolvedValue({})
  const engineCancel = vi.spyOn(actions, 'Cancel').mockResolvedValue({})
  mountCorePage('/onboarding', view)
  const progress = page.getByRole('region', { name: 'Model download progress' })
  await expect.element(progress).toBeVisible()
  expect(progress.element().closest('li')?.textContent).toContain('2')
  await expect
    .element(page.getByLabelText('Preparation progress'))
    .not.toBeInTheDocument()
  expect(
    page.getByRole('button', { name: 'Cancel download' }).elements(),
  ).toHaveLength(1)
  await page.getByRole('button', { name: 'Cancel download' }).click()
  expect(cancel).toHaveBeenCalledOnce()
  expect(engineCancel).not.toHaveBeenCalled()
})

test('file details show plan filenames, sizes, source, destination and current checking state', async () => {
  const view = fixture('models-download-in-progress')
  const plan = recordedPlan()
  const job = view.models!.download!
  job.plan = plan
  job.file_id = plan.files[1]!.file_id
  job.file_position = { index: 2, total: plan.files.length }
  job.phase = 'verify'
  job.overall = {
    bytes_done: plan.files[0]!.size + plan.files[1]!.size,
    bytes_total: plan.files.reduce((sum, file) => sum + file.size, 0),
  }
  cleanup = renderView(
    <ModelDownloadProgress view={view} retry={() => {}} />,
    view,
  ).cleanup
  await page.getByText('Details', { exact: true }).click()
  const files = page.getByRole('list', { name: 'Download file progress' })
  await expect.element(files).toBeVisible()
  expect(files.element().children).toHaveLength(plan.files.length)
  expect(files.element().children[1]!.textContent).toContain('checking')
  await expect
    .element(files)
    .toHaveTextContent(plan.files[1]!.path.split(/[\\/]/).at(-1)!)
  await expect.element(files).toHaveTextContent(plan.files[1]!.repository)
  await expect.element(files).toHaveTextContent(plan.files[1]!.revision)
  expect(files.element().textContent).not.toContain('https://')
  await expect
    .element(
      page.getByText(Object.values(plan.paths)[0]!, { exact: false }).last(),
    )
    .toBeInTheDocument()
})

test('a successful model job is a temporary one-line ready notice', async () => {
  const view = fixture('models-download-in-progress')
  const job = view.models!.download!
  job.state = 'done'
  job.phase = 'verify'
  const dismiss = vi.spyOn(actions, 'DismissModelJob').mockResolvedValue({})
  cleanup = renderView(
    <ModelDownloadProgress view={view} retry={() => {}} />,
    view,
  ).cleanup
  const name = view.models!.catalogue.find(
    (entry) => entry.id === job.id,
  )!.display_name
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent(`${name} is ready.`)
  await expect
    .element(page.getByRole('heading', { name: /Checking files/ }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Dismiss' }))
    .not.toBeInTheDocument()
  await expect.poll(() => dismiss.mock.calls.length, { timeout: 6500 }).toBe(1)
  await expect.element(page.getByRole('status')).not.toBeInTheDocument()
})

test('the title owl has a coral mask with a loadable SVG source', async () => {
  document.documentElement.dataset.theme = 'dark'
  mountCorePage('/', coreView())
  await expect
    .element(page.getByRole('link', { name: 'Bubo home' }))
    .toBeVisible()
  const mark = document.querySelector(
    '[aria-label="Bubo home"] > span[aria-hidden="true"]',
  )!
  // size-4 is one rem; Bubo's root font size is 18 px.
  expect(mark.getBoundingClientRect().width).toBe(18)
  const style = getComputedStyle(mark)
  expect(style.backgroundColor).toBe('rgb(232, 90, 90)')
  const url = style.maskImage.match(/url\("?([^")]+)"?\)/)?.[1]
  expect(url).toBeTruthy()
  const source = await fetch(url!)
  expect(source.ok).toBe(true)
  expect(source.headers.get('content-type')).toContain('image/svg+xml')
  expect(await source.text()).toContain('<svg')
})

test('a reused file being checked says checking while other reused files say done', async () => {
  const plan = recordedPlan()
  plan.files.forEach((file) => {
    file.state = 'reuse'
  })
  cleanup = renderView(
    <CurrentFileDetails
      plan={plan}
      fileId={plan.files[1]!.file_id}
      phase="verify"
      done={3}
      total={12}
    />,
    fixture(),
  ).cleanup
  await page.getByText('Details', { exact: true }).click()
  const files = page
    .getByRole('list', { name: 'Download file progress' })
    .element().children
  expect(files[0]!.textContent).toContain('done')
  expect(files[1]!.textContent).toContain('checking')
})

test('a repair without a plan does not claim the normal download folder as its destination', async () => {
  const view = fixture('models-download-in-progress')
  view.download_folder.path = 'C:\\fixture\\normal-downloads'
  view.models!.download!.kind = 'repair'
  view.models!.download!.plan = undefined
  view.models!.download!.repair_output = undefined
  cleanup = renderView(
    <ModelDownloadProgress view={view} retry={() => {}} />,
    view,
  ).cleanup
  await page.getByText('Details', { exact: true }).click()
  expect(
    page.getByRole('region', { name: 'Model download progress' }).element()
      .textContent,
  ).not.toContain(`Save folder: ${view.download_folder.path}`)
})

test.each(['light', 'dark'])(
  'primary button text keeps 4.5:1 contrast at rest and on hover in %s mode',
  async (theme) => {
    document.documentElement.dataset.theme = theme
    cleanup = renderView(<Button>Download fixture</Button>, fixture()).cleanup
    const button = page.getByRole('button', { name: 'Download fixture' })
    await expect.element(button).toBeVisible()
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 1
    const context = canvas.getContext('2d')!
    const paper = getComputedStyle(document.documentElement).backgroundColor
    function luminance(color: string) {
      context.fillStyle = paper
      context.fillRect(0, 0, 1, 1)
      context.fillStyle = color
      context.fillRect(0, 0, 1, 1)
      const linear = [...context.getImageData(0, 0, 1, 1).data]
        .slice(0, 3)
        .map((part) => {
          const value = part / 255
          return value <= 0.04045
            ? value / 12.92
            : ((value + 0.055) / 1.055) ** 2.4
        })
      return linear[0]! * 0.2126 + linear[1]! * 0.7152 + linear[2]! * 0.0722
    }
    function contrast() {
      const style = getComputedStyle(button.element())
      const values = [luminance(style.color), luminance(style.backgroundColor)]
      return (Math.max(...values) + 0.05) / (Math.min(...values) + 0.05)
    }
    expect(contrast()).toBeGreaterThanOrEqual(4.5)
    await button.hover()
    await new Promise((resolve) => setTimeout(resolve, 250))
    expect(contrast()).toBeGreaterThanOrEqual(4.5)
  },
)

test('a failed update link also carries the Error icon, word and panel', async () => {
  const view = fixture('update-failed')
  cleanup = renderView(<UpdateBanner update={view.update} />, view).cleanup
  const link = page.getByRole('link', { name: /Update failed/ })
  const panel = page.getByRole('alert')
  await expect.element(panel.getByText('Error', { exact: true })).toBeVisible()
  await expect.element(link).toBeVisible()
  expect(panel.element().querySelector('svg')).not.toBeNull()
  expect(getComputedStyle(panel.element()).borderLeftWidth).toBe('4px')
  expect(getComputedStyle(panel.element()).backgroundColor).not.toBe(
    'rgba(0, 0, 0, 0)',
  )
})

test('the active sidebar marker leaves room for the full collapsed icon', async () => {
  document.documentElement.dataset.theme = 'dark'
  mountCorePage('/models', coreView())
  await expect
    .element(page.getByRole('heading', { name: 'Models', exact: true }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Toggle Sidebar' }).last().click()
  await expect
    .poll(() =>
      document
        .querySelector('[data-slot="sidebar"][data-state]')
        ?.getAttribute('data-state'),
    )
    .toBe('collapsed')
  const link = page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Models', exact: true })
    .element()
  const icon = link.querySelector('svg')!
  expect(icon.getBoundingClientRect().right).toBeLessThanOrEqual(
    link.getBoundingClientRect().right -
      parseFloat(getComputedStyle(link).paddingRight) +
      0.5,
  )
  expect(getComputedStyle(link, '::before').backgroundColor).toBe(
    'rgb(232, 90, 90)',
  )
})

test.each(['engine', 'model'])(
  'the %s Cancel stays above an expanded file list',
  async (kind) => {
    const view = fixture('models-download-in-progress')
    const plan = recordedPlan()
    view.models!.download!.plan = plan
    view.models!.download!.overall = { bytes_done: 3, bytes_total: 12 }
    view.phase = 'preparing_engine'
    view.preparation = [{ ...plan, kind: 'runtime' }]
    view.approval = undefined
    cleanup = renderView(
      kind === 'engine' ? (
        <Preparation view={view} />
      ) : (
        <ModelDownloadProgress view={view} retry={() => {}} />
      ),
      view,
    ).cleanup
    await page.getByText('Details', { exact: true }).click()
    const list = page
      .getByRole('list', { name: 'Download file progress' })
      .element()
    const cancel = page
      .getByRole('button', { name: 'Cancel download' })
      .element()
    expect(cancel.getBoundingClientRect().bottom).toBeLessThanOrEqual(
      list.getBoundingClientRect().top,
    )
  },
)
