import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

// The design review's rules, as the Server page shows them.

test('a recovery link that names an Inference setting opens the section and shows the row', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.inference_expanded = false
  // The memory history takes a hash; the helper's type names only the routes.
  mountCorePage('/#setting:--top-k' as '/', view)
  const row = page.getByRole('group', { name: 'Top-k', exact: true })
  await expect.element(row).toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Inference', exact: true }))
    .toHaveAttribute('aria-expanded', 'true')
})

test('Reset model, context and sessions shows only while one of the four is saved', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.model_path = undefined
  view.saved.manifest_path = undefined
  view.saved.context = undefined
  view.saved.sessions = undefined
  const clean = mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Save server setup' }))
    .toBeVisible()
  expect(
    page
      .getByRole('button', { name: 'Reset model, context and sessions' })
      .query(),
  ).toBeNull()
  clean.replaceView({ ...view, saved: { ...view.saved, sessions: 2 } })
  await expect
    .element(
      page.getByRole('button', { name: 'Reset model, context and sessions' }),
    )
    .toBeVisible()
})

test('Context and Sessions are rows with an info popup and the unit', async () => {
  mountCorePage('/', coreView('stopped-saved-estimate'))
  await expect
    .element(page.getByRole('button', { name: 'About Context per session' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'About Sessions' }))
    .toBeVisible()
  await expect
    .element(
      page
        .getByRole('group', { name: 'Context per session', exact: true })
        .getByText('tokens', { exact: true }),
    )
    .toBeVisible()
})

test('a fit says it once; the numbers carry their real names', async () => {
  mountCorePage('/', coreView('stopped-saved-estimate'))
  await expect
    .element(page.getByRole('heading', { name: 'Fits', exact: true }))
    .toBeVisible()
  // No second "Fits. 12 GiB of memory would be left." under the heading.
  expect(document.body.textContent).not.toMatch(/Fits\.\s+\d+ GiB of memory/)
  await expect
    .element(page.getByText('GPU memory left', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('System memory left', { exact: true }))
    .toBeVisible()
  await expect
    .element(page.getByText('Available system memory', { exact: true }))
    .toBeVisible()
})

test('while the server runs, its facts come before the setup', async () => {
  mountCorePage('/', coreView('ready-busy'))
  await expect
    .element(page.getByRole('region', { name: 'Server setup', exact: true }))
    .toBeVisible()
  const setup = document.querySelector('[aria-label="Server setup"]')!
  const model = document.querySelector('[aria-label="Model in use"]')!
  expect(
    model.compareDocumentPosition(setup) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
})

test('at the default window the context hint stays on one line and the choices share one control width', async () => {
  await page.viewport(1280, 829)
  const view = coreView('ready-busy')
  view.saved.inference_expanded = true
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('group', { name: 'Context per session' }))
    .toBeVisible()
  await new Promise((resolve) => setTimeout(resolve, 400))
  const hint = [
    ...document.querySelectorAll('[data-setting="context"] span'),
  ].find((node) => /max/i.test(node.textContent ?? ''))
  expect(hint, 'the context hint').toBeTruthy()
  const lines = Math.round(
    hint!.getBoundingClientRect().height /
      parseFloat(getComputedStyle(hint!).lineHeight),
  )
  expect(lines).toBe(1)
  const width = (name: string) =>
    document
      .querySelector(`[data-setting="${name}"] select`)!
      .getBoundingClientRect().width
  // The two setup choices share one width; Prefill mode sits in the three-column
  // Inference grid and has the width of that cell.
  expect(width('--kv-cache-type')).toBe(width('--refusal-direction'))
  // The shared width shows the longest option whole: its text, the padding and
  // the arrow.
  const kv = document.querySelector<HTMLSelectElement>(
    '[data-setting="--kv-cache-type"] select',
  )!
  const style = getComputedStyle(kv)
  const context = document.createElement('canvas').getContext('2d')!
  context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`
  const longest = Math.max(
    ...[...kv.options].map((option) => context.measureText(option.text).width),
  )
  expect(kv.getBoundingClientRect().width).toBeGreaterThanOrEqual(
    longest + 2 * parseFloat(style.paddingLeft) + 20,
  )
})
