import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('setup comes before running telemetry and Server draws no charts', async () => {
  const view = coreView('ready-busy')
  view.gpu = {
    use_pct: 73,
    memory_used_bytes: 70 * 1024 ** 3,
    memory_total_bytes: 96 * 1024 ** 3,
    temperature_c: 61,
    chip_power_w: 83,
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Server setup', exact: true }))
    .toBeVisible()
  expect(
    document
      .querySelector('[aria-label="Server setup"]')!
      .compareDocumentPosition(
        document.querySelector('[aria-label="Reported request rates"]')!,
      ) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
  await expect
    .element(page.getByText(/Recommended context for/))
    .not.toBeInTheDocument()
  // One completed request is the figure above; a chart needs two.
  await expect
    .element(page.getByRole('region', { name: 'Recent performance' }))
    .not.toBeInTheDocument()
  expect(document.querySelector('.recharts-wrapper')).toBeNull()
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('region', { name: 'GPU', exact: true }))
    .toHaveTextContent('70.0 GiB / 96.0 GiB')
})

test('stopped views have no empty speed or GPU chart frames', async () => {
  const view = coreView()
  view.state = 'Stopped'
  const { replaceView } = mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Reported request rates' }))
    .not.toBeInTheDocument()
  expect(document.querySelector('.recharts-wrapper')).toBeNull()
  replaceView(coreView('failed-with-remedy'))
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .not.toBeInTheDocument()
  // A failure owns its fix: its cause, one button and Details, in one box
  // (ORACLE S15). The status row offers only Start, which is not allowed.
  const box = page.getByRole('alert')
  await expect
    .element(box)
    .toHaveTextContent("Bubo's engine files are missing.")
  await expect
    .element(
      box.getByRole('button', {
        name: 'Download the engine files',
        exact: true,
      }),
    )
    .toBeVisible()
  // One fix button on the page, and it is in the failure box, not the status row.
  expect(
    page
      .getByRole('button', { name: 'Download the engine files', exact: true })
      .elements(),
  ).toHaveLength(1)
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByRole('region', { name: 'GPU', exact: true }))
    .not.toBeInTheDocument()
})

test('context is grouped when idle, names its unit once, and saves the exact numeric edit', async () => {
  const view = coreView()
  view.saved.manifest_path = view.prepared_models[0]!.manifest_path
  const context = Number(
    view.settings.find((row) => row.name === '--context')!.effective.text,
  )
  const editedContext =
    view.memory_estimate!.recommended_context!.minimum_context_tokens
  const save = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: { memory_estimate: { ...view.memory_estimate!, fits: true } },
    }),
  )
  mountCorePage('/', view)
  const field = page.getByRole('spinbutton', { name: 'Context per session' })
  await expect.element(field).toHaveValue(context.toLocaleString())
  expect(document.getElementById('context-unit')).toBeNull()
  await field.fill(String(editedContext))
  await page.getByRole('button', { name: 'Save server setup' }).click()
  await expect.poll(() => save.mock.calls.length).toBe(1)
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { context: editedContext },
    clear: [],
  })
})
