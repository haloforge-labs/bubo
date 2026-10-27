import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { GUIError } from './gui-error'
import { fixture, renderView } from './test-view'
import { actions } from '@/lib/bindings'
import { bridgeReply } from './bridge-test-reply'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('runtime failure has one GUI recovery; raw code and command appear only in Details', async () => {
  const prepare = vi
    .spyOn(actions, 'Prepare')
    .mockResolvedValue(bridgeReply({}))
  cleanup = renderView(
    <GUIError
      error={{ code: 'runtime_missing', remedy: 'Run bubo prepare runtime.' }}
    />,
    fixture(),
  ).cleanup
  await expect
    .element(
      page.getByText('runtime_missing: Run bubo prepare runtime.', {
        exact: true,
      }),
    )
    .not.toBeVisible()
  await page
    .getByRole('button', { name: 'Download the engine files', exact: true })
    .click()
  expect(prepare).toHaveBeenCalledExactlyOnceWith()
  await page.getByText('Details', { exact: true }).click()
  await expect
    .element(
      page.getByText('runtime_missing: Run bubo prepare runtime.', {
        exact: true,
      }),
    )
    .toBeVisible()
})

test('unknown errors preserve their exact code and remedy in Details', async () => {
  cleanup = renderView(
    <GUIError error={{ code: 'future_failure', remedy: 'Future remedy.' }} />,
    fixture(),
  ).cleanup
  await page.getByText('Details', { exact: true }).click()
  await expect
    .element(page.getByText('future_failure: Future remedy.', { exact: true }))
    .toBeVisible()
})

test('the CLI detail (the cause) appears only inside Details', async () => {
  cleanup = renderView(
    <GUIError
      error={{
        code: 'device_lost',
        remedy: 'Restart on request.',
        detail: 'HIP error 700 at decode_attn',
      }}
    />,
    fixture(),
  ).cleanup
  await expect
    .element(page.getByText('The GPU stopped responding.'))
    .toBeVisible()
  const detail = page.getByText('HIP error 700 at decode_attn')
  await expect.element(detail).toBeInTheDocument()
  expect(detail.element().closest('details')).not.toBeNull()
})

test('a start the memory guard refused says what did not fit; the remedy sits below and the detail in Details', async () => {
  cleanup = renderView(
    <GUIError
      error={{
        code: 'memory_insufficient',
        remedy: 'Lower --context or --sessions, or free memory.',
        detail: '2 sessions at 65536 tokens need 47.3 GiB.',
        memory: {
          needed_bytes: 50784518144,
          available_bytes: 33503973376,
          sessions: 2,
          context: 65536,
        },
      }}
    />,
    fixture(),
  ).cleanup
  await expect
    .element(
      page.getByText(
        "The server didn't start: needs 47 GiB for 2 sessions at 65,536 tokens; 31 GiB is available.",
        { exact: true },
      ),
    )
    .toBeVisible()
  // The GUI says it with the form's own words; the CLI's text is in Details.
  const remedy = page.getByText(
    'Lower Context per session or Sessions, or close other apps.',
    { exact: true },
  )
  await expect.element(remedy).toBeVisible()
  expect(remedy.element().closest('details')).toBeNull()
  const cli = page.getByText(
    'memory_insufficient: Lower --context or --sessions, or free memory.',
    { exact: true },
  )
  expect(cli.element().closest('details')).not.toBeNull()
  const detail = page.getByText('2 sessions at 65536 tokens need 47.3 GiB.')
  expect(detail.element().closest('details')).not.toBeNull()
})
