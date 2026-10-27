import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions, subscribeViewEvents } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import LogsPage from './index'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('external logs explain ownership and copy the CLI status path exactly', async () => {
  const view = fixture('external-server')
  view.instance!.log_path = undefined
  view.log_path = 'D:/fixture/server.log'
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<LogsPage />, view).cleanup
  await expect
    .element(
      page.getByText('This server was started outside Bubo.', { exact: false }),
    )
    .toBeVisible()
  await page.getByRole('button', { name: 'Copy log path' }).click()
  expect(copy).toHaveBeenCalledExactlyOnceWith(view.log_path)
})

test('filters and copies shown lines and receives log events through the View cache', async () => {
  const view = fixture()
  view.logs = ['fixture ready', 'fixture Serve: done']
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const mounted = renderView(<LogsPage />, view)
  const handlers = new Map<string, (payload: unknown) => void>()
  const stop = subscribeViewEvents(mounted.client, (name, handler) => {
    handlers.set(name, handler)
    return () => handlers.delete(name)
  })
  cleanup = () => {
    stop()
    mounted.cleanup()
  }
  await page.getByRole('textbox', { name: 'Filter lines' }).fill('SERVE')
  await expect
    .element(page.getByLabelText('Log lines'))
    .toHaveTextContent('fixture Serve: done')
  await page.getByRole('button', { name: 'Copy shown lines' }).click()
  expect(clipboard).toHaveBeenLastCalledWith('fixture Serve: done')
  handlers.get('log:line')?.('fixture serve: new event')
  await expect
    .element(page.getByLabelText('Log lines'))
    .toHaveTextContent('fixture serve: new event')
  await page.getByRole('button', { name: 'Clear filter' }).click()
  await expect
    .element(page.getByLabelText('Log lines'))
    .toHaveTextContent('fixture ready')
  await expect.element(page.getByText('3 shown of 3 lines')).toBeVisible()
})

test('empty or unmatched logs never copy empty-state prose', async () => {
  const view = fixture()
  view.logs = ['fixture line']
  cleanup = renderView(<LogsPage />, view).cleanup
  await page.getByRole('textbox', { name: 'Filter lines' }).fill('no match')
  await expect
    .element(page.getByLabelText('Log lines'))
    .toHaveTextContent('No lines match the filter.')
  await expect
    .element(page.getByRole('button', { name: 'Copy shown lines' }))
    .toBeDisabled()
})

test('diagnostics preview and local support bundle use typed actions and preserve errors', async () => {
  const view = fixture()
  view.bundle_preview = 'fixture support files, redacted by supervisor'
  const diagnose = vi
    .spyOn(actions, 'Diagnose')
    .mockResolvedValue({} as Awaited<ReturnType<typeof actions.Diagnose>>)
  const preview = vi
    .spyOn(actions, 'PreviewBundle')
    .mockResolvedValue({} as Awaited<ReturnType<typeof actions.PreviewBundle>>)
  const save = vi.spyOn(actions, 'SaveBundle').mockResolvedValue({
    error: {
      code: 'bundle_write_failed',
      remedy: 'Choose another local path.',
      component: '',
      stage: '',
      retryable: true,
    },
  } as Awaited<ReturnType<typeof actions.SaveBundle>>)
  cleanup = renderView(<LogsPage />, view).cleanup
  await page.getByRole('button', { name: 'Run diagnostics' }).click()
  await page.getByRole('button', { name: 'Preview support bundle' }).click()
  expect(diagnose).toHaveBeenCalledTimes(1)
  expect(preview).toHaveBeenCalledTimes(1)
  await page
    .getByRole('textbox', { name: 'Local ZIP path' })
    .fill('C:\\fixture-support.zip')
  await page
    .getByRole('button', { name: 'Save support bundle locally' })
    .click()
  expect(save).toHaveBeenCalledWith('C:\\fixture-support.zip')
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('bundle_write_failed: Choose another local path.')
  await expect
    .element(page.getByText('Support bundle saved locally.'))
    .not.toBeInTheDocument()
})

test.each(['Ready', 'Stopped', 'Attached-external'] as const)(
  'the CLI log path remains visible for %s with no lines',
  async (state) => {
    const view = fixture()
    view.state = state
    view.managed = false
    view.logs = []
    view.log_path = 'D:/fixture/owned.log'
    view.instance = undefined
    const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
    cleanup = renderView(<LogsPage />, view).cleanup
    await expect
      .element(page.getByText(view.log_path, { exact: true }))
      .toBeVisible()
    await page.getByRole('button', { name: 'Copy log path' }).click()
    expect(copy).toHaveBeenCalledExactlyOnceWith(view.log_path)
    if (state !== 'Attached-external')
      await expect
        .element(
          page.getByText('This server was started outside Bubo.', {
            exact: false,
          }),
        )
        .not.toBeInTheDocument()
  },
)
