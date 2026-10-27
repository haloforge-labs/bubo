import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('asynchronous native shutdown errors remain visible from the View notice', async () => {
  const view = coreView()
  const detail = 'Exit failed: Fixture native shutdown error.'
  view.notice = { code: 'internal_error', detail }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('alert').filter({ hasText: detail }))
    .toHaveTextContent(detail)
})

for (const [request, binding] of [
  ['show', 'SetVisible'],
  ['start', 'Start'],
  ['stop', 'Stop'],
  ['restart', 'Restart'],
  ['quit', 'Exit'],
] as const) {
  test(`tray consumes and dispatches ${request} once across polling`, async () => {
    const take = vi
      .spyOn(actions, 'TakeTrayAction')
      .mockResolvedValue({ value: request })
    const dispatch =
      binding === 'SetVisible'
        ? vi.spyOn(actions, binding).mockResolvedValue({})
        : vi.spyOn(actions, binding).mockResolvedValue(bridgeReply({}))
    const view = coreView()
    view.window.tray_action = request
    const { replaceView } = mountCorePage('/', view)
    await expect.poll(() => dispatch.mock.calls.length).toBe(1)
    expect(take).toHaveBeenCalledOnce()
    if (request === 'show') expect(dispatch).toHaveBeenCalledWith(true)
    else expect(dispatch).toHaveBeenCalledWith()
    replaceView(structuredClone(view))
    await expect
      .element(page.getByRole('heading', { name: 'Server', exact: true }))
      .toBeVisible()
    expect(take).toHaveBeenCalledOnce()
    expect(dispatch).toHaveBeenCalledOnce()
  })
}

test('consumed stale tray request dispatches nothing; unknown request is an internal error', async () => {
  const take = vi
    .spyOn(actions, 'TakeTrayAction')
    .mockResolvedValue({ value: '' })
  const stop = vi.spyOn(actions, 'Stop').mockResolvedValue(bridgeReply({}))
  const view = coreView()
  view.window.tray_action = 'stop'
  const { replaceView } = mountCorePage('/', view)
  await expect.poll(() => take.mock.calls.length).toBe(1)
  expect(stop).not.toHaveBeenCalled()
  take.mockResolvedValue({ value: 'future-action' })
  replaceView({
    ...view,
    window: { ...view.window, tray_action: 'future-action' },
  })
  const detail = 'internal_error: Unknown tray action: future-action'
  await expect
    .element(page.getByRole('alert').filter({ hasText: detail }))
    .toHaveTextContent(detail)
})

test('typed tray action failures stay visible', async () => {
  vi.spyOn(actions, 'TakeTrayAction').mockResolvedValue({ value: 'restart' })
  vi.spyOn(actions, 'Restart').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'fixture_rejected',
        remedy: 'Fixture restart remedy.',
        retryable: false,
        component: '',
        stage: '',
      },
    }),
  )
  const view = coreView()
  view.window.tray_action = 'restart'
  mountCorePage('/', view)
  const detail = 'fixture_rejected: Fixture restart remedy.'
  await expect
    .element(page.getByRole('alert').filter({ hasText: detail }))
    .toHaveTextContent(detail)
})
