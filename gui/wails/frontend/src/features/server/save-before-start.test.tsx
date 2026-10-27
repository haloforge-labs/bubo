import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { bridgeReply } from '@/components/bridge-test-reply'
import { actions } from '@/lib/bindings'

afterEach(() => vi.restoreAllMocks())

for (const running of [false, true]) {
  test(`${running ? 'Restart' : 'Start'} saves the displayed setup before launching`, async () => {
    const view = coreView(running ? 'ready-idle' : 'stopped-saved-estimate')
    const calls: string[] = []
    const patch = vi
      .spyOn(actions, 'PatchPreferences')
      .mockImplementation(async () => {
        calls.push('save')
        return {}
      })
    const launch = vi
      .spyOn(actions, running ? 'Restart' : 'Start')
      .mockImplementation(async () => {
        calls.push('launch')
        return {}
      })
    vi.spyOn(actions, 'Estimate').mockResolvedValue(
      bridgeReply({
        estimate: view.saved_estimate,
        fit: { ...view.fit, class: 'fits' },
      }),
    )
    mountCorePage('/', view)
    await page.getByLabelText('Context per session').fill('65536')
    const button = page.getByRole('button', {
      name: running ? 'Restart server' : 'Start server',
      exact: true,
    })
    await expect.element(button).toBeEnabled()
    await button.click()
    await expect.poll(() => calls).toEqual(['save', 'launch'])
    expect(patch.mock.calls[0]?.[0].set.context).toBe(65536)
    expect(launch).toHaveBeenCalledTimes(1)
    expect(document.body.textContent).not.toContain(
      'Save server setup before starting or restarting.',
    )
  })
}

test('a rejected save never starts the server', async () => {
  const view = coreView('stopped-saved-estimate')
  vi.spyOn(actions, 'PatchPreferences').mockRejectedValue({
    code: 'storage_failed',
    remedy: 'Cannot save.',
  })
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const start = vi.spyOn(actions, 'Start').mockResolvedValue({})
  const { replaceView } = mountCorePage('/', view)
  await page.getByLabelText('Context per session').fill('65536')
  await page.getByRole('button', { name: 'Start server', exact: true }).click()
  await expect.poll(() => document.body.textContent).toContain('Cannot save.')
  expect(start).not.toHaveBeenCalled()
  replaceView({
    ...view,
    generation: view.generation + 1,
    status_feed: {
      ...view.status_feed,
      state: 'updating',
      sample_time: '2099-01-01T00:00:00Z',
    },
  })
  await expect
    .poll(() => document.body.textContent)
    .not.toContain('Cannot save.')
})

test('a fresh healthy status keeps an in-flight save and Start pending', async () => {
  const view = coreView('stopped-saved-estimate')
  let resolveSave!: (value: object) => void
  const saved = new Promise<object>((resolve) => {
    resolveSave = resolve
  })
  const patch = vi.spyOn(actions, 'PatchPreferences').mockReturnValue(saved)
  const start = vi.spyOn(actions, 'Start').mockResolvedValue({})
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const { replaceView } = mountCorePage('/', view)
  await page.getByLabelText('Context per session').fill('65536')
  const button = page.getByRole('button', { name: 'Start server', exact: true })
  await button.click()
  await expect.poll(() => patch.mock.calls.length).toBe(1)
  await expect.element(button).toBeDisabled()
  replaceView({
    ...view,
    status_label: 'Fresh stopped status',
    generation: view.generation + 1,
    status_feed: {
      ...view.status_feed,
      state: 'updating',
      sample_time: '2099-01-01T00:00:00Z',
    },
  })
  await expect
    .element(page.getByText('Fresh stopped status', { exact: true }))
    .toBeVisible()
  // Let the status render and its effects settle while the save is still unresolved.
  await new Promise((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(resolve)),
  )
  await expect
    .poll(() => button.element().hasAttribute('disabled'), { timeout: 1000 })
    .toBe(true)
  expect(start).not.toHaveBeenCalled()
  resolveSave({})
  await expect.poll(() => start.mock.calls.length).toBe(1)
  expect(patch).toHaveBeenCalledTimes(1)
})

test('Start saves a changed Tight setup and requests fresh consent before launch', async () => {
  const view = coreView('memory-gpu-headroom')
  view.saved.allow_tight_setup = true
  view.actions.start = { allowed: true, reason_code: '', reason: '' }
  const { replaceView } = mountCorePage('/', view)
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({ estimate: view.saved_estimate, fit: view.fit }),
  )
  const patch = vi
    .spyOn(actions, 'PatchPreferences')
    .mockImplementation(async () => {
      replaceView({
        ...view,
        saved: { ...view.saved, context: 65536, allow_tight_setup: false },
        actions: {
          ...view.actions,
          start: {
            allowed: false,
            reason_code: 'memory_tight',
            reason: 'This setup is tight.',
          },
        },
      })
      return {}
    })
  const start = vi.spyOn(actions, 'Start').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'memory_tight',
        component: 'supervisor',
        stage: 'start',
        retryable: false,
        remedy: 'This setup is tight.',
      },
    }),
  )
  await page.getByLabelText('Context per session').fill('65536')
  await page.getByRole('button', { name: 'Start server', exact: true }).click()
  await expect.poll(() => start.mock.calls.length).toBe(1)
  expect(patch).toHaveBeenCalledTimes(1)
  expect(patch.mock.calls[0]?.[0].set.context).toBe(65536)
  const consent = page.getByRole('checkbox', { name: /^Override:/ })
  await expect.element(consent).not.toBeChecked()
  await expect.element(consent).toBeEnabled()
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeDisabled()
  await expect
    .element(page.getByRole('heading', { name: 'Tight', exact: true }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Blocked: memory is tight.')
  expect(document.querySelectorAll('[role="alert"]')).toHaveLength(0)
})
