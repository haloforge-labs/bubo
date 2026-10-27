import { describe, expect, test, vi } from 'vitest'
import type { actions } from '@/lib/bindings'
import {
  openModelPlan,
  confirmModelPlan,
  cancelModelPlan,
} from '@/components/model-download-flow'
import { bridgeReply } from '@/components/bridge-test-reply'
import recorded from '@/lib/model-plan-test-data'

// Port of legacy j20-flow.test.js. Structured wrappers delegate to the
// same StartModelDownload / ConfirmModelDownload / CancelModelDownload Go calls.
function fakeApi() {
  const plan = structuredClone(recorded.plan) as unknown as NonNullable<
    Awaited<ReturnType<typeof actions.CheckModel>>['plan']
  >
  return {
    plan,
    api: {
      Download: vi.fn<typeof actions.Download>(async () =>
        bridgeReply({ plan }),
      ),
      CheckModel: vi.fn<typeof actions.CheckModel>(async () =>
        bridgeReply({ plan: { ...plan, kind: 'repair' } }),
      ),
      ConfirmDownload: vi.fn<typeof actions.ConfirmDownload>(async () =>
        bridgeReply({}),
      ),
      CancelDownload: vi.fn<typeof actions.CancelDownload>(async () =>
        bridgeReply({}),
      ),
    },
  }
}

describe('reviewed download binding order (J20)', () => {
  test('opening reads one plan and downloads nothing', async () => {
    const { api, plan } = fakeApi()
    expect(await openModelPlan(api, plan.id, 'download')).toEqual(plan)
    expect(api.Download).toHaveBeenCalledExactlyOnceWith(plan.id)
    expect(api.ConfirmDownload).not.toHaveBeenCalled()
    expect(api.CancelDownload).not.toHaveBeenCalled()
    expect(api.CheckModel).not.toHaveBeenCalled()
  })
  test('confirm never rereads the plan', async () => {
    const { api, plan } = fakeApi()
    await openModelPlan(api, plan.id, 'download')
    await confirmModelPlan(api, plan.id)
    expect(api.Download).toHaveBeenCalledTimes(1)
    expect(api.ConfirmDownload).toHaveBeenCalledExactlyOnceWith(plan.id)
    expect(api.Download.mock.invocationCallOrder[0]).toBeLessThan(
      api.ConfirmDownload.mock.invocationCallOrder[0]!,
    )
  })
  test('Not now drops the pending plan', async () => {
    const { api, plan } = fakeApi()
    await openModelPlan(api, plan.id, 'download')
    await cancelModelPlan(api)
    expect(api.CancelDownload).toHaveBeenCalledExactlyOnceWith()
    expect(api.ConfirmDownload).not.toHaveBeenCalled()
  })
  test('repair and offline check keep their reviewed semantics', async () => {
    const { api, plan } = fakeApi()
    expect((await openModelPlan(api, plan.id, 'repair')).kind).toBe('repair')
    expect(api.CheckModel).toHaveBeenCalledExactlyOnceWith(plan.id, true)
    await confirmModelPlan(api, plan.id)
    await api.CheckModel(plan.id, false)
    expect(api.CheckModel).toHaveBeenLastCalledWith(plan.id, false)
    expect(api.Download).not.toHaveBeenCalled()
  })
  test('only catalogue ID is forwarded; no folder or acceptance flag', async () => {
    const { api, plan } = fakeApi()
    await openModelPlan(api, plan.id, 'download')
    await confirmModelPlan(api, plan.id)
    expect(api.Download.mock.calls).toEqual([[plan.id]])
    expect(api.ConfirmDownload.mock.calls).toEqual([[plan.id]])
  })
  test('typed CLI failure and rejected bridge never report success', async () => {
    const { api, plan } = fakeApi()
    const error = { code: 'unsupported', remedy: 'Fixture plan unavailable.' }
    await expect(
      openModelPlan(
        { ...api, Download: async () => bridgeReply({ error }) },
        plan.id,
        'download',
      ),
    ).rejects.toEqual(error)
    await expect(
      openModelPlan(
        { ...api, Download: async () => bridgeReply({}) },
        plan.id,
        'download',
      ),
    ).rejects.toMatchObject({ code: 'protocol_mismatch' })
    await expect(
      confirmModelPlan(
        { ...api, ConfirmDownload: async () => bridgeReply({ error }) },
        plan.id,
      ),
    ).rejects.toEqual(error)
    api.Download.mockRejectedValueOnce(
      new Error('A model action is already running.'),
    )
    await expect(openModelPlan(api, plan.id, 'download')).rejects.toThrow(
      'already running',
    )
  })
})
