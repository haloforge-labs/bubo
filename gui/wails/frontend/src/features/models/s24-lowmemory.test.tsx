import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { bytes } from '@/components/model-facts'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import recordedReply from '@/generated/memory96-estimate-reply.json'

import { lowerMemoryPlanFixture as lowerMemoryPlan } from '@/components/core-page-test-support'

const id = 'huihui-qwen38-flash-next-lowmemory-v1'

afterEach(() => vi.restoreAllMocks())

for (const split of ['96-64', '96-72', '128-96']) {
  test(`published lower-memory model has the real size and calibrated estimate on ${split}`, async () => {
    const view = coreView(`onboarding-${split}`)
    const entry = view.models!.catalogue.find((row) => row.id === id)!
    expect(entry.display_name).toBe(
      'Qwen3.8 Flash Next · HaloForge Labs lower-memory quant (GSQ-RCO)',
    )
    expect(entry.download_enabled).toBe(true)
    expect(entry.disabled_reason).toBeNull()
    expect(entry.estimate.estimate_state).toBe('estimated')
    expect(entry.estimate.memory_estimate).not.toBeNull()
    mountCorePage('/models', view)
    const row = page.getByRole('row').filter({ hasText: entry.display_name })
    await expect.element(row).toHaveTextContent(bytes(entry.download_bytes))
    expect(row.element().textContent).not.toMatch(/tokens\/s|quality/)
    await expect
      .element(row.getByRole('button', { name: 'Download', exact: true }))
      .toBeEnabled()
  })
}

test('an installed lower-memory package is selectable and renders its CLI estimate', async () => {
  const view = coreView('onboarding-128-96')
  const entry = view.models!.catalogue.find((row) => row.id === id)!
  // Arrange a completed installation from the real CLI catalogue and package
  // metadata. This tests the UI, without claiming an on-disk installation.
  const folder = entry.local_folder
  const local = {
    id: entry.id,
    display_name: entry.display_name,
    manifest_path: `${folder}/bubo-package.json`,
    model_path: `${folder}/${entry.files.find((file) => file.role === 'main')!.name}`,
    size_bytes: entry.download_bytes,
    prepared_state: 'prepared',
    capabilities: {
      context_max: recordedReply.estimate.fit_inputs.native_context_tokens,
      mtp: entry.files.some((file) => file.role === 'mtp'),
      twin: entry.files.some((file) => file.role === 'twin'),
      mmproj: entry.files.some((file) => file.role === 'mmproj'),
    },
    repair_output: undefined,
  }
  view.models!.prepared_models = [local]
  view.prepared_models = [local]
  view.models!.scanned = []
  expect(recordedReply.estimate.memory_estimate).not.toBeNull()
  view.current_model = undefined
  const read = vi
    .spyOn(actions, 'Estimate')
    .mockResolvedValue(bridgeReply(recordedReply))
  const select = vi
    .spyOn(actions, 'PatchPreferences')
    .mockResolvedValue(bridgeReply({}))
  mountCorePage('/models', view)
  const installed = page.getByRole('region', { name: 'Installed models' })
  const row = installed.getByRole('row').filter({ hasText: entry.display_name })
  await expect.element(row).toHaveTextContent(bytes(entry.download_bytes))
  await expect.element(row).toHaveTextContent(recordedReply.fit.sentence)
  const button = row.getByRole('button', { name: 'Select for Server' })
  await expect.element(button).toBeEnabled()
  expect(read).toHaveBeenCalledExactlyOnceWith(folder, 0, 0)
  await button.click()
  expect(select).toHaveBeenCalledExactlyOnceWith({
    set: { manifest_path: local.manifest_path },
    clear: ['model_path'],
  })
})

test('published lower-memory Download opens the recorded plan and waits for consent', async () => {
  const view = coreView('stopped')
  const entry = view.models!.catalogue.find((row) => row.id === id)!
  const download = vi.spyOn(actions, 'Download').mockResolvedValue(
    bridgeReply({
      plan: lowerMemoryPlan.plan as unknown as NonNullable<
        Awaited<ReturnType<typeof actions.Download>>['plan']
      >,
    }),
  )
  const confirm = vi.spyOn(actions, 'ConfirmDownload')
  mountCorePage('/models', view)
  const row = page.getByRole('row').filter({ hasText: entry.display_name })
  expect(entry.disabled_reason).toBeNull()
  await expect
    .element(row.getByRole('button', { name: 'Download', exact: true }))
    .toBeEnabled()
  await row.getByRole('button', { name: 'Download', exact: true }).click()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  await expect
    .element(row.getByRole('button', { name: 'Try again' }))
    .not.toBeInTheDocument()
  await expect
    .element(row.getByRole('button', { name: 'Download the engine files' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('dialog'))
    .toHaveTextContent(entry.licence.name)
  await expect
    .element(
      page.getByRole('dialog').getByRole('link', { name: entry.licence.name }),
    )
    .toHaveAttribute('href', entry.licence.url)
  expect(download).toHaveBeenCalledExactlyOnceWith(entry.id)
  expect(confirm).not.toHaveBeenCalled()
})
