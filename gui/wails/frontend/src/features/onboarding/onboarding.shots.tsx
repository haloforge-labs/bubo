import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { bytes } from '@/lib/format'
import {
  coreView,
  mountCorePage,
  recordedRuntimePlan,
} from '@/components/core-page-test-support'

// 'stopped' has the default package, so it opens on the Server page; the
// recorded Stopped state with no model is runtime-ready-model-missing.
for (const name of [
  'not-prepared',
  'runtime-ready-model-missing',
  'preparing',
]) {
  test(`Onboarding: ${name}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    mountCorePage('/', coreView(name))
    await expect
      .element(page.getByRole('heading', { name: 'Get started', exact: true }))
      .toBeVisible()
    await document.fonts.ready
    expect(await commands.captureFeature('onboarding', name)).toMatch(/\.png$/)
  })
}

test('Onboarding: synthetic preparation approval over recorded plan', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('preparing')
  const runtime = recordedRuntimePlan()
  view.preparation = [runtime]
  view.phase = 'needs_engine'
  view.status_label = 'Setup needed: download the engine files'
  view.runtime.readiness.runtime = false
  view.models!.catalogue = []
  view.progress = undefined
  view.preparation_file = undefined
  view.approval = {
    components: ['runtime'],
    plan_digest: runtime.plan_digest,
  }
  mountCorePage('/', view)
  await expect
    .element(
      page.getByRole('dialog', {
        name: 'Download the engine files',
      }),
    )
    .toBeVisible()
  await document.fonts.ready
  expect(
    await commands.captureFeature('onboarding', 'synthetic-prepare-approval'),
  ).toMatch(/\.png$/)
})

test('V12 first-run choices show CLI download sizes and default-context GPU needs', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('runtime-ready-model-missing')
  const choices = view.models!.catalogue.filter(
    (entry) => entry.architecture === 'qwen4exp',
  )
  mountCorePage('/onboarding', view)
  const select = page.getByRole('combobox', { name: 'Model to download' })
  await expect
    .element(select)
    .toHaveValue(
      choices.find((entry) => entry.default_status === 'default')!.id,
    )
  const facts = page.getByRole('list', { name: 'Download choices' })
  expect(facts.getByRole('listitem').elements()).toHaveLength(3)
  for (const entry of choices) {
    const label = entry.selection_label || entry.display_name
    const row = facts.getByRole('listitem').filter({ hasText: label })
    await expect
      .element(row)
      .toHaveTextContent(`Download ${bytes(entry.download_bytes)}`)
    const memory = entry.estimate.memory_estimate
    if (memory?.gpu) {
      await expect
        .element(row)
        .toHaveTextContent(
          `Needs about ${bytes(memory.gpu.need.bytes, { unit: 'GiB', digits: 0 })} of GPU memory at ${memory.context_tokens.toLocaleString()} tokens`,
        )
    } else {
      await expect.element(row).not.toHaveTextContent('GPU memory')
    }
    if (!entry.download_enabled) {
      await expect
        .element(
          select.getByRole('option', {
            name: `${label} — ${entry.disabled_reason}`,
            exact: true,
          }),
        )
        .toBeDisabled()
      await expect.element(row).toHaveTextContent(entry.disabled_reason!)
    }
  }
  await document.fonts.ready
  expect(
    await commands.captureFeature('onboarding', 'v12-first-run-choices'),
  ).toMatch(/\.png$/)
})
