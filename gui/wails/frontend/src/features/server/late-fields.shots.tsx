import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

for (const name of [
  'npu-prefill-on',
  'npu-prefill-off',
  'npu-prefill-faulted',
  'unavailable',
]) {
  test(`Settings runtime ${name}`, async () => {
    document.documentElement.dataset.theme = inject('theme')
    const view = coreView(name.startsWith('npu-') ? name : 'npu-prefill-on')
    if (name === 'unavailable') view.npu_prefill_runtime = undefined
    mountCorePage('/settings', view)
    await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
    const hardware = page.getByRole('region', { name: 'NPU prefill status' })
    if (name === 'unavailable' || name === 'npu-prefill-off') {
      // An unavailable or switched-off NPU draws nothing at all.
      await expect
        .element(page.getByRole('region', { name: 'Hardware', exact: true }))
        .toBeInTheDocument()
      await expect.element(hardware).not.toBeInTheDocument()
      const shell = document.getElementById('shell-capture')!
      shell.style.height = '760px'
      shell.style.overflow = 'auto'
      page
        .getByRole('region', { name: 'Hardware', exact: true })
        .element()
        .scrollIntoView({ block: 'end' })
    } else {
      await expect.element(hardware).toBeInTheDocument()
      const shell = document.getElementById('shell-capture')!
      shell.style.height = '760px'
      shell.style.overflow = 'auto'
      // Establish the capture viewport before scrolling so Debug is inside the PNG.
      hardware.element().scrollIntoView({ block: 'start' })
    }
    await document.fonts.ready
    await commands.captureFeature('settings', name)
  })
}

test('Server ready-idle with the NPU off shows no NPU line', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView('ready-idle')
  view.npu_prefill_runtime = coreView('npu-prefill-off').npu_prefill_runtime
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('region', { name: 'Server setup' }))
    .toBeInTheDocument()
  await expect
    .element(page.getByRole('region', { name: 'NPU prefill status' }))
    .not.toBeInTheDocument()
  expect(document.body.textContent).not.toContain('NPU prefill:')
  await document.fonts.ready
  await commands.captureFeature('server', 'ready-idle-npu-off')
})

test('Server rejected override recovery', async () => {
  document.documentElement.dataset.theme = inject('theme')
  mountCorePage('/', coreView('switch-override-rejected'))
  await expect
    .element(page.getByRole('region', { name: 'Rejected override' }))
    .toBeVisible()
  await document.fonts.ready
  await commands.captureFeature('server', 'rejected-override-recovery')
})
