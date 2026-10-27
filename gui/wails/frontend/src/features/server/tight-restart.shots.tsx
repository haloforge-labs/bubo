import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { mountCorePage } from '@/components/core-page-test-support'

// Exact Views retained by TestTightRestartSavedCheckboxWorkflow, which drives
// Save/checkbox/Restart through AppService and NativeBackend. The scripts copy
// the corresponding run's retained inputs before capturing the browser.
for (const state of ['ready-tight-blocked', 'ready-tight-allowed']) {
  test(`Tight restart supervisor ${state}`, async () => {
    const view = inject('tightRestartViews')?.[state]
    expect(view, 'run the supervisor workflow before screenshots').toBeDefined()
    if (!view) throw new Error('The supervisor workflow View is missing.')
    document.documentElement.dataset.theme = inject('theme')
    mountCorePage('/', structuredClone(view))
    await expect
      .element(page.getByRole('region', { name: 'Server setup', exact: true }))
      .toBeVisible()
    const shell = document.getElementById('shell-capture')!
    shell.style.height = '760px'
    shell.style.overflow = 'auto'
    await document.fonts.ready
    await commands.captureFeature('server', `update-reuse-${state}-status`)
    const panel = page.getByRole('region', { name: 'Memory fit' })
    // Before the fix this actual supervisor View hides Tight as "In use".
    // Keep that omission visible in the before capture.
    if (panel.elements().length)
      panel.element().scrollIntoView({ block: 'start' })
    await document.fonts.ready
    await commands.captureFeature('server', `update-reuse-${state}`)
  })
}
