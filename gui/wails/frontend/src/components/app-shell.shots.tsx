import { QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, RouterProvider } from '@tanstack/react-router'
import { expect, inject, test } from 'vitest'
import { commands, page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { createAppRouter } from '@/router'
import { connectBindings, viewKey } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'
import { scenarioByName } from '../../tests/scenarios'

for (const name of ['ready-idle']) {
  test(`recorded shell: ${name}`, async () => {
    const scenario = scenarioByName(name)
    const view = structuredClone(scenario.view)
    document.documentElement.dataset.theme = inject('theme')
    const client = createQueryClient()
    client.setQueryData(viewKey, view)
    const disconnect = connectBindings({ getView: async () => view })
    const router = createAppRouter()
    router.update({ history: createMemoryHistory({ initialEntries: ['/'] }) })
    try {
      render(
        <div id="shell-capture">
          <QueryClientProvider client={client}>
            <RouterProvider router={router} />
          </QueryClientProvider>
        </div>,
      )
      await expect
        .element(page.getByRole('heading', { name: 'Server', exact: true }))
        .toBeVisible()
      await document.fonts.ready
      const path = await commands.captureFeature(
        'server',
        `shell-${scenario.name}`,
      )
      expect(path).toMatch(/\.png$/)
    } finally {
      disconnect()
      client.clear()
    }
  })
}
