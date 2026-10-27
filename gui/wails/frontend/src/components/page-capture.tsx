import { QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, RouterProvider } from '@tanstack/react-router'
import { expect, inject } from 'vitest'
import { commands, page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { createAppRouter } from '@/router'
import { connectBindings, viewKey, type View } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'

export async function capturePage({
  feature,
  scenario,
  view,
  prepare,
}: {
  feature: string
  scenario: string
  view: View
  prepare?: () => Promise<void>
}) {
  document.documentElement.dataset.theme = inject('theme')
  const client = createQueryClient()
  client.setQueryData(viewKey, view)
  const disconnect = connectBindings({ getView: async () => view })
  const router = createAppRouter()
  router.update({
    history: createMemoryHistory({
      initialEntries: [feature === 'server' ? '/' : `/${feature}`],
    }),
  })
  try {
    await render(
      <div
        id="shell-capture"
        style={{ width: 1100, height: 760, overflow: 'auto' }}
      >
        <QueryClientProvider client={client}>
          <RouterProvider router={router} />
        </QueryClientProvider>
      </div>,
    )
    await expect
      .element(
        page.getByRole('heading', {
          level: 1,
          name: feature[0]!.toUpperCase() + feature.slice(1),
          exact: true,
        }),
      )
      .toBeVisible()
    if (prepare) await prepare()
    await document.fonts.ready
    expect(await commands.captureFeature(feature, scenario)).toMatch(/\.png$/)
  } finally {
    disconnect()
    client.clear()
  }
}
