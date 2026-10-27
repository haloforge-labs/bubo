import { QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, RouterProvider } from '@tanstack/react-router'
import { expect, test, vi } from 'vitest'
import { render } from 'vitest-browser-react'
import { sidebarData } from '@/components/layout/data/sidebar-data'
import { connectBindings, viewKey } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'
import { createAppRouter } from '@/router'
import { scenarios } from '../../tests/scenarios'

const routes = [
  ...sidebarData.navGroups.flatMap((group) =>
    group.items.map((item) => item.url),
  ),
  '/onboarding',
]
const cases = scenarios.flatMap((scenario) =>
  routes.map((route) => ({ scenario, route })),
)

test('render sweep has routes and recorded states', () => {
  expect(routes.length).toBeGreaterThan(0)
  expect(scenarios.length).toBeGreaterThan(0)
  expect(new Set(routes).size).toBe(routes.length)
  console.log(
    `render sweep: ${routes.length} routes × ${scenarios.length} scenarios = ${cases.length} cases`,
  )
})

test.each(cases)(
  '$scenario.name $route renders content without page errors',
  async ({ scenario, route }) => {
    const errors: string[] = []
    const onError = (event: ErrorEvent) => {
      errors.push(`pageerror: ${event.message}`)
    }
    const onRejection = (event: PromiseRejectionEvent) => {
      errors.push(`unhandledrejection: ${String(event.reason)}`)
    }
    const browserWindow = document.defaultView
    if (!browserWindow) throw new Error('Browser window is unavailable')
    browserWindow.addEventListener('error', onError)
    browserWindow.addEventListener('unhandledrejection', onRejection)
    const consoleError = vi
      .spyOn(console, 'error')
      .mockImplementation((...args: unknown[]) => {
        errors.push(`console.error: ${args.map(String).join(' ')}`)
      })
    const consoleWarn = vi
      .spyOn(console, 'warn')
      .mockImplementation((...args: unknown[]) => {
        errors.push(`console.warn: ${args.map(String).join(' ')}`)
      })
    const client = createQueryClient()
    client.setQueryData(viewKey, scenario.view)
    const disconnect = connectBindings({ getView: async () => scenario.view })
    const router = createAppRouter()
    router.update({ history: createMemoryHistory({ initialEntries: [route] }) })

    try {
      render(
        <QueryClientProvider client={client}>
          <RouterProvider router={router} />
        </QueryClientProvider>,
      )
      let contentError: unknown
      try {
        await expect
          .poll(
            () =>
              document.querySelector('#main-content')?.textContent?.trim()
                .length ?? 0,
          )
          .toBeGreaterThan(0)
      } catch (error) {
        contentError = error
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      expect(errors, `${scenario.name} ${route}: ${errors.join('\n')}`).toEqual(
        [],
      )
      if (contentError) throw contentError
    } finally {
      disconnect()
      client.clear()
      consoleError.mockRestore()
      consoleWarn.mockRestore()
      browserWindow.removeEventListener('error', onError)
      browserWindow.removeEventListener('unhandledrejection', onRejection)
    }
  },
)
