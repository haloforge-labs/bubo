import type { ReactNode } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
  RouterProvider,
} from '@tanstack/react-router'
import { render } from 'vitest-browser-react'
import { connectBindings, viewKey, type View } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'
import { scenarioByName } from '../../tests/scenarios'

// A fixture is the generated scenario View, exactly as Go built it from the
// recordings. Tests no longer rewrite it (E21): a client name, a catalogue row
// or a download flag that a test needs comes from the recording or the test
// sets it itself, in the test, where a reader sees it.
export function fixture(name = 'ready-idle'): View {
  return structuredClone(scenarioByName(name).view)
}

export function renderView(content: ReactNode, view: View) {
  const client = createQueryClient()
  client.setQueryData(viewKey, view)
  const disconnect = connectBindings({ getView: async () => view })
  const routeTree = createRootRoute({ component: () => content })
  const router = createRouter({ routeTree, history: createMemoryHistory() })
  const screen = render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return {
    client,
    screen,
    cleanup: () => {
      disconnect()
      client.clear()
    },
  }
}
