import { createHashHistory, createRouter } from '@tanstack/react-router'
import { routeTree } from '@/generated/routeTree.gen'

export function createAppRouter() {
  return createRouter({ routeTree, history: createHashHistory() })
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof createAppRouter>
  }
}
