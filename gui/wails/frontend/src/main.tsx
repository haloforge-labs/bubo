import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { queryClient } from '@/lib/query-client'
import { createAppRouter } from '@/router'
import { subscribeViewEvents } from '@/lib/bindings'
import '@/styles/app.css'

const root = document.getElementById('root')
if (!root) throw new Error('The application root is missing.')
const router = createAppRouter()

function DesktopApp() {
  useEffect(() => subscribeViewEvents(), [])
  return <RouterProvider router={router} />
}

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <DesktopApp />
    </QueryClientProvider>
  </StrictMode>,
)
