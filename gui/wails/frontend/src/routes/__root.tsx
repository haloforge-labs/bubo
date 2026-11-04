import { useEffect } from 'react'
import { createRootRoute, Outlet } from '@tanstack/react-router'
import { applyFont, interfaceFont } from '@/lib/font'
import { Toaster } from 'sonner'
import { ViewShell } from '@/components/view-shell'
import { GUIError } from '@/components/gui-error'
import { WhatsNewDialog } from '@/components/whats-new-dialog'
import { useView } from '@/lib/bindings'

function RootContent() {
  const view = useView()
  // The saved font decides once the view has arrived.
  const saved = view.data?.saved.font
  const loaded = view.data !== undefined
  useEffect(() => {
    if (loaded) applyFont(interfaceFont(saved))
  }, [loaded, saved])
  return (
    <ViewShell>
      <button
        type="button"
        className="fixed left-1 top-1 z-50 h-1 w-1 overflow-hidden opacity-0 focus:h-auto focus:w-auto focus:overflow-visible focus:rounded-md focus:bg-background focus:px-3 focus:py-2 focus:opacity-100 focus:shadow-md"
        onClick={() => {
          const main = document.getElementById('main-content')
          main?.setAttribute('tabindex', '-1')
          main?.focus()
        }}
      >
        Skip to content
      </button>
      {view.data?.startup_phase === 'startup_slow' && (
        <GUIError error={{ code: 'startup_slow', remedy: '' }} />
      )}
      <Outlet />
      <WhatsNewDialog />
      <Toaster />
    </ViewShell>
  )
}

export const Route = createRootRoute({
  component: RootContent,
  errorComponent: () => (
    <ViewShell>
      <GUIError error={{ code: 'page_error', remedy: '' }} />
    </ViewShell>
  ),
  notFoundComponent: () => <p>This page could not be found.</p>,
})
