import type { ReactNode } from 'react'
import { AppShell } from '@/components/app-shell'
import { useView } from '@/lib/bindings'
import { NativeWindowActions } from '@/components/native-window-actions'
import { serverAddresses } from '@/lib/format'
import { useRouterState } from '@tanstack/react-router'
import { GUIError } from '@/components/gui-error'
import { StatusLine } from '@/components/status-line'
import { guiError } from '@/lib/gui-errors'
import { GPUHistoryCollector } from '@/lib/gpu-history'

export function ViewShell({ children }: { children: ReactNode }) {
  const view = useView()
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const statusLabel = view.error
    ? view.error.message
    : view.data
      ? view.data.status_label
      : 'Waiting for the desktop View.'
  return (
    <AppShell
      statusLabel={statusLabel}
      addresses={view.data ? serverAddresses(view.data) : undefined}
      update={view.data?.update}
      product={view.data?.product}
      gettingStarted={view.data?.prepared === false}
      // Connect shows the same addresses in its body.
      showAddresses={pathname !== '/connect'}
      // Server's own page says the status; so do the first-start and
      // missing-CLI bodies. The header repeats it nowhere else.
      showServerSummary={
        !!view.error ||
        pathname !== '/' ||
        (!view.data?.prepared &&
          view.data?.phase !== 'checking' &&
          view.data?.phase !== 'cli_missing')
      }
    >
      {view.data && (
        <GPUHistoryCollector view={view.data} at={view.dataUpdatedAt} />
      )}
      {view.data?.warnings.map((warning) => (
        <p
          key={warning}
          role="status"
          className="mb-4 text-sm text-muted-foreground"
        >
          {warning}
        </p>
      ))}
      {children}
      {view.data?.notice &&
        // A saved support bundle is confirmed by the Logs page itself; nothing
        // else in the shell repeats it.
        view.data.notice.code !== 'support_bundle_saved' &&
        (view.data.notice.code === 'settings_saved' ? (
          <StatusLine>
            {
              guiError({
                code: view.data.notice.code,
                remedy: view.data.notice.detail,
              }).message
            }
          </StatusLine>
        ) : (
          <GUIError
            error={{
              code: view.data.notice.code,
              remedy: view.data.notice.detail,
            }}
          />
        ))}
      {view.data && <NativeWindowActions window={view.data.window} />}
    </AppShell>
  )
}
