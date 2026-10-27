import type { ReactNode } from 'react'
import { AppSidebar } from '@/components/layout/app-sidebar'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar'
import type { View } from '@/lib/bindings'
import { UpdateBanner } from '@/components/update-banner'
import { CopyText } from '@/components/copy-text'

export function AppShell({
  children,
  statusLabel,
  serverState,
  addresses,
  update,
  renderUpdateBanner,
  gettingStarted = false,
  showServerSummary = true,
  showAddresses = true,
  product,
}: {
  children: ReactNode
  statusLabel?: string
  serverState?: View['state']
  addresses?: View['endpoints']
  update?: View['update']
  renderUpdateBanner?: (update: View['update']) => ReactNode
  gettingStarted?: boolean
  showServerSummary?: boolean
  showAddresses?: boolean
  product?: View['product']
}) {
  return (
    <SidebarProvider>
      <AppSidebar gettingStarted={gettingStarted} version={product?.version} />
      <SidebarInset className="min-w-0">
        <Header className="min-h-20 border-b">
          {showServerSummary && (
            <div className="text-sm" role="status">
              {statusLabel ?? serverState}
            </div>
          )}
          <div
            className="peer ml-auto text-sm empty:ml-0"
            aria-label="Application update"
            data-update-state={update?.state}
          >
            {update &&
              (renderUpdateBanner ? (
                renderUpdateBanner(update)
              ) : (
                <UpdateBanner update={update} />
              ))}
          </div>
          {showServerSummary && (
            <>
              {showAddresses &&
                (addresses?.openai_base_url ||
                  addresses?.anthropic_base_url) && (
                  <div
                    className="ml-auto flex min-w-0 flex-wrap items-center gap-2 text-xs peer-has-[a]:ml-0"
                    aria-label="Server addresses"
                  >
                    {addresses.openai_base_url && (
                      <CopyText
                        compact
                        label="Copy OpenAI address"
                        text={addresses.openai_base_url}
                      />
                    )}
                    {addresses.anthropic_base_url && (
                      <CopyText
                        compact
                        label="Copy Anthropic address"
                        text={addresses.anthropic_base_url}
                      />
                    )}
                  </div>
                )}
            </>
          )}
        </Header>
        <Main id="main-content" className="p-6 md:p-10">
          {children}
        </Main>
      </SidebarInset>
    </SidebarProvider>
  )
}
