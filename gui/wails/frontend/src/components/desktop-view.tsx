import { ErrorNote } from '@/components/error-note'
import { QueryClientContext } from '@tanstack/react-query'
import { useContext, type ReactNode } from 'react'
import { useView, type View } from '@/lib/bindings'
import { guiError } from '@/lib/gui-errors'

export function DesktopView({
  children,
}: {
  children: (view: View, updatedAt: number) => ReactNode
}) {
  const client = useContext(QueryClientContext)
  if (!client)
    return (
      <ErrorNote data-code="internal_error">
        {
          guiError({
            code: 'internal_error',
            remedy: 'Desktop View is missing.',
          }).message
        }
      </ErrorNote>
    )
  return <ConnectedView>{children}</ConnectedView>
}

function ConnectedView({
  children,
}: {
  children: (view: View, updatedAt: number) => ReactNode
}) {
  const query = useView()
  if (!query.data)
    return query.error ? (
      <ErrorNote>{query.error.message}</ErrorNote>
    ) : (
      <p>Reading desktop status…</p>
    )
  return (
    <>
      {query.error && (
        <ErrorNote>
          <span>{query.error.message}</span>
        </ErrorNote>
      )}
      {children(query.data, query.dataUpdatedAt)}
    </>
  )
}
