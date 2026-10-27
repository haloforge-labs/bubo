import { Link } from '@tanstack/react-router'
import type { View } from '@/lib/bindings'
import { updatePresentation } from '@/lib/format'
import { ErrorNote } from '@/components/error-note'

export function UpdateBanner({ update }: { update: View['update'] }) {
  const labels: Record<string, string> = {
    available: 'Update available',
    progress: 'Downloading update',
    ready: 'Update ready',
    failed: 'Update failed',
    rollback_pending: "Update didn't finish",
  }
  // A failed start of the updater or a failed check is not a failed update.
  const label = update.check_failed
    ? undefined
    : labels[updatePresentation(update)]
  const failed = ['failed', 'rollback_pending'].includes(
    updatePresentation(update),
  )
  if (!label) return null
  const link = (
    <Link
      to="/about"
      className={
        failed
          ? 'inline-flex min-h-9 items-center gap-2 whitespace-nowrap text-sm underline-offset-4 hover:underline'
          : 'inline-flex min-h-9 items-center gap-2 whitespace-nowrap rounded border px-3 py-1 text-sm hover:bg-secondary'
      }
    >
      {label}
      {update.available_version ? (
        <>
          {' '}
          · <span className="tabular-nums">{update.available_version}</span>
        </>
      ) : (
        ''
      )}
      <span aria-hidden="true">→</span>
    </Link>
  )
  return failed ? (
    <ErrorNote className="flex items-center gap-2 space-y-0">{link}</ErrorNote>
  ) : (
    link
  )
}
