import type { ComponentProps, ReactNode } from 'react'
import { cn } from '@/lib/utils'

// The one disclosure: a native details element with one marker (▸ closed, ▾ open),
// used for every "More ..." / "Details" / "Debug" / "Model folder" fold in the app.
export function Disclosure({
  summary,
  summaryClassName,
  className,
  children,
  ...props
}: Omit<ComponentProps<'details'>, 'summary'> & {
  summary: ReactNode
  summaryClassName?: string
}) {
  return (
    <details {...props} className={cn('group', className)}>
      <summary
        role="button"
        className={cn(
          'flex w-fit cursor-pointer list-none items-center gap-1.5 [&::-webkit-details-marker]:hidden',
          summaryClassName,
        )}
      >
        <span
          aria-hidden="true"
          className="inline-block w-3 text-muted-foreground group-open:rotate-90"
        >
          ▸
        </span>
        <span>{summary}</span>
      </summary>
      {children}
    </details>
  )
}
