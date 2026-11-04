import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

// The one status style: a plain line for what happened or what is pending.
// Failures keep the error box (ErrorNote); success and pending never use it.
export function StatusLine({ className, ...props }: ComponentProps<'p'>) {
  return (
    <p
      role="status"
      {...props}
      className={cn('text-sm text-muted-foreground', className)}
    />
  )
}
