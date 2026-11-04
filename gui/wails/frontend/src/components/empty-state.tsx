import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

// The one empty-state pattern: a single plain sentence, no chart, no table, no
// count. Pages show it instead of their content while there is none.
export function EmptyState({ className, ...props }: ComponentProps<'p'>) {
  return <p {...props} className={cn('text-muted-foreground', className)} />
}
