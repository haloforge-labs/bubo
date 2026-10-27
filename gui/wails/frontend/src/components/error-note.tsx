import type { ComponentProps } from 'react'
import { TriangleAlert } from 'lucide-react'
import { cn } from '@/lib/utils'

export function ErrorNote({
  children,
  className,
  ...props
}: ComponentProps<'div'>) {
  return (
    <div
      {...props}
      role="alert"
      className={cn('error-note space-y-2 text-sm', className)}
    >
      <span className="flex items-center gap-2 font-medium text-destructive">
        <TriangleAlert aria-hidden="true" className="size-4 shrink-0" />
        Error
      </span>
      {children}
    </div>
  )
}
