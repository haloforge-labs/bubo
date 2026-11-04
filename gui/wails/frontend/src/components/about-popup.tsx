import type { ReactNode } from 'react'

// The ⓘ at the end of a one-line row. It opens the row's About text as a small
// popup under the row, so the row stays one line.
export function AboutPopup({
  label,
  children,
  className = '',
}: {
  label: string
  children: ReactNode
  className?: string
}) {
  return (
    <details className={`relative text-xs text-muted-foreground ${className}`}>
      <summary
        role="button"
        aria-label={`About ${label}`}
        title={`About ${label}`}
        className="w-fit cursor-pointer list-none px-1 text-base leading-none [&::-webkit-details-marker]:hidden"
      >
        <span aria-hidden="true">ⓘ</span>
      </summary>
      <div className="absolute right-0 top-full z-20 mt-1 w-72 max-w-[80vw] space-y-2 rounded-md border bg-background p-3 shadow-md">
        {children}
      </div>
    </details>
  )
}
