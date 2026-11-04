import type { ReactNode } from 'react'

export function Page({
  title,
  inlineTitle,
  children,
}: {
  title: string
  // The first block of the page (a status row) sits beside the title, not under
  // it; the blocks after it clear the title.
  inlineTitle?: boolean
  children?: ReactNode
}) {
  return (
    <section className="@container max-w-5xl" aria-label={title}>
      <h1
        className={
          inlineTitle
            ? 'float-left mr-6 text-3xl font-normal'
            : 'mb-3 text-3xl font-normal'
        }
      >
        {title}
      </h1>
      {children}
    </section>
  )
}
