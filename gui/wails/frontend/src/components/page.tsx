import type { ReactNode } from 'react'

export function Page({
  title,
  children,
}: {
  title: string
  children?: ReactNode
}) {
  return (
    <section className="max-w-5xl" aria-label={title}>
      <h1 className="mb-6 text-3xl font-normal">{title}</h1>
      {children}
    </section>
  )
}
