import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { Separator } from '@/components/ui/separator'
import { SidebarTrigger } from '@/components/ui/sidebar'

type HeaderProps = React.HTMLAttributes<HTMLElement> & {
  fixed?: boolean
  // The header holds only the sidebar toggle: it is short and has no separator.
  bare?: boolean
  ref?: React.Ref<HTMLElement>
}

export function Header({
  className,
  fixed,
  bare,
  children,
  ...props
}: HeaderProps) {
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    const onScroll = () => {
      setOffset(document.body.scrollTop || document.documentElement.scrollTop)
    }

    // Add scroll listener to the body
    document.addEventListener('scroll', onScroll, { passive: true })

    // Clean up the event listener on unmount
    return () => document.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={cn(
        'z-50',
        fixed && 'header-fixed peer/header sticky top-0 w-[inherit]',
        offset > 10 && fixed ? 'shadow' : 'shadow-none',
        className,
      )}
      {...props}
    >
      <div
        className={cn(
          'relative flex flex-wrap items-center gap-3 p-3 sm:gap-4',
          bare ? 'min-h-12' : 'min-h-16',
          offset > 10 &&
            fixed &&
            'after:absolute after:inset-0 after:-z-10 after:bg-background/20 after:backdrop-blur-lg',
        )}
      >
        <SidebarTrigger variant="outline" className="max-md:scale-125" />
        {!bare && <Separator orientation="vertical" className="h-6" />}
        {children}
      </div>
    </header>
  )
}
