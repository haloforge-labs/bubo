import type { ElementType } from 'react'
import type { LinkProps } from '@tanstack/react-router'

export type NavLink = {
  title: string
  icon?: ElementType
  url: NonNullable<LinkProps['to']>
}
export type NavGroup = {
  title: string
  items: NavLink[]
}
export type SidebarData = { navGroups: NavGroup[] }
