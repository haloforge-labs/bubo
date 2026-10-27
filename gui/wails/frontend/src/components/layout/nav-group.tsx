import { useLinkProps, useLocation } from '@tanstack/react-router'
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from '@/components/ui/sidebar'
import type { NavLink, NavGroup as NavGroupProps } from './types'

// Retain the template's flat-link branch; Bubo has no nested/demo groups.
export function NavGroup({
  title,
  items,
  gettingStarted = false,
}: NavGroupProps & { gettingStarted?: boolean }) {
  const href = useLocation({ select: (location) => location.pathname })
  return (
    <SidebarGroup>
      <SidebarGroupLabel>{title}</SidebarGroupLabel>
      <SidebarMenu>
        {items.map((item) => (
          <SidebarMenuLink
            key={item.url}
            item={
              gettingStarted && item.url === '/'
                ? { ...item, title: 'Get started' }
                : item
            }
            href={href}
          />
        ))}
      </SidebarMenu>
    </SidebarGroup>
  )
}

function SidebarMenuLink({ item, href }: { item: NavLink; href: string }) {
  const { setOpenMobile } = useSidebar()
  // Home also hosts setup; both setup URLs mark the first navigation item.
  const link = useLinkProps({
    to: item.url,
    activeOptions: { exact: true },
    onClick: () => setOpenMobile(false),
  })
  const active =
    href === item.url || (item.url === '/' && href === '/onboarding')
  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={active} tooltip={item.title}>
        <a {...link} aria-current={active ? 'page' : undefined}>
          {item.icon && <item.icon aria-hidden="true" />}
          <span>{item.title}</span>
        </a>
      </SidebarMenuButton>
    </SidebarMenuItem>
  )
}
