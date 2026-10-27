import { Link } from '@tanstack/react-router'
import owlUrl from '../../../../icons/owl.svg?url'
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from '@/components/ui/sidebar'

export function AppTitle() {
  const { setOpenMobile } = useSidebar()
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          size="lg"
          className="gap-2 bg-sidebar py-0 hover:bg-sidebar active:bg-sidebar"
          asChild
        >
          <Link
            to="/"
            aria-label="Bubo home"
            onClick={() => setOpenMobile(false)}
          >
            <span
              aria-hidden="true"
              className="size-4 shrink-0 bg-brand [mask-size:contain] [mask-repeat:no-repeat] [mask-position:center]"
              style={{ maskImage: `url("${owlUrl}")` }}
            />
            <span className="grid flex-1 text-start text-sm leading-tight">
              <span className="truncate text-2xl text-foreground">Bubo</span>
            </span>
          </Link>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
