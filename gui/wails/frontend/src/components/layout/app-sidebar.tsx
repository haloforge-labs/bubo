import { links } from '@/lib/links'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from '@/components/ui/sidebar'
import { AppTitle } from './app-title'
import { sidebarData } from './data/sidebar-data'
import { NavGroup } from './nav-group'
import { actions } from '@/lib/bindings'

// Imported from shadcn-admin; Bubo has one workspace and no account menu.
export function AppSidebar({
  gettingStarted = false,
  version,
}: {
  gettingStarted?: boolean
  version?: string
}) {
  return (
    <Sidebar collapsible="icon" variant="sidebar" aria-label="Sidebar">
      <SidebarHeader>
        <AppTitle />
      </SidebarHeader>
      <SidebarContent>
        <nav aria-label="Main navigation">
          {sidebarData.navGroups.map((props) => (
            <NavGroup
              key={props.title}
              {...props}
              gettingStarted={gettingStarted}
            />
          ))}
        </nav>
      </SidebarContent>
      <SidebarFooter className="group-data-[collapsible=icon]:hidden">
        <div className="space-y-1 px-2 py-3 text-sm text-muted-foreground">
          <p>Halowin Bubo{version && ` ${version}`}</p>
          <p>by HaloForge Labs</p>
          <div className="flex gap-4">
            {[
              ['Website', links.website],
              ['GitHub', links.repository],
            ].map(([label, url]) => (
              <a
                key={url}
                href={url}
                className="underline"
                onClick={(event) => {
                  event.preventDefault()
                  void actions.OpenExternal(url!)
                }}
              >
                {label}
              </a>
            ))}
          </div>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
