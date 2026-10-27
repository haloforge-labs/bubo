import {
  Activity,
  Boxes,
  Cable,
  Info,
  Logs,
  Server,
  Settings,
} from 'lucide-react'
import type { SidebarData } from '../types'

export const sidebarData: SidebarData = {
  navGroups: [
    {
      title: 'Workspace',
      items: [
        { title: 'Server', url: '/', icon: Server },
        { title: 'Models', url: '/models', icon: Boxes },
        { title: 'Connect', url: '/connect', icon: Cable },
        { title: 'Requests', url: '/requests', icon: Activity },
        { title: 'Logs', url: '/logs', icon: Logs },
        { title: 'Settings', url: '/settings', icon: Settings },
        { title: 'About', url: '/about', icon: Info },
      ],
    },
  ],
}
