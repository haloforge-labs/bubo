import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/settings'

export const Route = createFileRoute('/settings')({ component: Page })
