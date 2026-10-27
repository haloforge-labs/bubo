import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/logs'

export const Route = createFileRoute('/logs')({ component: Page })
