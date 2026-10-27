import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/about'

export const Route = createFileRoute('/about')({ component: Page })
