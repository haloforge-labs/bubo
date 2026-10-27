import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/onboarding'

export const Route = createFileRoute('/onboarding')({ component: Page })
