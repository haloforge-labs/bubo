import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/requests'

export const Route = createFileRoute('/requests')({ component: Page })
