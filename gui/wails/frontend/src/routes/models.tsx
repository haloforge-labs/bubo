import { createFileRoute } from '@tanstack/react-router'
import Page from '@/features/models'

export const Route = createFileRoute('/models')({ component: Page })
