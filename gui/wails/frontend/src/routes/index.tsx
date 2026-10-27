import { createFileRoute } from '@tanstack/react-router'
import Server from '@/features/server'
import Onboarding from '@/features/onboarding'
import { useView } from '@/lib/bindings'

export function HomePage({
  packagePrepared,
  startingCLI,
  statusLabel,
}: {
  packagePrepared?: boolean
  startingCLI?: boolean
  statusLabel?: string
}) {
  // The body uses the header's label: Go owns the one wording for this state.
  if (startingCLI)
    return (
      <section aria-label="Server">
        <h1 className="mb-6 text-3xl">Server</h1>
        <p role="status" className="text-muted-foreground">
          {statusLabel}
        </p>
      </section>
    )
  if (packagePrepared === false) return <Onboarding />
  if (packagePrepared === true) return <Server />
  return (
    <section aria-label="Server">
      <h1 className="mb-6 text-3xl">Server</h1>
      <p className="text-muted-foreground">
        Package preparation status is unavailable until the desktop View is
        connected.
      </p>
    </section>
  )
}

function HomeRoute() {
  const view = useView()
  return (
    <HomePage
      packagePrepared={view.data?.prepared}
      startingCLI={view.data?.startup_phase === 'starting_cli'}
      statusLabel={view.data?.status_label}
    />
  )
}

export const Route = createFileRoute('/')({ component: HomeRoute })
