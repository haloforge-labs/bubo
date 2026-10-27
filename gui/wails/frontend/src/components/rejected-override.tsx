import { Link } from '@tanstack/react-router'
import { actions, useView, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { ErrorNote } from '@/components/error-note'
import { serverRowSection } from '@/lib/server-settings'

export function RejectedOverride({ failure }: { failure: View['failure'] }) {
  const command = useActionTask()
  const view = useView().data
  const name = failure?.switch
  const row = view?.switches.find((row) => row.name === name)
  const serverSetting = row && serverRowSection(row)
  const label =
    view?.switches.find((row) => row.name === name)?.label || 'server setting'
  if (!name) return null
  return (
    <section aria-label="Rejected override" className="space-y-2">
      <ErrorNote className="space-y-2">
        <p className="text-sm text-destructive">
          The server did not start because of the {label} setting you changed.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            disabled={command.isPending}
            onClick={() =>
              command.mutate(() =>
                actions.SetSwitchOverrides({ set: [], clear: [name] }),
              )
            }
          >
            Reset {label}
          </Button>
          <Button variant="outline" asChild>
            <Link
              to={serverSetting ? '/' : '/settings'}
              hash={serverSetting ? 'server-setup' : ''}
            >
              Open setting
            </Link>
          </Button>
        </div>
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Details</summary>
          <p className="pt-2 font-mono">
            {failure.code}: {failure.remedy}
          </p>
        </details>
      </ErrorNote>
      <ActionError error={command.error} />
    </section>
  )
}
