import { links } from '@/lib/links'
import { useNavigate } from '@tanstack/react-router'
import { actions, viewKey } from '@/lib/bindings'
import { asGUIFailure, guiError } from '@/lib/gui-errors'
import type { GUIFailure } from '@/lib/gui-errors'
import { useActionTask } from '@/lib/action-task'
import { Button } from '@/components/ui/button'
import { queryClient } from '@/lib/query-client'
import { ErrorNote } from '@/components/error-note'

export function GUIError({
  error,
  recovery = true,
  retry,
  buttonOnly = false,
  recoverAction,
}: {
  error: unknown
  recovery?: boolean
  retry?: () => Promise<unknown>
  // A state, not a failure: only the one button, in no box, with no sentence.
  buttonOnly?: boolean
  // The update card recovers through its own binding.
  recoverAction?: () => Promise<unknown>
}) {
  // The identified setting has its own single recovery box (reset/open).
  if (
    error &&
    typeof error === 'object' &&
    'switch' in error &&
    error.switch &&
    !('code' in error && error.code === 'setting_read_only')
  )
    return null
  const failure = asGUIFailure(error)
  return failure ? (
    <FailureContent
      key={`${failure.code}:${failure.remedy}`}
      failure={failure}
      recovery={recovery}
      retry={retry}
      buttonOnly={buttonOnly}
      recoverAction={recoverAction}
    />
  ) : null
}

function FailureContent({
  failure: original,
  recovery,
  retry,
  buttonOnly,
  recoverAction,
}: {
  failure: GUIFailure
  recovery: boolean
  retry?: () => Promise<unknown>
  buttonOnly: boolean
  recoverAction?: () => Promise<unknown>
}) {
  const task = useActionTask()
  const navigate = useNavigate()
  const failure = asGUIFailure(task.error) ?? original
  const words = guiError(failure)
  const target = {
    server: '/',
    'server-setup': '/',
    models: '/models',
    settings: '/settings',
    connect: '/connect',
    about: '/about',
    logs: '/logs',
  } as const
  const action =
    words.recovery === 'prepare'
      ? actions.Prepare
      : words.recovery === 'retry'
        ? (retry ??
          (() => queryClient.invalidateQueries({ queryKey: viewKey })))
        : words.recovery === 'vcredist'
          ? () =>
              actions.OpenExternal(
                'https://aka.ms/vs/17/release/vc_redist.x64.exe',
              )
          : words.recovery === 'download'
            ? () => actions.OpenExternal(links.releases)
            : words.recovery === 'restart'
              ? actions.Restart
              : words.recovery === 'recover'
                ? (recoverAction ?? actions.InstallUpdate)
                : undefined
  if (buttonOnly) {
    if (words.recovery === 'none' || !recovery) return null
    return (
      <Button
        disabled={task.isPending}
        onClick={() =>
          action
            ? task.mutate(action)
            : void navigate({
                to: target[words.recovery as keyof typeof target],
              })
        }
      >
        {words.action}
      </Button>
    )
  }
  return (
    <ErrorNote className="space-y-2">
      <p className="text-sm text-destructive">{words.message}</p>

      {recovery &&
        words.recovery !== 'none' &&
        (action ? (
          <Button disabled={task.isPending} onClick={() => task.mutate(action)}>
            {words.action}
          </Button>
        ) : (
          <Button
            variant="outline"
            role={words.recovery === 'logs' ? 'link' : undefined}
            onClick={() => {
              void navigate({
                to: target[words.recovery as keyof typeof target],
                hash: words.recovery === 'server-setup' ? 'server-setup' : '',
              }).then(() => {
                if (words.recovery === 'server-setup')
                  document.getElementById('server-setup')?.scrollIntoView()
              })
            }}
          >
            {words.action}
          </Button>
        ))}
      {words.below && <p className="text-sm">{words.below}</p>}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">Details</summary>
        <p className="pt-2 font-mono">
          {failure.code}: {failure.remedy}
        </p>
        {failure.detail && (
          <p className="pt-2 font-mono break-all">{failure.detail}</p>
        )}
      </details>
    </ErrorNote>
  )
}
