import { useEffect, useRef } from 'react'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { GUIError } from '@/components/gui-error'
import { guiError } from '@/lib/gui-errors'

const setupCodes = [
  'runtime_missing',
  'model_missing',
  'source_missing',
  'not_prepared',
]

// A reason that describes a state, not a failed action. It is never red: the
// status label and the progress line already say what is going on. A disabled
// control with such a reason is hidden, not shown with an explanation.
const stateReasons = [
  'busy',
  'update_installing',
  'closing',
  'initializing',
  'external_read_only',
]

// The supervisor admits the saved setup after its memory check.
export function Lifecycle({
  view,
  blocked,
  serverPage = false,
  onStartRequested,
  onStartFailed,
  beforeStart,
  hasUnsavedSetup = false,
}: {
  view: View
  blocked?: string
  serverPage?: boolean
  onUseSuggestedContext?: () => void
  onStartRequested?: () => void
  onStartFailed?: () => void
  beforeStart?: () => Promise<void>
  hasUnsavedSetup?: boolean
}) {
  const command = useActionTask()
  const {
    error: commandError,
    isPending: commandPending,
    reset: resetCommand,
  } = command
  const observation = `${view.generation}:${view.status_feed?.sample_time}`
  const previousObservation = useRef(observation)
  useEffect(() => {
    if (
      previousObservation.current !== observation &&
      !!commandError &&
      !commandPending &&
      !view.failure &&
      view.status_feed?.state === 'updating'
    )
      resetCommand()
    previousObservation.current = observation
  }, [
    observation,
    view.failure,
    view.status_feed?.state,
    resetCommand,
    commandError,
    commandPending,
  ])
  const calls = {
    start: actions.Start,
    stop: actions.Stop,
    restart: actions.Restart,
  }
  const labels = {
    start: 'Start server',
    stop: 'Stop server',
    restart: 'Restart server',
  }
  const reasons = [
    ...new Set(
      (['start', 'stop', 'restart'] as const).flatMap((name) => {
        const action = view.actions[name]
        return action.reason_code === 'not_applicable' ||
          action.reason_code === 'setting_rejected'
          ? []
          : action.reason_code
            ? [action.reason_code]
            : []
      }),
    ),
  ]
  // Setup that is missing is a state: one button, no box, no disabled Start.
  // A failure is not a state: its own row shows the cause and the fix, and
  // this row offers Start only.
  const setupReasons = view.failure
    ? []
    : reasons.filter((code) => setupCodes.includes(code))
  // Start is drawn unless one of the rules below hides it.
  const start = view.actions.start
  const startShown = !(
    start.reason_code === 'not_applicable' ||
    (!start.allowed &&
      (stateReasons.includes(start.reason_code) ||
        setupCodes.includes(start.reason_code))) ||
    start.reason_code === 'rollback_pending' ||
    (view.failure && !start.allowed)
  )
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        {!view.failure?.switch &&
          setupReasons.map((code) => (
            <GUIError
              key={code}
              buttonOnly
              error={{
                code,
                remedy: [view.actions.start, view.actions.restart].find(
                  (action) => action.reason_code === code,
                )!.reason,
              }}
            />
          ))}
        {(['start', 'stop', 'restart'] as const).map((name) => {
          const action = view.actions[name]
          if (action.reason_code === 'not_applicable') return null
          if (!action.allowed && stateReasons.includes(action.reason_code))
            return null
          if (!action.allowed && setupCodes.includes(action.reason_code))
            return null
          if (name === 'start' && action.reason_code === 'rollback_pending')
            return null
          if (name === 'start' && view.failure && !action.allowed) return null
          return (
            <Button
              key={name}
              variant={
                view.phase === 'stalled'
                  ? name === 'restart'
                    ? 'default'
                    : 'outline'
                  : name === 'stop' || name === 'restart' || view.failure
                    ? 'outline'
                    : 'default'
              }
              disabled={
                command.isPending ||
                (!action.allowed &&
                  !(
                    hasUnsavedSetup &&
                    name !== 'stop' &&
                    [
                      'memory_tight',
                      'memory_does_not_fit',
                      'memory_does_not_fit_any',
                      'estimate_unavailable',
                      'setting_rejected',
                    ].includes(action.reason_code)
                  )) ||
                (name !== 'stop' && !!blocked)
              }
              onClick={() => {
                if (name !== 'stop') onStartRequested?.()
                command.mutate(
                  async () => {
                    if (name !== 'stop') await beforeStart?.()
                    return calls[name]()
                  },
                  {
                    onError: () => {
                      if (name !== 'stop') onStartFailed?.()
                    },
                  },
                )
              }}
            >
              {labels[name]}
            </Button>
          )
        })}
      </div>
      {blocked && (
        <p className="w-full text-sm text-muted-foreground">{blocked}</p>
      )}
      {!view.failure?.switch &&
        reasons.map((code) => {
          const action = [
            view.actions.start,
            view.actions.stop,
            view.actions.restart,
          ].find((row) => row.reason_code === code)!
          if (setupCodes.includes(code)) return null
          if (
            serverPage &&
            [
              'memory_tight',
              'memory_does_not_fit',
              'memory_does_not_fit_any',
            ].includes(code)
          )
            return null // The setup owns the single current memory panel.
          if (
            (code === 'estimate_unavailable' ||
              code === 'readiness_unavailable') &&
            view.fit.class === 'unavailable'
          )
            // No estimate yet. A disabled Start says why in one muted line
            // (Go's own sentence for this state); a hidden Start needs none.
            return startShown && !view.actions.start.allowed ? (
              <p key={code} className="w-full text-sm text-muted-foreground">
                {view.fit.sentence}
              </p>
            ) : null
          if (code === 'busy') return null
          // The status label already says a server started outside Bubo.
          if (code === 'external_read_only') return null
          if (stateReasons.includes(code))
            return (
              <p key={code} className="w-full text-sm text-muted-foreground">
                {guiError({ code, remedy: action.reason }).message}
              </p>
            )
          // An allowed action never needs a reason on screen.
          if (action.allowed) return null
          return (
            <div key={code} className="w-full">
              <GUIError
                error={{
                  code,
                  remedy: action.reason,
                  label:
                    code === 'rollback_pending'
                      ? view.update.previous_version
                      : undefined,
                  serverStopped: !view.serving,
                }}
                recovery={!serverPage || code !== 'memory_does_not_fit'}
              />
            </div>
          )
        })}
      <ActionError
        error={
          serverPage &&
          command.error &&
          [
            'memory_tight',
            'memory_does_not_fit',
            'memory_does_not_fit_any',
          ].includes(command.error.code)
            ? undefined
            : command.error
        }
      />
    </>
  )
}
