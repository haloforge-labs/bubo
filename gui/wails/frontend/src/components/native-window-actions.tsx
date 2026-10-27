import { requireActionSuccess } from '@/lib/bindings'
import { useEffect } from 'react'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'

export function NativeWindowActions({
  window: desktop,
}: {
  window: View['window']
}) {
  const tray = useActionTask()
  const take = tray.mutate
  useEffect(() => {
    if (!desktop.tray_action) return
    take(async () => {
      // Consume the native queue atomically; a stale polled View is not an action.
      const { value: action } = requireActionSuccess(
        await actions.TakeTrayAction(),
      )
      switch (action) {
        case '':
          return
        case 'show':
          return actions.SetVisible(true)
        case 'start':
          return actions.Start()
        case 'stop':
          return actions.Stop()
        case 'restart':
          return actions.Restart()
        case 'quit':
          return actions.Exit()
        default:
          throw {
            code: 'internal_error',
            remedy: `Unknown tray action: ${action}`,
          }
      }
    })
  }, [desktop.tray_action, take])

  return <ActionError error={tray.error} />
}
