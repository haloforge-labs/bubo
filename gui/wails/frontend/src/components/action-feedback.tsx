import { useEffect, useRef, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { GUIError } from '@/components/gui-error'
import { useActionTask } from '@/lib/action-task'
export { useActionTask } from '@/lib/action-task'

export function ActionError({ error }: { error?: unknown }) {
  return <GUIError error={error} />
}

export function ActionButton({
  children,
  action,
  disabled,
  done,
  variant = 'outline',
  size,
  busyLabel = 'Working…',
  onPendingChange,
}: {
  children: ReactNode
  action: () => Promise<unknown>
  disabled?: boolean
  done?: string
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'sm'
  // What the button says while its action runs.
  busyLabel?: string
  // Tells the parent while the action runs, so a sibling can be switched off.
  onPendingChange?: (pending: boolean) => void
}) {
  const task = useActionTask()
  const pending = task.pending
  // The parent hears about a change of state only, never about a new callback.
  const report = useRef(onPendingChange)
  useEffect(() => {
    report.current = onPendingChange
  })
  useEffect(() => {
    report.current?.(pending)
  }, [pending])
  return (
    <div className="space-y-2">
      <Button
        variant={variant}
        size={size}
        disabled={disabled || task.pending}
        onClick={() => void task.run(action)}
      >
        {task.pending ? busyLabel : children}
      </Button>
      <GUIError error={task.error} recovery={false} />
      {task.completed && done && <p role="status">{done}</p>}
    </div>
  )
}
