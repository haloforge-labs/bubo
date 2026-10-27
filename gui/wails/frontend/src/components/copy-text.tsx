import { Check, Copy } from 'lucide-react'
import { useEffect } from 'react'
import { useActionTask } from '@/components/action-feedback'
import { GUIError } from '@/components/gui-error'
import { Button } from '@/components/ui/button'
import { actions } from '@/lib/bindings'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

type CopyTextProps = {
  text: string
  label?: string
  disabled?: boolean
  compact?: boolean
  iconOnly?: boolean
  variant?: 'outline' | 'ghost'
}

export function CopyText(props: CopyTextProps) {
  return <CopyControl key={props.text} {...props} />
}

function CopyControl({
  text,
  label = 'Copy',
  disabled,
  compact = false,
  iconOnly = false,
  variant = 'outline',
}: CopyTextProps) {
  const task = useActionTask()
  const { completed, reset } = task
  useEffect(() => {
    if (!completed) return
    const timer = setTimeout(reset, 2000)
    return () => clearTimeout(timer)
  }, [completed, reset])
  const Icon = task.completed ? Check : Copy
  const button = (
    <Button
      size={iconOnly ? 'icon' : compact ? 'sm' : undefined}
      className={variant === 'ghost' ? 'px-2' : undefined}
      variant={variant}
      aria-label={label}
      disabled={disabled || !text || task.pending}
      onClick={() => void task.run(() => actions.CopyText(text))}
    >
      <Icon aria-hidden="true" />
      {!iconOnly && (task.completed ? 'Copied' : label)}
    </Button>
  )
  return (
    <div className={iconOnly ? 'inline-flex items-center' : 'space-y-2'}>
      {iconOnly ? (
        <Tooltip>
          <TooltipTrigger asChild>{button}</TooltipTrigger>
          <TooltipContent>{task.completed ? 'Copied' : label}</TooltipContent>
        </Tooltip>
      ) : (
        button
      )}
      {task.completed && (
        <span role="status" className="sr-only">
          Copied.
        </span>
      )}
      <GUIError error={task.error} recovery={false} />
    </div>
  )
}
