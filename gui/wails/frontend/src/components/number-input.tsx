import { useState, type ComponentProps } from 'react'
import { Input } from '@/components/ui/input'

// Intl supplies locale separators. Keep a text editor while typing so grouping
// never changes the caret/selection; no formatted-input library is installed.
export function NumberInput({
  value,
  onValueChange,
  onBlur,
  ...props
}: Omit<ComponentProps<typeof Input>, 'type' | 'value' | 'onChange'> & {
  value?: number
  onValueChange: (value: number) => void
}) {
  const [draft, setDraft] = useState<string>()
  const valid = value != null && Number.isFinite(value)
  const parts = new Intl.NumberFormat().formatToParts(12345.6)
  const group = parts.find((part) => part.type === 'group')?.value
  const decimal = parts.find((part) => part.type === 'decimal')?.value ?? '.'
  return (
    <Input
      {...props}
      type="text"
      role="spinbutton"
      inputMode="decimal"
      aria-valuenow={valid ? value : undefined}
      aria-valuemin={props.min != null ? Number(props.min) : undefined}
      aria-valuemax={props.max != null ? Number(props.max) : undefined}
      value={
        draft ??
        (valid
          ? value.toLocaleString(undefined, { maximumFractionDigits: 20 })
          : '')
      }
      onBlur={(event) => {
        setDraft(undefined)
        onBlur?.(event)
      }}
      onKeyDown={(event) => {
        if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
          event.preventDefault()
          const step = props.step === 'any' ? 1 : Number(props.step ?? 1)
          const next =
            (valid ? value : 0) + (event.key === 'ArrowUp' ? step : -step)
          onValueChange(
            Math.min(
              props.max != null ? Number(props.max) : Infinity,
              Math.max(props.min != null ? Number(props.min) : -Infinity, next),
            ),
          )
          setDraft(undefined)
        }
      }}
      onChange={(event) => {
        const text = event.target.value
        setDraft(text)
        const ungrouped = group ? text.replaceAll(group, '') : text
        onValueChange(
          text.trim() ? Number(ungrouped.replace(decimal, '.')) : NaN,
        )
      }}
    />
  )
}
