import { ErrorNote } from '@/components/error-note'
import { useId } from 'react'
import {
  Controller,
  type UseFormReturn,
  type FieldValues,
} from 'react-hook-form'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { fieldsFromRegistry, type RegistryRow } from '@/lib/registry'

function effortParts(defaultValue: unknown, value: unknown) {
  return [
    ...new Map(
      [String(defaultValue ?? ''), String(value ?? '')].flatMap((text) =>
        text
          .split(',')
          .filter((part) => part.includes(':'))
          .map((part) => {
            const [effort, bias] = part.trim().split(':')
            return [effort!, bias ?? ''] as const
          }),
      ),
    ).entries(),
  ]
}

export function RegistryFields<T extends FieldValues>({
  rows,
  form,
  disabled,
  sliders,
  compact,
  defaultHint,
  numericInput,
}: {
  rows: readonly RegistryRow[]
  form: UseFormReturn<Record<string, unknown>, unknown, T>
  disabled?: boolean
  sliders?: boolean
  compact?: boolean
  defaultHint?: string
  numericInput?: boolean
}) {
  const id = useId()
  const fields = fieldsFromRegistry(rows)
  return (
    <>
      {fields.map((field, index) => {
        if (field.control === 'unsupported') return null
        const inputId = `${id}-${index}`
        const descriptionId = `${inputId}-description`
        return (
          <Controller
            key={field.name}
            name={field.name}
            control={form.control}
            render={({ field: input, fieldState }) => (
              <div className="space-y-2">
                <Label htmlFor={inputId}>{field.label}</Label>
                {compact ? (
                  <>
                    <p className="text-xs text-muted-foreground">
                      Default:{' '}
                      {defaultHint ??
                        field.valueLabels?.[String(field.defaultValue)] ??
                        String(field.defaultValue ?? 'automatic')}
                    </p>
                    <details className="text-xs text-muted-foreground">
                      <summary
                        role="button"
                        aria-label={`About ${field.label}`}
                        className="w-fit cursor-pointer underline underline-offset-2"
                      >
                        About
                      </summary>
                      <p id={descriptionId} className="pt-2">
                        {field.description}
                      </p>
                      {field.suggestion && (
                        <p>
                          Suggested: {field.suggestion}. Other values are
                          accepted when supported.
                        </p>
                      )}
                    </details>
                  </>
                ) : (
                  <p
                    id={descriptionId}
                    className="text-sm text-muted-foreground"
                  >
                    {field.description}
                    {field.suggestion && (
                      <span className="block">
                        Suggested: {field.suggestion}. Other values are accepted
                        when supported.
                      </span>
                    )}
                  </p>
                )}
                {field.allowed.format === 'effort:bias comma list' ? (
                  <div
                    className={
                      compact
                        ? 'grid grid-cols-3 gap-2'
                        : 'flex flex-wrap gap-3'
                    }
                  >
                    {effortParts(field.defaultValue, input.value).map(
                      ([effort, bias], effortIndex, parts) => {
                        return (
                          <div key={effort} className="space-y-1">
                            <Label htmlFor={`${inputId}-${effort}`}>
                              {effort}
                            </Label>
                            <Input
                              id={`${inputId}-${effort}`}
                              type="number"
                              step="any"
                              min={field.suggestedMinimum}
                              max={field.suggestedMaximum}
                              disabled={disabled}
                              value={bias ?? ''}
                              onChange={(event) => {
                                parts[effortIndex] = [
                                  effort,
                                  event.target.value,
                                ]
                                input.onChange(
                                  parts
                                    .map(
                                      ([effort, bias]) => `${effort}:${bias}`,
                                    )
                                    .join(','),
                                )
                              }}
                            />
                            {compact &&
                              field.suggestedMinimum !== undefined &&
                              field.suggestedMaximum !== undefined && (
                                <input
                                  type="range"
                                  aria-label={`${field.label} ${effort} suggested range`}
                                  className="w-full accent-primary"
                                  min={field.suggestedMinimum}
                                  max={field.suggestedMaximum}
                                  step="0.1"
                                  value={Number(bias)}
                                  disabled={disabled}
                                  onChange={(event) => {
                                    parts[effortIndex] = [
                                      effort,
                                      event.target.value,
                                    ]
                                    input.onChange(
                                      parts
                                        .map(
                                          ([name, value]) => `${name}:${value}`,
                                        )
                                        .join(','),
                                    )
                                  }}
                                />
                              )}
                          </div>
                        )
                      },
                    )}
                  </div>
                ) : field.control === 'switch' ? (
                  <Switch
                    id={inputId}
                    name={input.name}
                    ref={input.ref}
                    checked={input.value === field.booleanValues?.[1]}
                    disabled={disabled}
                    onBlur={input.onBlur}
                    onCheckedChange={(checked) =>
                      input.onChange(field.booleanValues?.[checked ? 1 : 0])
                    }
                    aria-describedby={descriptionId}
                    aria-invalid={fieldState.invalid}
                  />
                ) : field.control === 'select' ? (
                  // A native select preserves boolean and numeric enum values by indexing allowed.values.
                  // Radix Select only accepts strings; no CLI values are coerced into invented string values.
                  <select
                    id={inputId}
                    name={input.name}
                    ref={input.ref}
                    onBlur={input.onBlur}
                    value={
                      field.allowed.values?.findIndex(
                        (value) => value === input.value,
                      ) ?? -1
                    }
                    onChange={(event) =>
                      input.onChange(
                        field.allowed.values?.[Number(event.target.value)],
                      )
                    }
                    aria-describedby={descriptionId}
                    aria-invalid={fieldState.invalid}
                    className="min-h-9 max-w-full rounded-md border bg-background px-2 text-sm"
                  >
                    {!field.allowed.values?.some(
                      (value) => value === input.value,
                    ) && (
                      <option value={-1} disabled>
                        {input.value == null
                          ? 'Choose a value'
                          : (field.valueLabels?.[String(input.value)] ??
                            String(input.value))}
                      </option>
                    )}
                    {field.allowed.values?.map((value, optionIndex) => (
                      <option key={optionIndex} value={optionIndex}>
                        {field.valueLabels?.[String(value)] ?? String(value)}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    className={compact ? 'max-w-40' : undefined}
                    id={inputId}
                    name={input.name}
                    ref={input.ref}
                    inputMode={
                      field.control === 'number' ||
                      (input.value !== '' &&
                        typeof input.value === 'string' &&
                        Number.isFinite(Number(input.value)))
                        ? 'decimal'
                        : undefined
                    }
                    onBlur={input.onBlur}
                    type={
                      (numericInput || field.control === 'number') &&
                      !(
                        typeof input.value === 'string' &&
                        /^-?\d+$/.test(input.value) &&
                        !Number.isSafeInteger(Number(input.value))
                      ) &&
                      !field.valueLabels?.[String(input.value)]
                        ? 'number'
                        : 'text'
                    }
                    placeholder={numericInput ? defaultHint : undefined}
                    value={
                      typeof input.value === 'string' ||
                      typeof input.value === 'number'
                        ? (field.valueLabels?.[String(input.value)] ??
                          input.value)
                        : ''
                    }
                    onChange={(event) =>
                      input.onChange(
                        (() => {
                          const raw =
                            Object.entries(field.valueLabels ?? {}).find(
                              ([, label]) => label === event.target.value,
                            )?.[0] ?? event.target.value
                          return field.control === 'number'
                            ? raw === ''
                              ? NaN
                              : field.allowed.type === 'integer' &&
                                  /^-?\d+$/.test(raw) &&
                                  !Number.isSafeInteger(Number(raw))
                                ? raw
                                : Number(raw)
                            : raw
                        })(),
                      )
                    }
                    min={field.allowed.minimum}
                    max={field.allowed.maximum}
                    step={field.allowed.type === 'integer' ? 1 : 'any'}
                    aria-describedby={descriptionId}
                    aria-invalid={fieldState.invalid}
                  />
                )}
                {(sliders || field.range) &&
                  field.control === 'number' &&
                  field.range && (
                    <div className="space-y-1">
                      <input
                        type="range"
                        aria-label={`${field.label} suggested range`}
                        aria-describedby={descriptionId}
                        className="w-full accent-primary"
                        min={field.range.minimum}
                        max={field.range.maximum}
                        step="any"
                        value={
                          typeof input.value === 'number'
                            ? input.value
                            : field.range.minimum
                        }
                        disabled={
                          disabled ||
                          typeof input.value !== 'number' ||
                          input.value < field.range.minimum ||
                          input.value > field.range.maximum
                        }
                        onChange={(event) =>
                          input.onChange(event.target.valueAsNumber)
                        }
                      />
                      <div className="flex justify-between gap-4 text-xs text-muted-foreground">
                        <span>
                          Suggested minimum{' '}
                          <span className="tabular-nums">
                            {field.range.minimum}
                          </span>
                        </span>
                        <span>
                          Suggested maximum{' '}
                          <span className="tabular-nums">
                            {field.range.maximum}
                          </span>
                        </span>
                      </div>
                    </div>
                  )}
                {fieldState.error && (
                  <ErrorNote className="text-sm text-destructive">
                    <span>{fieldState.error.message}</span>
                  </ErrorNote>
                )}
              </div>
            )}
          />
        )
      })}
    </>
  )
}
