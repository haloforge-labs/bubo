import { ErrorNote } from '@/components/error-note'
import { useId, type ReactNode, type Ref } from 'react'
import {
  Controller,
  type UseFormReturn,
  type FieldValues,
} from 'react-hook-form'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { AboutPopup } from '@/components/about-popup'
import { SettingRow } from '@/components/setting-row'
import { fieldFromRow } from '@/lib/registry'
import { valueWord } from '@/lib/presentation'

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

// The effort:bias list with one entry changed, as the text the CLI takes.
function effortText(
  parts: readonly (readonly [string, string])[],
  index: number,
  effort: string,
  bias: string,
) {
  return parts
    .map((part, at) => (at === index ? ([effort, bias] as const) : part))
    .map(([name, value]) => `${name}:${value}`)
    .join(',')
}

export type Field = Exclude<
  ReturnType<typeof fieldFromRow>,
  { control: 'unsupported' }
>

const GiB = 1024 ** 3
const MiB = 1024 ** 2

// A byte setting is typed in the unit a person reads: MiB below 1 GiB, else GiB.
// Zero means Automatic where the registry says so.
export interface BytesSetting {
  auto: boolean
  unitBytes: number
}

export function bytesSetting(
  defaultText: string,
  current: number | undefined,
): BytesSetting {
  return {
    auto: /^0\s*=\s*auto/i.test(defaultText),
    unitBytes: (current ?? 0) < GiB ? MiB : GiB,
  }
}

// The unit stays visible, an Automatic (empty) field included.
function bytesUnit(setting: BytesSetting) {
  return setting.unitBytes === MiB ? 'MiB' : 'GiB'
}

function FieldSelect({
  field,
  name,
  value: current,
  ref,
  onBlur,
  onChange,
  disabled,
  descriptionId,
  invalid,
  id,
}: {
  field: Field
  name: string
  value: unknown
  ref: Ref<HTMLSelectElement>
  onBlur: () => void
  onChange: (value: unknown) => void
  disabled?: boolean
  descriptionId: string
  invalid: boolean
  id: string
}) {
  return (
    // A native select preserves boolean and numeric enum values by indexing allowed.values.
    // Radix Select only accepts strings; no CLI values are coerced into invented string values.
    <select
      id={id}
      name={name}
      ref={ref}
      disabled={disabled}
      onBlur={onBlur}
      value={
        field.allowed.values?.findIndex((value) => value === current) ?? -1
      }
      onChange={(event) =>
        onChange(field.allowed.values?.[Number(event.target.value)])
      }
      aria-describedby={descriptionId}
      aria-invalid={invalid}
      className="h-9 w-full min-w-0 rounded-md border bg-background px-2 text-sm"
    >
      {!field.allowed.values?.some((value) => value === current) && (
        <option value={-1} disabled>
          {current == null
            ? 'Choose a value'
            : (field.valueLabels?.[String(current)] ?? String(current))}
        </option>
      )}
      {field.allowed.values?.map((value, optionIndex) => (
        <option key={optionIndex} value={optionIndex}>
          {field.valueLabels?.[String(value)] ?? String(value)}
        </option>
      ))}
    </select>
  )
}

function FieldControl({
  field,
  name,
  value,
  ref,
  onBlur,
  onChange,
  invalid,
  disabled,
  numericInput,
  placeholder,
  bytes,
  inputId,
  descriptionId,
}: {
  field: Field
  name: string
  value: unknown
  ref: Ref<HTMLInputElement & HTMLSelectElement & HTMLButtonElement>
  onBlur: () => void
  onChange: (value: unknown) => void
  invalid: boolean
  disabled?: boolean
  numericInput?: boolean
  placeholder?: string
  bytes?: BytesSetting
  inputId: string
  descriptionId: string
}) {
  if (bytes)
    return (
      <Input
        id={inputId}
        name={name}
        type="number"
        min="0"
        step="any"
        disabled={disabled}
        placeholder={bytes.auto ? 'Automatic' : undefined}
        value={
          bytes.auto && Number(value) === 0
            ? ''
            : Number(value ?? 0) / bytes.unitBytes
        }
        onBlur={onBlur}
        onChange={(event) =>
          onChange(
            event.target.value === ''
              ? 0
              : Number(event.target.value) * bytes.unitBytes,
          )
        }
        aria-describedby={descriptionId}
        aria-invalid={invalid}
      />
    )
  if (field.control === 'switch')
    return (
      <Switch
        id={inputId}
        name={name}
        ref={ref}
        checked={value === field.booleanValues?.[1]}
        disabled={disabled}
        onBlur={onBlur}
        onCheckedChange={(checked) =>
          onChange(field.booleanValues?.[checked ? 1 : 0])
        }
        aria-describedby={descriptionId}
        aria-invalid={invalid}
      />
    )
  if (field.control === 'select')
    return (
      <FieldSelect
        field={field}
        name={name}
        value={value}
        ref={ref}
        onBlur={onBlur}
        onChange={onChange}
        disabled={disabled}
        descriptionId={descriptionId}
        invalid={invalid}
        id={inputId}
      />
    )
  return (
    <Input
      id={inputId}
      name={name}
      ref={ref}
      inputMode={
        field.control === 'number' ||
        (value !== '' &&
          typeof value === 'string' &&
          Number.isFinite(Number(value)))
          ? 'decimal'
          : undefined
      }
      onBlur={onBlur}
      type={
        (numericInput || field.control === 'number') &&
        !(
          typeof value === 'string' &&
          /^-?\d+$/.test(value) &&
          !Number.isSafeInteger(Number(value))
        ) &&
        !field.valueLabels?.[String(value)]
          ? 'number'
          : 'text'
      }
      placeholder={numericInput ? placeholder : undefined}
      value={
        typeof value === 'string' || typeof value === 'number'
          ? (field.valueLabels?.[String(value)] ?? value)
          : ''
      }
      onChange={(event) =>
        onChange(
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
      aria-invalid={invalid}
    />
  )
}

// The About text of a row: the explanation, then the default and the range. It
// sits behind the ⓘ of the row, so the row itself stays one line. The range is
// said once: the registry's suggestion, else the range the input enforces.
export function SettingAbout({
  paragraphs,
  defaultText,
  range,
  suggestion,
  id,
}: {
  paragraphs: readonly string[]
  defaultText?: string
  range?: { minimum: number; maximum: number }
  suggestion?: string
  id?: string
}) {
  return (
    <>
      {paragraphs.map((text, index) => (
        <p key={index} id={index === 0 ? id : undefined}>
          {text}
        </p>
      ))}
      {defaultText && <p>Default: {defaultText}</p>}
      {range && !suggestion && (
        <p>
          Suggested range: {range.minimum} to {range.maximum}.
        </p>
      )}
      {suggestion && <p>Suggested: {suggestion}. Other values are accepted.</p>}
    </>
  )
}

// The effort map is one setting with one value per reasoning effort. It sits in
// the grid like any other: a caption line (name, ⓘ, Reset) and one ordinary cell
// per effort. The server reads High and Max as Extra high and Minimal as Low, so
// the caption says it; there is no separate cell that could not be set.
function EffortMap({
  field,
  value,
  onChange,
  disabled,
  reset,
  about,
  error,
}: {
  field: Field
  value: unknown
  onChange: (value: unknown) => void
  disabled?: boolean
  reset?: ReactNode
  about: ReactNode
  error?: ReactNode
}) {
  const id = useId()
  const parts = effortParts(field.defaultValue, value)
  const levels = parts.map(([effort]) => effort)
  const reads = [
    levels.includes('low') && 'Minimal reads as Low',
    levels.includes('xhigh') && 'High and Max read as Extra high',
  ].filter(Boolean)
  return (
    <>
      <div
        data-setting={field.name}
        role="group"
        aria-label={field.label}
        className="col-span-full flex flex-wrap items-center gap-x-2 gap-y-0.5"
      >
        <Label>{field.label}</Label>
        <AboutPopup label={field.label} className="order-last">
          {about}
        </AboutPopup>
        {reads.length > 0 && (
          <span className="text-xs text-muted-foreground">
            {reads.join('; ')}
          </span>
        )}
        {reset}
      </div>
      {parts.map(([effort, bias], effortIndex) => (
        <SettingRow
          key={effort}
          label={valueWord(field.name, effort)}
          htmlFor={`${id}-${effort}`}
          control={
            <Input
              id={`${id}-${effort}`}
              type="number"
              step="any"
              min={field.suggestedMinimum}
              max={field.suggestedMaximum}
              disabled={disabled}
              value={bias ?? ''}
              onChange={(event) =>
                onChange(
                  effortText(parts, effortIndex, effort, event.target.value),
                )
              }
            />
          }
        />
      ))}
      {error}
    </>
  )
}

// One edited setting: the shared row with the control of its registry field.
export function RegistryField<T extends FieldValues>({
  field,
  form,
  disabled,
  hint,
  aboutParagraphs,
  defaultText,
  reset,
  showDefault,
  numericInput,
  placeholder,
  bytes,
  unit,
}: {
  field: Field
  form: UseFormReturn<Record<string, unknown>, unknown, T>
  disabled?: boolean
  hint?: ReactNode
  aboutParagraphs: readonly string[]
  defaultText: string
  reset?: ReactNode
  // Shows the default value (a disabled row for a setting the model cannot use).
  showDefault?: boolean
  numericInput?: boolean
  // What an empty number field says: the numeric default, else "Default".
  placeholder?: string
  bytes?: BytesSetting
  unit?: string
}) {
  const id = useId()
  const inputId = `${id}-field`
  const descriptionId = `${inputId}-description`
  const effort = field.allowed.format === 'effort:bias comma list'
  // A choice with a long option (over 24 characters) grows into the hint's
  // width; a number or a short choice keeps the one control width.
  const grow =
    field.control === 'select' &&
    Math.max(
      0,
      ...(field.allowed.values ?? []).map(
        (value) => (field.valueLabels?.[String(value)] ?? String(value)).length,
      ),
    ) > 24
  const about = (
    <SettingAbout
      id={descriptionId}
      paragraphs={aboutParagraphs}
      defaultText={defaultText}
      range={field.range}
      suggestion={field.suggestion}
    />
  )
  if (effort)
    return (
      <Controller
        name={field.name}
        control={form.control}
        render={({ field: input, fieldState }) => (
          <EffortMap
            field={field}
            value={showDefault ? field.defaultValue : input.value}
            onChange={input.onChange}
            disabled={disabled}
            reset={reset}
            about={about}
            error={
              fieldState.error && (
                <ErrorNote className="col-span-full text-sm text-destructive">
                  <span>{fieldState.error.message}</span>
                </ErrorNote>
              )
            }
          />
        )}
      />
    )
  return (
    <Controller
      name={field.name}
      control={form.control}
      render={({ field: input, fieldState }) => (
        <SettingRow
          name={field.name}
          label={field.label}
          htmlFor={inputId}
          wide={grow}
          unit={bytes ? bytesUnit(bytes) : unit}
          hint={hint}
          reset={reset}
          about={about}
          control={
            <>
              <div className="min-w-0 flex-1">
                <FieldControl
                  field={field}
                  name={input.name}
                  value={showDefault ? field.defaultValue : input.value}
                  ref={input.ref}
                  onBlur={input.onBlur}
                  onChange={input.onChange}
                  invalid={fieldState.invalid}
                  disabled={disabled}
                  numericInput={numericInput}
                  placeholder={placeholder}
                  bytes={bytes}
                  inputId={inputId}
                  descriptionId={descriptionId}
                />
              </div>
              {field.control === 'number' && field.range && !bytes && (
                <input
                  type="range"
                  aria-label={`${field.label} suggested range`}
                  aria-describedby={descriptionId}
                  className="min-w-16 flex-1 accent-primary"
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
              )}
            </>
          }
          error={
            fieldState.error && (
              <ErrorNote className="text-sm text-destructive">
                <span>{fieldState.error.message}</span>
              </ErrorNote>
            )
          }
        />
      )}
    />
  )
}
