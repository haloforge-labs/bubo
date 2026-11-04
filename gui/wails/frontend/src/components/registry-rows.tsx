import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { Link } from '@tanstack/react-router'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import {
  bytesSetting,
  RegistryField,
  SettingAbout,
} from '@/components/registry-form'
import { SettingRow } from '@/components/setting-row'
import { Button } from '@/components/ui/button'
import type { View } from '@/lib/bindings'
import { guiError } from '@/lib/gui-errors'
import {
  aboutParagraphs,
  defaultWords,
  settingHint,
  unavailableHint,
  unitOf,
} from '@/lib/presentation'
import { fieldFromRow, schemaFromRegistry } from '@/lib/registry'
import type { RegistryValues } from '@/lib/registry'
import {
  prepareRow,
  samplingRows,
  type RegistryEntry,
} from '@/lib/registry-entry'
import { markerListUnset } from '@/lib/server-values'

type Row = View['switches'][number]

// The form of a set of registry rows. Server setup and Settings both edit
// settings through this one engine: the same rows, the same unset / reset /
// read-only rules, the same patch the supervisor takes. A page supplies the rows
// and decides where the Save button is.
export function useRegistryForm(view: View, rows: Row[]) {
  const entries = useMemo(
    () => rows.map((row) => prepareRow(row, view)),
    [rows, view],
  )
  const [clears, setClears] = useState<string[]>([])
  const [editing, setEditing] = useState<string[]>([])
  const [error, setError] = useState('')
  const editable = entries.filter(
    (entry) =>
      !entry.blocked &&
      !entry.modelRoute &&
      entry.converted &&
      !entry.error &&
      (entry.value !== undefined || editing.includes(entry.row.name)),
  )
  const defaults = Object.fromEntries(
    editable
      .filter((entry) => entry.value !== undefined)
      .map((entry) => [entry.row.name, entry.value]),
  ) as RegistryValues
  const form = useForm<Record<string, unknown>, unknown, RegistryValues>({
    resolver: (values, context, options) =>
      zodResolver(
        schemaFromRegistry(
          editable
            .filter(
              (entry) =>
                !clears.includes(entry.row.name) &&
                values[entry.row.name] !== defaults[entry.row.name],
            )
            .map((entry) => entry.converted!),
        ).passthrough(),
      )(values, context, options),
    defaultValues: defaults,
  })
  const signature = JSON.stringify(defaults)
  useEffect(() => {
    // Polls refresh clean values and keep the user's unsaved fields.
    form.reset(JSON.parse(signature) as RegistryValues, {
      keepDirtyValues: true,
    })
  }, [signature, form])
  const changed = [
    ...new Set([...Object.keys(form.formState.dirtyFields), ...clears]),
  ]
  function patch() {
    const values = form.getValues()
    const clearNames = new Set(clears)
    for (const name of changed) {
      if (clears.includes(name)) continue
      if (name === '--seed' && String(values[name] ?? '').trim() === '') {
        clearNames.add(name)
        continue
      }
      const entry = entries.find((item) => item.row.name === name)
      const field = entry?.converted && fieldFromRow(entry.converted)
      if (
        !entry ||
        !field ||
        field.control === 'unsupported' ||
        !field.schema.safeParse(values[name]).success
      ) {
        setError(`Check ${entry?.converted?.label ?? name}.`)
        return undefined
      }
      if (markerListUnset(name, values[name])) clearNames.add(name)
    }
    setError('')
    return {
      switch_values: changed
        .filter((name) => !clearNames.has(name))
        .map((name) => ({ name, value: String(values[name]) })),
      clear: [...clearNames].map((name) => {
        const state = view.switch_overrides.find((row) => row.name === name)
        return state?.write_via === 'preference'
          ? state.preference_field || name
          : name
      }),
    }
  }
  return {
    entries,
    form,
    clears,
    setClears,
    editing,
    setEditing,
    changed,
    dirty: changed.length > 0,
    error,
    patch,
    // The saved values become the clean ones.
    reset() {
      form.reset(form.getValues())
      setClears([])
    },
    // Back to what was saved.
    discard() {
      form.reset(defaults)
      setClears([])
    },
  }
}

export type RegistryForm = ReturnType<typeof useRegistryForm>

// What the CLI says about the saved model's use of a setting (Prefill mode,
// In-engine abliteration, Q8): the GUI shows the setting off, disabled, with a
// short phrase, and keeps the CLI's own reason in the About text. The Go side
// (capabilitySwitches) drops the same values before serve.
function capabilityOf(
  view: View,
  name: string,
  picked: ModelCapabilities | null | undefined,
) {
  const capabilities =
    picked === undefined
      ? (view.saved_estimate?.capabilities ??
        view.saved_estimate?.memory_estimate?.capabilities)
      : picked
  return name === '--prefill-mode'
    ? capabilities?.prefill_mode_fast
    : name === '--refusal-direction'
      ? capabilities?.refusal_direction
      : name === '--kv-cache-type'
        ? capabilities?.kv_cache_q8
        : undefined
}

type ModelCapabilities = NonNullable<
  NonNullable<View['saved_estimate']>['capabilities']
>

// Whose facts the rows show. Unset: the saved model's. A page that lets the
// user pick another model before saving supplies that model's facts (null while
// they are not known yet), so the rows and the picker always describe one model.
export const CapabilitySource = createContext<
  ModelCapabilities | null | undefined
>(undefined)

// One setting row, whatever its state: a model choice, read-only, unset, or an
// editor. The one place that decides how a setting looks and what Reset does.
export function RegistryRow({
  entry,
  ctrl,
  view,
  disabled,
}: {
  entry: RegistryEntry
  ctrl: RegistryForm
  view: View
  disabled?: boolean
}) {
  const { row, state, converted, value } = entry
  const picked = useContext(CapabilitySource)
  const label = converted?.label ?? row.label ?? row.name
  const paragraphs = aboutParagraphs(row)
  const defaultText = defaultWords(row)
  const about = (
    <SettingAbout
      paragraphs={entry.error ? [entry.error, ...paragraphs] : paragraphs}
      defaultText={defaultText}
    />
  )
  const resetNow = ctrl.clears.includes(row.name)
  const changedNow =
    state?.saved != null || ctrl.changed.includes(row.name) || resetNow
  // The one Reset: it clears this setting's saved value (Undo takes it back).
  const resetButton = changedNow ? (
    <Button
      type="button"
      variant="link"
      className="ml-2 h-auto px-1 py-0 text-xs"
      aria-label={`${resetNow ? 'Undo reset of' : 'Reset'} ${label}${resetNow ? '' : ' to default'}`}
      onClick={() =>
        ctrl.setClears((current) =>
          resetNow
            ? current.filter((name) => name !== row.name)
            : [...current, row.name],
        )
      }
    >
      {resetNow ? 'Undo' : 'Reset'}
    </Button>
  ) : undefined
  if (entry.modelRoute)
    return (
      <SettingRow
        name={row.name}
        label={label}
        about={about}
        control={
          <Link to="/models" className="underline underline-offset-4">
            Choose a model
          </Link>
        }
      />
    )
  if (entry.blocked || entry.error)
    return (
      <SettingRow
        name={row.name}
        label={label}
        about={about}
        wide
        reset={entry.error ? resetButton : undefined}
        control={
          <span
            data-code={entry.blocked ? 'setting_read_only' : 'not_reported'}
            className="text-sm text-muted-foreground"
          >
            {entry.error
              ? "This setting can't be changed here."
              : guiError({ code: 'setting_read_only', remedy: '' }).message}
          </span>
        }
      />
    )
  const hasEdit = ctrl.editing.includes(row.name)
  if (entry.hidden && !hasEdit)
    return (
      <SettingRow
        name={row.name}
        label={label}
        about={about}
        control={
          <>
            <span className="text-sm text-muted-foreground">Value hidden</span>
            <Button
              type="button"
              variant="link"
              className="h-auto px-1 py-0 text-xs"
              disabled={disabled}
              aria-label={`Replace ${label}`}
              onClick={() => {
                ctrl.setEditing((current) => [...current, row.name])
                ctrl.form.setValue(row.name, '')
              }}
            >
              Replace
            </Button>
          </>
        }
      />
    )
  if (value === undefined && !hasEdit)
    return (
      <SettingRow
        name={row.name}
        label={label}
        about={about}
        hint={settingHint(row.name)}
        control={
          <Button
            type="button"
            variant="outline"
            className="w-full justify-start"
            disabled={disabled}
            aria-label={`Change ${label}`}
            onClick={() => {
              ctrl.setEditing((current) => [...current, row.name])
              const initial = converted?.defaultValue
              if (
                typeof initial === 'string' ||
                typeof initial === 'number' ||
                typeof initial === 'boolean'
              )
                ctrl.form.setValue(
                  row.name,
                  markerListUnset(row.name, initial) ? '' : initial,
                )
            }}
          >
            <span className="text-muted-foreground">Default</span>
          </Button>
        }
      />
    )
  const field = converted && fieldFromRow(converted)
  if (!field || field.control === 'unsupported') return null
  const capability = capabilityOf(view, row.name, picked)
  const unavailable = capability?.available === false
  const restart =
    state && ['pending', 'unknown_until_restart'].includes(state.restart_state)
  return (
    <RegistryField
      field={field}
      form={ctrl.form}
      disabled={disabled || resetNow || unavailable}
      showDefault={unavailable}
      numericInput={samplingRows.includes(row.name)}
      defaultText={row.name === '--seed' ? 'Random' : defaultText}
      placeholder={
        row.name === '--seed'
          ? 'Random'
          : typeof converted?.defaultValue === 'number'
            ? String(converted.defaultValue)
            : 'Default'
      }
      aboutParagraphs={
        unavailable && capability?.reason
          ? [capability.reason, ...paragraphs]
          : paragraphs
      }
      bytes={
        row.name.endsWith('-bytes') && typeof value === 'number'
          ? bytesSetting(row.default_text, value)
          : undefined
      }
      unit={unitOf(row.name)}
      hint={
        unavailable
          ? unavailableHint(capability?.reason ?? '')
          : resetNow
            ? 'Default after saving'
            : restart
              ? 'After restart'
              : settingHint(row.name)
      }
      // An unusable value the app still remembers can be cleared too.
      reset={resetButton}
    />
  )
}
