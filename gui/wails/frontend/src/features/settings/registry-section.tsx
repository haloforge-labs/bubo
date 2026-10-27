import { ErrorNote } from '@/components/error-note'
import { useEffect, useMemo, useState } from 'react'
import { Link } from '@tanstack/react-router'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { RegistryFields } from '@/components/registry-form'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { actions, requireActionSuccess, type View } from '@/lib/bindings'
import {
  schemaFromRegistry,
  fieldFromRow,
  UnsupportedRegistryError,
  type RegistryValues,
} from '@/lib/registry'
import { guiError } from '@/lib/gui-errors'
import { bytes } from '@/lib/format'
import {
  effectiveValue,
  groupRows,
  groupLabel,
  registryRow,
  registryDefault,
} from './registry'
import { serverRowSection } from '@/lib/server-settings'
import { markerListUnset, unsetSettingText } from '@/lib/server-values'
import {
  guiRegistryPresentation,
  packageManagedSetting,
} from '@/lib/registry-presentation'
import { profileAliases } from '@/generated/profile-aliases'
import { CurrentGPU } from './current-gpu'
import { NPUPrefill } from './npu-prefill'

function prepare(row: View['switches'][number], view: View) {
  const state = view.switch_overrides?.find((entry) => entry.name === row.name)
  const modelRoute =
    state?.write_via === 'preference' &&
    (state.preference_field === 'model_path' ||
      state.preference_field === 'manifest_path')
  const blocked =
    !state ||
    (!!state.write_block &&
      !(
        state.write_via === 'preference' &&
        state.write_block === 'supervisor_owned'
      ))
  try {
    const converted = registryRow(row)
    const field = fieldFromRow(converted)
    if (field.booleanValues?.every((value) => value !== undefined)) {
      converted.valueLabels = {
        ...converted.valueLabels,
        [String(field.booleanValues[0])]: 'Off',
        [String(field.booleanValues[1])]: 'On',
      }
    }
    converted.purpose = row.note || row.purpose
    if (row.name === '--profile') {
      const profiles = view.profiles.filter((profile) => profile.gui)
      converted.allowed = {
        type: 'enum',
        values: profiles.map((profile) => profile.name),
      }
      converted.valueLabels = Object.fromEntries(
        profiles.map((profile) => [profile.name, profile.label]),
      )
    }
    schemaFromRegistry([converted])
    let value =
      effectiveValue(view, row.name) ??
      (row.name === '--profile' ? registryDefault(row) : undefined)
    if (row.name === '--profile' && typeof value === 'string')
      value = profileAliases[value] ?? value
    return {
      row,
      state,
      converted,
      value:
        value ??
        (converted.allowed.format === 'effort:bias comma list'
          ? registryDefault(row)
          : undefined),
      modelRoute,
      blocked,
      error: '',
    }
  } catch (cause) {
    if (!(cause instanceof UnsupportedRegistryError)) throw cause
    return { row, state, modelRoute, blocked, error: cause.message }
  }
}

export function RegistrySection({
  view,
  audience,
  group,
  sliders,
  onDirtyChange,
  includeUserRemainder,
}: {
  view: View
  audience: string
  group?: string
  sliders?: boolean
  onDirtyChange?: (dirty: boolean) => void
  includeUserRemainder?: boolean
}) {
  const [filter, setFilter] = useState('')
  const [debugFilter, setDebugFilter] = useState('')
  const [clears, setClears] = useState<string[]>([])
  const [editingUnset, setEditingUnset] = useState<string[]>([])
  const save = useActionTask()
  const entries = useMemo(
    () =>
      view.switches
        .map(guiRegistryPresentation)
        .filter(
          (row) =>
            row.gui_control &&
            (row.audience === audience ||
              (includeUserRemainder &&
                row.audience === 'user' &&
                row.name !== '--api-key' &&
                !serverRowSection(row)) ||
              (audience === 'advanced' && row.audience === 'debug')) &&
            (!group || row.group === group),
        )
        .map((row) => prepare(row, view)),
    [view, audience, group, includeUserRemainder],
  )
  const editable = entries.filter(
    (entry) =>
      !entry.blocked &&
      !entry.modelRoute &&
      entry.converted &&
      !entry.error &&
      (entry.value !== undefined || editingUnset.includes(entry.row.name)),
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
    // Polls refresh clean values and retain the user's unsaved fields.
    form.reset(JSON.parse(signature) as RegistryValues, {
      keepDirtyValues: true,
    })
  }, [signature, form])
  const changed = [
    ...new Set([...Object.keys(form.formState.dirtyFields), ...clears]),
  ]
  const count = changed.length
  useEffect(() => {
    onDirtyChange?.(count > 0)
  }, [count, onDirtyChange])
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange])

  function renderGroups(sectionAudience: string, query: string) {
    const groups = groupRows(
      entries.map((entry) =>
        includeUserRemainder && entry.row.audience === 'user'
          ? { ...entry.row, audience: 'advanced' }
          : entry.row,
      ),
      sectionAudience,
      query,
    )
    return (
      <>
        {Object.entries(groups).map(([name, rows]) => (
          <section key={name} aria-label={name} className="space-y-3">
            <h3 className="text-xl">{groupLabel(name)}</h3>
            {name === 'Hardware' &&
              sectionAudience === 'advanced' &&
              view.serving && (
                <>
                  <NPUPrefill runtime={view.npu_prefill_runtime} />
                  <CurrentGPU gpu={view.gpu} />
                </>
              )}
            <div className="grid grid-cols-1 items-start gap-x-5 gap-y-3 min-[560px]:grid-cols-2 min-[900px]:grid-cols-3">
              {rows.map((row) => {
                const entry = entries.find(
                  (entry) => entry.row.name === row.name,
                )!
                const { state, converted, value, blocked, modelRoute, error } =
                  entry
                const reset = clears.includes(row.name)
                return (
                  <div
                    key={row.name}
                    className="min-w-0 space-y-2 border-t py-3"
                    data-setting={row.name}
                  >
                    {packageManagedSetting(row.name) && (
                      <p className="text-xs text-muted-foreground">
                        Supplied by the model package
                      </p>
                    )}
                    {modelRoute ? (
                      <>
                        <p>{row.label ?? row.name}</p>
                        <p className="text-sm text-muted-foreground">
                          Choose a downloaded model or import one on Models.
                        </p>
                        <Link
                          to="/models"
                          className="underline underline-offset-4"
                        >
                          Choose a model
                        </Link>
                      </>
                    ) : (
                      <>
                        {state &&
                          ['pending', 'unknown_until_restart'].includes(
                            state.restart_state,
                          ) && (
                            <Badge variant="outline">
                              used after restart
                              {state.restart_state === 'unknown_until_restart'
                                ? ' · effective value not yet reported'
                                : ''}
                            </Badge>
                          )}
                        {state?.restart_state === 'not_running' && (
                          <Badge variant="outline">used on next start</Badge>
                        )}
                        {blocked || error ? (
                          <div
                            data-code={
                              blocked ? 'setting_read_only' : 'not_reported'
                            }
                          >
                            <p>{row.label ?? row.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {error
                                ? "This setting can't be changed here."
                                : guiError({
                                    code: 'setting_read_only',
                                    remedy: '',
                                  }).message}
                            </p>
                            {error && (
                              <details>
                                <summary>Details</summary>
                                <p>{error}</p>
                              </details>
                            )}
                          </div>
                        ) : value === undefined &&
                          !editingUnset.includes(row.name) ? (
                          <div>
                            <p>{row.label ?? row.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {unsetSettingText(row.name)}
                            </p>
                            <Button
                              type="button"
                              variant="outline"
                              onClick={() =>
                                setEditingUnset((current) => [
                                  ...current,
                                  row.name,
                                ])
                              }
                            >
                              Set {row.label ?? row.name}
                            </Button>
                          </div>
                        ) : row.name.endsWith('-bytes') &&
                          typeof value === 'number' ? (
                          <fieldset disabled={save.pending || reset}>
                            <label
                              htmlFor={`setting-${row.name}`}
                              className="block font-medium"
                            >
                              {row.label ?? row.name}
                            </label>
                            <div className="flex items-center gap-2">
                              <Input
                                id={`setting-${row.name}`}
                                type="number"
                                min="0"
                                step="any"
                                className="max-w-36"
                                placeholder={
                                  /^0\s*=\s*auto/i.test(row.default_text)
                                    ? 'Automatic'
                                    : undefined
                                }
                                value={
                                  /^0\s*=\s*auto/i.test(row.default_text) &&
                                  Number(form.watch(row.name) ?? value) === 0
                                    ? ''
                                    : Number(form.watch(row.name) ?? value) /
                                      (value < 1024 ** 3
                                        ? 1024 ** 2
                                        : 1024 ** 3)
                                }
                                onChange={(event) =>
                                  form.setValue(
                                    row.name,
                                    Number(event.target.value) *
                                      (value < 1024 ** 3
                                        ? 1024 ** 2
                                        : 1024 ** 3),
                                    { shouldDirty: true, shouldValidate: true },
                                  )
                                }
                              />
                              <span>
                                {/^0\s*=\s*auto/i.test(row.default_text) &&
                                Number(form.watch(row.name) ?? value) === 0
                                  ? ''
                                  : value < 1024 ** 3
                                    ? 'MiB'
                                    : 'GiB'}
                              </span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Default: {row.default_text}
                            </p>
                            <details className="text-xs text-muted-foreground">
                              <summary
                                role="button"
                                aria-label={`About ${row.label ?? row.name}`}
                                className="cursor-pointer"
                              >
                                About
                              </summary>
                              <p>{row.note || row.purpose}</p>
                            </details>
                            {form.formState.errors[row.name] && (
                              <ErrorNote className="text-sm text-destructive">
                                <span>
                                  {String(
                                    form.formState.errors[row.name]?.message,
                                  )}
                                </span>
                              </ErrorNote>
                            )}
                          </fieldset>
                        ) : (
                          <fieldset disabled={save.pending || reset}>
                            <RegistryFields
                              rows={[converted!]}
                              form={form}
                              sliders={sliders}
                              compact
                              defaultHint={row.default_text}
                              disabled={save.pending || reset}
                            />
                          </fieldset>
                        )}
                        {!blocked && value !== undefined && (
                          <>
                            {(state?.saved != null ||
                              reset ||
                              value === undefined) && (
                              <p className="text-sm text-muted-foreground">
                                {reset
                                  ? 'Default will be used after saving.'
                                  : `${state?.saved != null ? 'Set by you' : 'Current'}: ${row.name.endsWith('-bytes') && typeof value === 'number' ? bytes(value) : (converted?.valueLabels?.[String(value)] ?? String(value))}`}
                              </p>
                            )}
                            {state?.saved != null && (
                              <details className="text-sm text-muted-foreground">
                                <summary>Default details</summary>
                                <p>Default: {row.default_text}</p>
                              </details>
                            )}
                          </>
                        )}
                        {(state?.saved != null ||
                          reset ||
                          converted?.allowed.format ===
                            'effort:bias comma list') && (
                          <Button
                            type="button"
                            variant="outline"
                            disabled={
                              save.pending ||
                              (state?.saved == null &&
                                !form.formState.dirtyFields[row.name])
                            }
                            onClick={() =>
                              setClears((current) =>
                                reset
                                  ? current.filter((name) => name !== row.name)
                                  : [...current, row.name],
                              )
                            }
                          >
                            {reset ? 'Undo reset' : 'Reset to default'}
                          </Button>
                        )}
                      </>
                    )}
                    {(blocked ||
                      error ||
                      (value === undefined &&
                        !editingUnset.includes(row.name))) && (
                      <>
                        <p className="text-xs text-muted-foreground">
                          Default: {row.default_text}
                        </p>
                        <details className="text-xs text-muted-foreground">
                          <summary
                            role="button"
                            aria-label={`About ${row.label ?? row.name}`}
                            className="w-fit cursor-pointer underline underline-offset-2"
                          >
                            About
                          </summary>
                          <p className="pt-2">{row.note || row.purpose}</p>
                        </details>
                      </>
                    )}
                  </div>
                )
              })}
            </div>
          </section>
        ))}
        {!Object.keys(groups).length && <p>No matching settings.</p>}
      </>
    )
  }

  return (
    <form
      className="space-y-5"
      noValidate
      onSubmit={form.handleSubmit(async (values) => {
        const dirty = entries.filter((entry) =>
          changed.includes(entry.row.name),
        )
        await save.run(async () => {
          if (!dirty.length) return
          const markerUnset = (entry: (typeof dirty)[number]) =>
            markerListUnset(entry.row.name, values[entry.row.name])
          const set = dirty
            .filter(
              (entry) =>
                !clears.includes(entry.row.name) && !markerUnset(entry),
            )
            .map((entry) => ({
              name: entry.row.name,
              value: String(values[entry.row.name]),
            }))
          const clear = dirty
            .filter(
              (entry) =>
                clears.includes(entry.row.name) ||
                (markerUnset(entry) && entry.state?.saved != null),
            )
            .map((entry) =>
              entry.state?.write_via === 'preference'
                ? entry.state.preference_field!
                : entry.row.name,
            )
          requireActionSuccess(
            await actions.PatchPreferences({
              set: {},
              clear,
              ...(set.length ? { switch_values: set } : {}),
            }),
          )
          // Clean the entire transaction only after the supervisor accepts it.
          for (const entry of dirty)
            form.resetField(entry.row.name, {
              defaultValue: values[entry.row.name],
            })
          setClears((current) =>
            current.filter(
              (name) => !dirty.some((entry) => entry.row.name === name),
            ),
          )
        })
      })}
    >
      <Input
        aria-label={
          audience === 'advanced'
            ? 'Find a setting'
            : `Filter ${audience} settings`
        }
        placeholder="Find a setting"
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
      />
      {renderGroups(audience, filter)}
      {audience === 'advanced' && (
        <Collapsible className="border-t pt-5">
          <CollapsibleTrigger className="min-h-10 text-xl">
            Debug
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-5 pt-4">
            <p className="text-sm text-muted-foreground">
              These settings are for troubleshooting.
            </p>
            <Input
              aria-label="Filter debug settings"
              placeholder="Filter debug settings…"
              value={debugFilter}
              onChange={(event) => setDebugFilter(event.target.value)}
            />
            {renderGroups('debug', debugFilter)}
          </CollapsibleContent>
        </Collapsible>
      )}
      {count > 0 && (
        <div
          className="sticky bottom-0 z-10 flex flex-wrap items-center gap-3 border-t bg-background py-3"
          aria-label="Save settings"
        >
          <span aria-live="polite" className="text-sm">
            {count} unsaved {count === 1 ? 'change' : 'changes'}
          </span>
          <Button
            type="submit"
            variant="outline"
            disabled={save.pending || count === 0}
          >
            Save changes
          </Button>
          <Button
            type="button"
            variant="ghost"
            disabled={save.pending || count === 0}
            onClick={() => {
              form.reset(defaults)
              setClears([])
            }}
          >
            Discard changes
          </Button>
        </div>
      )}
      <ActionError error={save.error} />
      {save.completed && count === 0 && (
        <p role="status">Request saved. Validated when the server starts.</p>
      )}
    </form>
  )
}
