import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { RegistryFields } from '@/components/registry-form'
import { Button } from '@/components/ui/button'
import type { View } from '@/lib/bindings'
import {
  fieldFromRow,
  type RegistryValues,
  type RegistryValue,
} from '@/lib/registry'
import {
  effectiveValue,
  registryDefault,
  registryRow,
} from '@/lib/registry-view'
import { serverRowSection } from '@/lib/server-settings'
import { markerListUnset, unsetSettingText } from '@/lib/server-values'
import { profileAliases } from '@/generated/profile-aliases'
import { guiRegistryPresentation } from '@/lib/registry-presentation'

const rethinking =
  'Rethinking penalty. Lowers the chance of "wait" and "let me rethink" restarts. It nudges those words and does not cut thinking off. The defaults are recommended: low −2, medium −1, xhigh 0. Use 0 for vanilla. If low or medium thinks too long, go more negative.'

export function useSetupRegistry(view: View) {
  const sampling = ['--temperature', '--top-p', '--top-k']
  const rows = view.switches.filter(
    (row) =>
      serverRowSection(row) &&
      (serverRowSection(row) !== 'Setup' || row.name === '--profile'),
  )
  const [clears, setClears] = useState<string[]>([])
  const [editing, setEditing] = useState<string[]>([])
  const [error, setError] = useState('')
  const defaults = Object.fromEntries(
    rows.flatMap((row) => {
      try {
        let value =
          effectiveValue(view, row.name) ??
          (serverRowSection(row) === 'Network' ||
          row.name === '--profile' ||
          row.name === '--reasoning-effort' ||
          row.allowed.properties?.format?.text === 'effort:bias comma list'
            ? registryDefault(row)
            : undefined)
        if (row.name === '--profile' && typeof value === 'string')
          value = profileAliases[value] ?? value
        if (
          value === undefined &&
          (sampling.includes(row.name) || row.name === '--seed')
        )
          value = ''
        return value === undefined || markerListUnset(row.name, value)
          ? []
          : [[row.name, value]]
      } catch {
        return []
      }
    }),
  ) as RegistryValues
  const signature = JSON.stringify(defaults)
  const form = useForm<Record<string, unknown>, unknown, RegistryValues>({
    defaultValues: defaults,
  })
  useEffect(() => {
    form.reset(JSON.parse(signature) as RegistryValues, {
      keepDirtyValues: true,
    })
  }, [signature, form])
  const changed = [
    ...new Set([...Object.keys(form.formState.dirtyFields), ...clears]),
  ]
  const currentValues = form.watch()
  function summary() {
    const value = (name: string) => {
      const row = rows.find((row) => row.name === name)
      if (!row) return undefined
      const current = clears.includes(name) ? undefined : currentValues[name]
      return current === '' || current == null
        ? (registryDefault(row) ?? row.built_in)
        : current
    }
    const temperature = value('--temperature')
    const parts = [
      [
        'Temperature',
        typeof temperature === 'number' && Number.isInteger(temperature)
          ? temperature.toFixed(1)
          : temperature,
      ],
      ['Top-p', value('--top-p')],
      ['Top-k', value('--top-k')],
    ]
      .filter(([, current]) => current != null)
      .map(([label, current]) => `${label} ${current}`)
    const seed = value('--seed')
    if (seed != null && String(seed) !== '-1' && String(seed) !== '')
      parts.push(`Seed ${seed}`)
    const reasoning = value('--reasoning-effort')
    if (reasoning != null) parts.push(`Reasoning ${reasoning}`)
    return parts.join(' · ')
  }
  function patch() {
    const values = form.getValues()
    const clearNames = new Set(clears)
    for (const name of changed) {
      if (clears.includes(name)) continue
      if (name === '--seed' && String(values[name] ?? '').trim() === '') {
        clearNames.add(name)
        continue
      }
      const row = rows.find((row) => row.name === name)!
      const field = fieldFromRow(registryRow(row))
      if (
        field.control === 'unsupported' ||
        !field.schema.safeParse(values[name]).success
      ) {
        setError(`Check ${row.label ?? row.name}.`)
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
        return state?.preference_field || name
      }),
    }
  }
  function reset() {
    form.reset(form.getValues())
    setClears([])
  }
  function render(section: string) {
    const sectionRows = rows
      .filter((row) => serverRowSection(row) === section)
      .sort((left, right) => {
        const order =
          section === 'Network'
            ? ['--port', '--host', '--served-model-name']
            : section === 'Inference'
              ? [
                  '--temperature',
                  '--top-p',
                  '--top-k',
                  '--seed',
                  '--reasoning-effort',
                  'GUFO_THINK_MARKER_BIAS_BY_EFFORT',
                ]
              : []
        const rank = (name: string) =>
          order.includes(name) ? order.indexOf(name) : order.length
        return rank(left.name) - rank(right.name)
      })
    const renderRow = (sourceRow: View['switches'][number]) => {
      const row = guiRegistryPresentation(sourceRow)
      const state = view.switch_overrides.find((item) => item.name === row.name)
      const blocked =
        !state ||
        (!!state.write_block &&
          !(
            state.write_via === 'preference' &&
            state.write_block === 'supervisor_owned'
          ))
      const unset =
        defaults[row.name] === undefined && !editing.includes(row.name)
      const reset = clears.includes(row.name)
      let converted
      let valueDefault: RegistryValue | undefined
      let detail = ''
      try {
        converted = registryRow(row)
        if (row.name === '--seed') converted.valueLabels = {}
        if (row.name === '--reasoning-effort') converted.purpose = rethinking
        if (row.name === 'GUFO_THINK_MARKER_BIAS_BY_EFFORT') {
          converted.label = 'Rethinking penalty'
          converted.purpose = rethinking
        }
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
        valueDefault = converted.defaultValue
        const field = fieldFromRow(converted)
        if (field.control === 'unsupported') detail = field.error.message
      } catch (cause) {
        detail = String(cause)
      }
      return (
        <div
          key={row.name}
          data-setting={row.name}
          role="group"
          aria-label={
            row.name === '--profile' ? 'Mode' : (row.label ?? row.name)
          }
          className={`min-w-0 space-y-2 border-t py-3 ${row.name === 'GUFO_THINK_MARKER_BIAS_BY_EFFORT' ? 'min-[560px]:col-span-2' : ''}`}
        >
          {detail || blocked ? (
            <>
              <p>{row.label ?? row.name}</p>
              <p className="text-sm text-muted-foreground">
                This setting can't be changed here.
              </p>
              {detail && (
                <details>
                  <summary>Details</summary>
                  <p>{detail}</p>
                </details>
              )}
            </>
          ) : unset ? (
            <>
              <p>{row.label ?? row.name}</p>
              <p className="text-sm text-muted-foreground">
                {unsetSettingText(row.name)}
              </p>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setEditing((current) => [...current, row.name])
                  if (
                    typeof valueDefault === 'string' ||
                    typeof valueDefault === 'number' ||
                    typeof valueDefault === 'boolean'
                  )
                    form.setValue(
                      row.name,
                      markerListUnset(row.name, valueDefault)
                        ? ''
                        : valueDefault,
                    )
                }}
              >
                Set {row.label ?? row.name}
              </Button>
            </>
          ) : (
            <RegistryFields
              rows={[
                {
                  ...converted!,
                  label: row.name === '--profile' ? 'Mode' : converted!.label,
                },
              ]}
              form={form}
              disabled={reset}
              compact={section === 'Inference'}
              defaultHint={
                row.name === '--seed'
                  ? 'Random'
                  : (row.values.default?.text ??
                    row.built_in ??
                    row.default_text)
              }
              numericInput={
                sampling.includes(row.name) || row.name === '--seed'
              }
            />
          )}
          {((!blocked && !detail) || state?.saved != null) && (
            <Button
              type="button"
              variant="ghost"
              className="h-8 px-0 py-1 text-xs leading-normal"
              disabled={reset}
              onClick={() => setClears((current) => [...current, row.name])}
            >
              Reset{' '}
              {row.name === '--profile' ? 'Mode' : (row.label ?? row.name)} to
              default
            </Button>
          )}
          {reset && (
            <p className="text-sm text-muted-foreground">
              Default will be used after saving.
            </p>
          )}
          {state?.saved != null && !reset && (
            <>
              <p className="text-sm text-muted-foreground">
                Set by you:{' '}
                {row.name === '--profile'
                  ? (view.profiles.find(
                      (profile) =>
                        profile.name ===
                        (profileAliases[state.saved!] ?? state.saved),
                    )?.label ?? state.saved)
                  : state.saved}
              </p>
              <details className="text-sm text-muted-foreground">
                <summary>Default details</summary>
                <p>Default: {row.default_text}</p>
              </details>
            </>
          )}
        </div>
      )
    }
    const grid =
      'grid grid-cols-1 items-start gap-x-5 gap-y-2 min-[560px]:grid-cols-2 min-[900px]:grid-cols-3'
    if (section !== 'Inference')
      return <div className={grid}>{sectionRows.map(renderRow)}</div>
    const primary = [
      '--temperature',
      '--top-p',
      '--top-k',
      '--seed',
      '--reasoning-effort',
      'GUFO_THINK_MARKER_BIAS_BY_EFFORT',
    ]
    return (
      <>
        <div className={grid}>
          {sectionRows
            .filter((row) => primary.includes(row.name))
            .map(renderRow)}
        </div>
        <details className="border-t pt-3">
          <summary className="cursor-pointer text-sm">
            More inference settings
          </summary>
          <div className={grid}>
            {sectionRows
              .filter((row) => !primary.includes(row.name))
              .map(renderRow)}
          </div>
        </details>
      </>
    )
  }
  return {
    dirty: changed.length > 0,
    setupChanged: changed.includes('--profile'),
    patch,
    reset,
    render,
    summary: summary(),
    error,
  }
}
