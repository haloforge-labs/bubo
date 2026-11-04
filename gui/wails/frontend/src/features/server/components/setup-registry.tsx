import { useMemo } from 'react'
import { Disclosure } from '@/components/disclosure'
import { RegistryRow, useRegistryForm } from '@/components/registry-rows'
import { SettingGroup } from '@/components/setting-row'
import type { View } from '@/lib/bindings'
import { guiRegistryPresentation, valueWord } from '@/lib/presentation'
import { canonicalSwitches, registryDefault } from '@/lib/registry-view'
import { serverRowSection } from '@/lib/server-settings'

type Row = View['switches'][number]

const order: Record<string, string[]> = {
  Network: ['--port', '--host', '--served-model-name'],
  Inference: [
    '--temperature',
    '--top-p',
    '--top-k',
    '--seed',
    '--reasoning-effort',
    '--prefill-mode',
    'GUFO_THINK_MARKER_BIAS_BY_EFFORT',
  ],
}
const primary = order.Inference!

// The settings of the Server page, read and edited through the one registry
// engine that Settings uses too. The page decides only which rows belong to
// which section and where they sit.
export function useSetupRegistry(view: View) {
  const rows = useMemo(
    () =>
      canonicalSwitches(view.switches)
        .filter(
          (row) =>
            serverRowSection(row) &&
            (serverRowSection(row) !== 'Setup' || row.name === '--profile'),
        )
        .map(guiRegistryPresentation),
    [view.switches],
  )
  const ctrl = useRegistryForm(view, rows)
  const currentValues = ctrl.form.watch()
  function summary() {
    const value = (name: string) => {
      const row = rows.find((row) => row.name === name)
      if (!row) return undefined
      const current = ctrl.clears.includes(name)
        ? undefined
        : currentValues[name]
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
    if (reasoning != null)
      parts.push(
        `Reasoning effort ${valueWord('--reasoning-effort', reasoning)}`,
      )
    return parts.join(' · ')
  }
  const sectionEntries = (section: string) =>
    ctrl.entries
      .filter((entry) => serverRowSection(entry.row) === section)
      .sort((left, right) => {
        const names = order[section] ?? []
        const rank = (name: string) =>
          names.includes(name) ? names.indexOf(name) : names.length
        return rank(left.row.name) - rank(right.row.name)
      })
  const group = (entries: typeof ctrl.entries) => (
    <SettingGroup>
      {entries.map((entry) => (
        <RegistryRow
          key={entry.row.name}
          entry={entry}
          ctrl={ctrl}
          view={view}
        />
      ))}
    </SettingGroup>
  )
  return {
    dirty: ctrl.dirty,
    setupChanged: ctrl.changed.includes('--profile'),
    patch: ctrl.patch,
    reset: ctrl.reset,
    error: ctrl.error,
    summary: summary(),
    has: (section: string) =>
      rows.some((row: Row) => serverRowSection(row) === section),
    // One setting by name, for a group the page lays out itself (the KV cache
    // under the helper files).
    renderRow: (name: string) => {
      const entry = ctrl.entries.find((item) => item.row.name === name)
      return entry ? (
        <RegistryRow entry={entry} ctrl={ctrl} view={view} />
      ) : null
    },
    render: (section: string) => {
      const entries = sectionEntries(section)
      if (section !== 'Inference') return group(entries)
      return (
        <>
          {group(entries.filter((entry) => primary.includes(entry.row.name)))}
          <Disclosure
            summary="More inference settings"
            className="border-t pt-3"
            summaryClassName="text-sm"
          >
            <div className="pt-3">
              {group(
                entries.filter((entry) => !primary.includes(entry.row.name)),
              )}
            </div>
          </Disclosure>
        </>
      )
    },
  }
}
