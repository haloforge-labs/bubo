import { ErrorNote } from '@/components/error-note'
import { useEffect, useMemo, useState } from 'react'
import { Disclosure } from '@/components/disclosure'
import { EmptyState } from '@/components/empty-state'
import { RegistryRow, useRegistryForm } from '@/components/registry-rows'
import { SettingGroup } from '@/components/setting-row'
import { StatusLine } from '@/components/status-line'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { actions, requireActionSuccess, type View } from '@/lib/bindings'
import { canonicalSwitches, groupLabel, groupRows } from '@/lib/registry-view'
import { serverRowSection, sideFileSwitches } from '@/lib/server-settings'
import { guiRegistryPresentation } from '@/lib/presentation'
import {
  useEndReveal,
  useRevealedSetting,
  useRevealSetting,
} from '@/lib/reveal'
import { CurrentGPU } from './current-gpu'
import { NPUPrefill } from './npu-prefill'

// The Advanced settings (and the user settings that are not on the Server page),
// read and edited through the same registry engine and the same setting row as
// the Server page. This file decides only the grouping, the filter, the Debug
// fold and the Save bar.
export function RegistrySection({
  view,
  audience,
  group,
  onDirtyChange,
  includeUserRemainder,
}: {
  view: View
  audience: string
  group?: string
  onDirtyChange?: (dirty: boolean) => void
  includeUserRemainder?: boolean
}) {
  const [filter, setFilter] = useState('')
  const [debugFilter, setDebugFilter] = useState('')
  // A link that names a setting clears a filter that would hide it, then shows it.
  const revealed = useRevealedSetting()
  const endReveal = useEndReveal()
  const [followed, setFollowed] = useState<string>()
  if (followed !== revealed) {
    setFollowed(revealed)
    if (revealed) {
      setFilter('')
      setDebugFilter('')
    }
  }
  useRevealSetting(revealed, `${filter}|${debugFilter}`)
  const save = useActionTask()
  const rows = useMemo(
    () =>
      canonicalSwitches(view.switches)
        .map(guiRegistryPresentation)
        .filter(
          (row) =>
            row.gui_control &&
            !(sideFileSwitches as readonly string[]).includes(row.name) &&
            (row.audience === audience ||
              (includeUserRemainder &&
                row.audience === 'user' &&
                row.name !== '--api-key' &&
                !serverRowSection(row)) ||
              (audience === 'advanced' && row.audience === 'debug')) &&
            (!group || row.group === group),
        ),
    [view.switches, audience, group, includeUserRemainder],
  )
  const ctrl = useRegistryForm(view, rows)
  const count = ctrl.changed.length
  useEffect(() => {
    onDirtyChange?.(count > 0)
  }, [count, onDirtyChange])
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange])

  function renderGroups(sectionAudience: string, query: string) {
    const groups = groupRows(
      ctrl.entries.map((entry) =>
        includeUserRemainder && entry.row.audience === 'user'
          ? { ...entry.row, audience: 'advanced' }
          : entry.row,
      ),
      sectionAudience,
      query,
    )
    return (
      <>
        {Object.entries(groups).map(([name, groupedRows]) => (
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
            <SettingGroup>
              {groupedRows.map((row) => {
                const entry = ctrl.entries.find(
                  (item) => item.row.name === row.name,
                )!
                return (
                  <RegistryRow
                    key={row.name}
                    entry={entry}
                    ctrl={ctrl}
                    view={view}
                    disabled={save.pending}
                  />
                )
              })}
            </SettingGroup>
          </section>
        ))}
        {!Object.keys(groups).length && (
          <EmptyState>No matching settings.</EmptyState>
        )}
      </>
    )
  }

  return (
    <form
      className="space-y-5"
      noValidate
      onSubmit={(event) => {
        event.preventDefault()
        const patch = ctrl.patch()
        if (!patch) return
        void save.run(async () => {
          if (!ctrl.dirty) return
          requireActionSuccess(
            await actions.PatchPreferences({
              set: {},
              clear: patch.clear,
              ...(patch.switch_values.length
                ? { switch_values: patch.switch_values }
                : {}),
            }),
          )
          // Clean the whole transaction only after the supervisor accepts it.
          ctrl.reset()
        })
      }}
    >
      <Input
        aria-label={
          audience === 'advanced'
            ? 'Find a setting'
            : `Filter ${audience} settings`
        }
        placeholder="Find a setting"
        value={filter}
        onChange={(event) => {
          setFilter(event.target.value)
          endReveal()
        }}
      />
      {renderGroups(audience, filter)}
      {audience === 'advanced' && (
        <Disclosure
          summary="Debug"
          className="border-t pt-5"
          summaryClassName="min-h-10 text-xl"
        >
          <div className="space-y-5 pt-4">
            <p className="text-sm text-muted-foreground">
              These settings are for troubleshooting.
            </p>
            <Input
              aria-label="Filter debug settings"
              placeholder="Filter debug settings…"
              value={debugFilter}
              onChange={(event) => {
                setDebugFilter(event.target.value)
                endReveal()
              }}
            />
            {renderGroups('debug', debugFilter)}
          </div>
        </Disclosure>
      )}
      {count > 0 && (
        <div
          className="sticky bottom-0 z-10 flex flex-wrap items-center gap-3 border-t bg-background py-3"
          aria-label="Save settings"
        >
          <span aria-live="polite" className="text-sm">
            {count} unsaved {count === 1 ? 'change' : 'changes'}
          </span>
          <Button type="submit" variant="outline" disabled={save.pending}>
            Save changes
          </Button>
          <Button
            type="button"
            variant="ghost"
            disabled={save.pending}
            onClick={ctrl.discard}
          >
            Discard changes
          </Button>
        </div>
      )}
      {ctrl.error && <ErrorNote>{ctrl.error}</ErrorNote>}
      <ActionError error={save.error} />
      {save.completed && count === 0 && (
        <StatusLine>Settings saved.</StatusLine>
      )}
    </form>
  )
}
