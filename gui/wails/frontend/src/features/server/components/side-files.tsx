import { useState, type ReactNode } from 'react'
import { SettingGroup, SettingRow } from '@/components/setting-row'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { actions, type View } from '@/lib/bindings'
import { savedImmediately } from '@/lib/presentation'
import { sideFileSwitches } from '@/lib/server-settings'

// The two optional helper files of a plain model file, and under them the KV
// cache row, each one line of one grid so the controls share a column. A package
// model brings its own helper files, so those rows show only when the model in
// the picker is a plain file, saved or not. They are saved as the CLI's own
// --mtp-model / --mmproj settings, which the supervisor drops when the model
// changes, so a model picked and not yet saved is saved first: the CLI checks
// the chosen file against the saved model.
const rows = [
  {
    name: sideFileSwitches[0],
    kind: 'draft',
    label: 'MTP draft file',
    short: savedImmediately,
    about:
      'Optional. The MTP (multi-token prediction) draft file lets the model write faster. Choose the MTP file that matches this model, or leave it empty to run without it. Used after the server restarts.',
  },
  {
    name: sideFileSwitches[1],
    kind: 'image',
    label: 'mmproj file',
    short: savedImmediately,
    about:
      'Optional. The mmproj file (the multimodal projector) lets the model read pictures. Choose the one that matches this model, or leave it empty to run without image input. Used after the server restarts.',
  },
] as const

export function SideFiles({
  view,
  plainFile,
  unsaved,
  saveModel,
  cacheRow,
}: {
  view: View
  // The model in the picker is a plain model file.
  plainFile: boolean
  // The picker holds a model that is not saved yet.
  unsaved: boolean
  saveModel: () => Promise<unknown>
  cacheRow?: ReactNode
}) {
  const task = useActionTask()
  // What the CLI said about the file just chosen, per row. A file the CLI says
  // does not fit is not saved.
  const [verdicts, setVerdicts] = useState<
    Record<string, { fits: boolean; text: string }>
  >({})
  const shown = !plainFile
    ? []
    : rows.filter((row) =>
        view.switches.some((sw) => sw.name === row.name && sw.gui_control),
      )
  if (!shown.length && !cacheRow) return null
  const browse = (row: (typeof rows)[number]) =>
    task.mutate(async () => {
      if (unsaved) await saveModel()
      const result = await actions.BrowseSideFile(row.kind)
      if (result.cancelled || !result.file) return
      if (result.checked && !result.fits) {
        setVerdicts((current) => ({
          ...current,
          [row.name]: {
            fits: false,
            text: result.reason || 'This file does not fit this model.',
          },
        }))
        return
      }
      setVerdicts((current) => ({
        ...current,
        [row.name]: {
          fits: true,
          text: result.checked
            ? ['Fits this model.', result.note].filter(Boolean).join(' ')
            : 'Bubo could not check this file against the model.',
        },
      }))
      await actions.PatchPreferences({
        set: {},
        switch_values: [{ name: row.name, value: result.file }],
        clear: [],
      })
    })
  const clear = (name: string) =>
    task.mutate(() => actions.PatchPreferences({ set: {}, clear: [name] }))
  return (
    <div role="group" aria-label="Files and cache" className="space-y-2">
      {/* The two helper files share one grid row, as Context and Sessions do. */}
      <SettingGroup className="empty:hidden">
        {shown.map((row) => {
          const path = view.saved.switch_overrides?.[row.name]
          const verdict = verdicts[row.name]
          return (
            <SettingRow
              key={row.name}
              name={row.name}
              label={row.label}
              wide
              hint={unsaved ? undefined : row.short}
              about={<p>{row.about}</p>}
              control={
                <>
                  <span
                    title={path}
                    className={`flex h-9 min-w-0 flex-1 items-center rounded-md border px-2 text-sm ${path ? '' : 'text-muted-foreground'}`}
                  >
                    <span className="truncate">
                      {path ? path.split(/[\\/]/).pop() : 'None'}
                    </span>
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    className="px-3"
                    disabled={task.isPending}
                    aria-label={`Browse for ${row.label}`}
                    onClick={() => browse(row)}
                  >
                    Browse…
                  </Button>
                  {path && (
                    <Button
                      type="button"
                      variant="ghost"
                      className="px-2"
                      disabled={task.isPending}
                      aria-label={`Clear ${row.label}`}
                      onClick={() => clear(row.name)}
                    >
                      Clear
                    </Button>
                  )}
                </>
              }
              error={
                verdict && (
                  <p
                    role={verdict.fits ? 'status' : 'alert'}
                    className={`text-sm ${verdict.fits ? 'text-muted-foreground' : 'text-destructive'}`}
                  >
                    {verdict.text}
                  </p>
                )
              }
            />
          )
        })}
        {unsaved && shown.length > 0 && (
          <p className="col-span-full text-xs text-muted-foreground">
            Choosing a file saves the model you picked.
          </p>
        )}
      </SettingGroup>
      {cacheRow}
      <div className="empty:hidden">
        <ActionError error={task.error} />
      </div>
    </div>
  )
}
