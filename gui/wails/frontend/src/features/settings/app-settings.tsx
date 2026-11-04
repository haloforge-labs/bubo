import { ErrorNote } from '@/components/error-note'
import { useEffect, useId } from 'react'
import { Controller, useForm } from 'react-hook-form'
import {
  ActionButton,
  ActionError,
  useActionTask,
} from '@/components/action-feedback'
import { SettingGroup, SettingRow } from '@/components/setting-row'
import { StatusLine } from '@/components/status-line'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { actions, type View } from '@/lib/bindings'
import {
  ChangeDownloadFolder,
  downloadFolderHelp,
  FolderLocked,
  FolderSpace,
  UseDefaultDownloadFolder,
} from '@/components/download-folder'
import { PathText } from '@/components/path-text'
import { applyFont, interfaceFont, type InterfaceFont } from '@/lib/font'
import {
  additionalModelFolders,
  appSettingWords,
  savedImmediately,
} from '@/lib/presentation'

// Where new model downloads go: the chosen folder, or the default one. Changing
// it is saved at once (there is no Save for it).
function DownloadFolderSection({ view }: { view: View }) {
  const folder = view.download_folder
  return (
    <section
      id="download-folder"
      className="space-y-3"
      aria-label="Model download folder"
    >
      <h3 className="flex flex-wrap items-baseline gap-x-3 text-xl">
        Model download folder
        <span className="text-xs text-muted-foreground">
          {savedImmediately}
        </span>
      </h3>
      <SettingGroup one>
        <SettingRow
          label="Folder"
          wide
          about={<p>{downloadFolderHelp}</p>}
          control={
            <>
              {folder.path && (
                <span className="min-w-0">
                  <PathText path={folder.path} />
                  {folder.is_default && (
                    <span className="text-muted-foreground"> (default)</span>
                  )}
                </span>
              )}
              {folder.free_bytes != null && (
                <span className="text-sm text-muted-foreground">
                  <FolderSpace folder={folder} />
                </span>
              )}
              <ChangeDownloadFolder view={view} />
              <UseDefaultDownloadFolder view={view} />
            </>
          }
        />
      </SettingGroup>
      <FolderLocked folder={folder} />
    </section>
  )
}

// The typeface of the whole app. It is applied and saved at once; no restart.
function FontSection({ view }: { view: View }) {
  const save = useActionTask()
  const current = interfaceFont(view.saved.font)
  const choose = async (next: InterfaceFont) => {
    if (next === current) return
    applyFont(next)
    // The view applies the saved value; a failed save puts the old face back.
    if (
      !(await save.run(() =>
        actions.PatchPreferences({ set: { font: next }, clear: [] }),
      ))
    )
      applyFont(current)
  }
  return (
    <section className="space-y-3" aria-label="Appearance">
      <SettingGroup one>
        <SettingRow
          name="font"
          label={appSettingWords.font}
          about={<p>The typeface of the whole app. It is saved at once.</p>}
          control={
            <div className="flex gap-2">
              {(
                [
                  ['serif', 'Serif'],
                  ['sans', 'Sans serif'],
                ] as const
              ).map(([value, text]) => (
                <Button
                  key={value}
                  type="button"
                  variant="outline"
                  className={
                    current === value
                      ? 'border-primary text-primary'
                      : undefined
                  }
                  aria-pressed={current === value}
                  disabled={save.pending}
                  onClick={() => void choose(value)}
                >
                  {text}
                </Button>
              ))}
            </div>
          }
        />
      </SettingGroup>
      <ActionError error={save.error} />
    </section>
  )
}

// The app's own preferences. A value that is the default needs no Reset; the
// defaults are the ones the supervisor falls back to when a preference is cleared.
const appDefaults = {
  start_gui_at_login: false,
  close_to_tray: false,
  start_server_with_gui: false,
  check_for_updates: true,
  runtime_path: '',
}

export function AppSettings({
  view,
  onDirtyChange,
}: {
  view: View
  onDirtyChange?: (dirty: boolean) => void
}) {
  const id = useId()
  const defaults = {
    start_gui_at_login: view.saved.start_gui_at_login,
    close_to_tray: view.saved.close_to_tray,
    start_server_with_gui: view.saved.start_server_with_gui,
    check_for_updates: view.check_for_updates,
    runtime_path: view.saved.runtime_path ?? '',
  }
  const form = useForm({ defaultValues: defaults })
  const signature = JSON.stringify(defaults)
  useEffect(() => {
    form.reset(JSON.parse(signature) as typeof defaults, {
      keepDirtyValues: true,
    })
  }, [signature, form])
  const folder = useForm({ defaultValues: { path: '' } })
  const { dirtyFields } = form.formState
  const count = Object.keys(dirtyFields).length
  useEffect(() => {
    onDirtyChange?.(count > 0)
  }, [count, onDirtyChange])
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange])
  const save = useActionTask()
  const add = useActionTask()
  const notDefault = (
    Object.keys(appDefaults) as (keyof typeof appDefaults)[]
  ).some((name) => defaults[name] !== appDefaults[name])
  return (
    <div className="space-y-8">
      <DownloadFolderSection view={view} />
      <FontSection view={view} />
      <form
        className="space-y-5"
        onSubmit={form.handleSubmit(async (values) => {
          const set: Parameters<typeof actions.PatchPreferences>[0]['set'] = {}
          const clear: string[] = []
          for (const name of [
            'start_gui_at_login',
            'close_to_tray',
            'start_server_with_gui',
            'check_for_updates',
          ] as const)
            if (dirtyFields[name]) set[name] = values[name]
          if (dirtyFields.runtime_path) {
            if (values.runtime_path) set.runtime_path = values.runtime_path
            else clear.push('runtime_path')
          }
          if (await save.run(() => actions.PatchPreferences({ set, clear })))
            form.reset(values)
        })}
      >
        <SettingGroup>
          {(
            [
              ['start_gui_at_login', appSettingWords.startAtLogin],
              ['close_to_tray', appSettingWords.closeToTray],
              ['start_server_with_gui', appSettingWords.startServer],
              ['check_for_updates', appSettingWords.checkForUpdates],
            ] as const
          ).map(([name, label]) => (
            <Controller
              key={name}
              name={name}
              control={form.control}
              render={({ field }) => (
                <SettingRow
                  name={name}
                  label={label}
                  htmlFor={`${id}-${name}`}
                  control={
                    <Switch
                      id={`${id}-${name}`}
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  }
                />
              )}
            />
          ))}
          <SettingRow
            name="runtime_path"
            label={appSettingWords.engineFolder}
            htmlFor={`${id}-runtime`}
            wide
            full
            control={
              <Input
                id={`${id}-runtime`}
                className="min-w-0 flex-1"
                {...form.register('runtime_path')}
                placeholder={appSettingWords.defaultEngineFolder}
              />
            }
          />
        </SettingGroup>
        {count > 0 && (
          <div
            className="sticky bottom-0 flex flex-wrap items-center gap-3 border-t bg-background py-3"
            aria-label="Save settings"
          >
            <span aria-live="polite" className="text-sm">
              {count} unsaved {count === 1 ? 'change' : 'changes'}
            </span>
            <Button
              variant="outline"
              type="submit"
              disabled={save.pending || count === 0}
            >
              Save app settings
            </Button>
            <Button
              variant="ghost"
              type="button"
              disabled={save.pending || count === 0}
              onClick={() => form.reset()}
            >
              Discard changes
            </Button>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          {notDefault && (
            <>
              <Button
                variant="ghost"
                type="button"
                disabled={save.pending}
                onClick={async () => {
                  if (
                    await save.run(() =>
                      actions.PatchPreferences({
                        set: {},
                        clear: Object.keys(appDefaults),
                      }),
                    )
                  )
                    form.reset()
                }}
              >
                Restore app defaults
              </Button>
              <span className="text-xs text-muted-foreground">
                {savedImmediately}
              </span>
            </>
          )}
          <ActionError error={save.error} />
          {save.completed && count === 0 && (
            <StatusLine>App settings saved.</StatusLine>
          )}
        </div>
      </form>
      <section className="space-y-3" aria-label={additionalModelFolders.title}>
        <h3 className="flex flex-wrap items-baseline gap-x-3 text-xl">
          {additionalModelFolders.title}
          <span className="text-xs text-muted-foreground">
            {additionalModelFolders.hint} · {savedImmediately}
          </span>
        </h3>
        <SettingGroup one>
          {(view.saved.model_folders ?? []).map((path) => (
            <SettingRow
              key={path}
              label="Folder"
              wide
              control={
                <>
                  <code className="min-w-0 break-all text-sm">{path}</code>
                  <ActionButton action={() => actions.RemoveModelFolder(path)}>
                    Remove folder
                  </ActionButton>
                </>
              }
            />
          ))}
          <form
            onSubmit={folder.handleSubmit(async ({ path }) => {
              if (await add.run(() => actions.AddModelFolder(path)))
                folder.reset()
            })}
          >
            <SettingRow
              label="Add a folder"
              htmlFor={`${id}-folder`}
              wide
              control={
                <>
                  <Input
                    id={`${id}-folder`}
                    className="min-w-0 flex-1"
                    {...folder.register('path', {
                      required: 'Enter a folder path.',
                    })}
                  />
                  <Button
                    type="submit"
                    variant="outline"
                    disabled={add.pending}
                  >
                    Add folder
                  </Button>
                </>
              }
              error={
                (folder.formState.errors.path || add.error) && (
                  <>
                    {folder.formState.errors.path && (
                      <ErrorNote>
                        <span>{folder.formState.errors.path.message}</span>
                      </ErrorNote>
                    )}
                    <ActionError error={add.error} />
                  </>
                )
              }
            />
          </form>
        </SettingGroup>
      </section>
    </div>
  )
}
