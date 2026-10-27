import { ErrorNote } from '@/components/error-note'
import { useEffect, useId } from 'react'
import { Controller, useForm } from 'react-hook-form'
import {
  ActionButton,
  ActionError,
  useActionTask,
} from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
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

// Where new model downloads go: the chosen folder, or the default one.
function DownloadFolderSection({ view }: { view: View }) {
  const folder = view.download_folder
  return (
    <section
      id="download-folder"
      className="space-y-3"
      aria-label="Model download folder"
    >
      <h3 className="text-xl">Model download folder</h3>
      <p>
        {/* A label alone says nothing: "(default)" shows only beside the path. */}
        {folder.path && (
          <>
            <PathText path={folder.path} />
            {folder.is_default && (
              <span className="text-muted-foreground"> (default)</span>
            )}
          </>
        )}
      </p>
      {folder.free_bytes != null && (
        <p className="text-sm">
          <FolderSpace folder={folder} />
        </p>
      )}
      <p className="text-sm text-muted-foreground">{downloadFolderHelp}</p>
      <div className="flex flex-wrap items-center gap-3">
        <ChangeDownloadFolder view={view} />
        <UseDefaultDownloadFolder view={view} />
      </div>
      <FolderLocked folder={folder} />
    </section>
  )
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
  return (
    <div className="space-y-8">
      <DownloadFolderSection view={view} />
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
        {(
          [
            ['start_gui_at_login', 'Start this app at login'],
            ['close_to_tray', 'Keep this app in the tray when closing'],
            ['start_server_with_gui', 'Start the server with this app'],
            ['check_for_updates', 'Check for updates at start'],
          ] as const
        ).map(([name, label]) => (
          <Controller
            key={name}
            name={name}
            control={form.control}
            render={({ field }) => (
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor={`${id}-${name}`}>{label}</Label>
                <Switch
                  id={`${id}-${name}`}
                  checked={field.value}
                  onCheckedChange={field.onChange}
                />
              </div>
            )}
          />
        ))}
        <div className="space-y-2">
          <Label htmlFor={`${id}-runtime`}>Runtime folder</Label>
          <Input
            id={`${id}-runtime`}
            {...form.register('runtime_path')}
            placeholder="Use the default"
          />
        </div>
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
          <Button
            variant="ghost"
            type="button"
            disabled={save.pending}
            onClick={async () => {
              if (
                await save.run(() =>
                  actions.PatchPreferences({
                    set: {},
                    clear: [
                      'start_gui_at_login',
                      'close_to_tray',
                      'start_server_with_gui',
                      'check_for_updates',
                      'runtime_path',
                    ],
                  }),
                )
              )
                form.reset()
            }}
          >
            Use app defaults
          </Button>
          <ActionError error={save.error} />
          {save.completed && <p role="status">App preferences saved.</p>}
        </div>
      </form>
      <section className="space-y-4" aria-label="Model folders">
        <h3 className="text-xl">Model folders</h3>
        {(view.saved.model_folders ?? []).map((path) => (
          <div className="flex flex-wrap items-center gap-4" key={path}>
            <code className="min-w-0 break-all text-sm">{path}</code>
            <ActionButton action={() => actions.RemoveModelFolder(path)}>
              Remove folder
            </ActionButton>
          </div>
        ))}
        {!view.saved.model_folders?.length && (
          <p className="text-muted-foreground">No additional model folders.</p>
        )}
        <form
          className="space-y-2"
          onSubmit={folder.handleSubmit(async ({ path }) => {
            if (await add.run(() => actions.AddModelFolder(path)))
              folder.reset()
          })}
        >
          <Label htmlFor={`${id}-folder`}>Add a model folder</Label>
          <Input
            id={`${id}-folder`}
            {...folder.register('path', { required: 'Enter a folder path.' })}
          />
          {folder.formState.errors.path && (
            <ErrorNote>
              <span>{folder.formState.errors.path.message}</span>
            </ErrorNote>
          )}
          <Button type="submit" variant="outline" disabled={add.pending}>
            Add folder
          </Button>
          <ActionError error={add.error} />
        </form>
      </section>
    </div>
  )
}
