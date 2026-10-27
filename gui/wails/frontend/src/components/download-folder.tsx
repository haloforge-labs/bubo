import type { ReactNode } from 'react'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import { guiError } from '@/lib/gui-errors'
import { bytes } from '@/lib/format'

// The download folder decides only where NEW downloads go. Models downloaded
// earlier stay where they are and keep working (Models lists every package the
// CLI finds in the folders it is given).
export const downloadFolderHelp =
  'New downloads go here. Models you already downloaded stay where they are and keep working.'

// The CLI names a drive with its root; the screen writes "C:".
export function driveName(drive: string) {
  return drive.replace(/[\\/]+$/, '')
}

type DownloadFolder = View['download_folder']

// "D: has 892.1 GiB free"; nothing at all until the figure is known.
export function FolderSpace({ folder }: { folder: DownloadFolder }) {
  if (folder.free_bytes == null) return null
  return (
    <span>
      {driveName(folder.drive) || 'This drive'} has{' '}
      <span className="tabular-nums">{bytes(folder.free_bytes)}</span> free
    </span>
  )
}

// Why the folder cannot change now: the sentence of the Go reason code.
export function FolderLocked({ folder }: { folder: DownloadFolder }) {
  const { change } = folder
  if (change.allowed || !change.reason_code) return null
  // A state, not a failure: one quiet line with the sentence of the code.
  return (
    <p className="text-sm text-muted-foreground">
      {guiError({ code: change.reason_code, remedy: change.reason }).message}
    </p>
  )
}

// Opens the folder picker. The folder is saved by Go; a cancelled picker
// changes nothing. onChanged runs after a saved change (the consent dialog
// re-plans there).
export function ChangeDownloadFolder({
  view,
  children = 'Change…',
  onChanged,
}: {
  view: View
  children?: ReactNode
  // Runs after a saved change, with the folder that was chosen.
  onChanged?: (folder: string) => Promise<unknown>
}) {
  const task = useActionTask()
  return (
    <>
      <Button
        variant="outline"
        size="sm"
        disabled={!view.download_folder.change.allowed || task.pending}
        onClick={() =>
          void task.run(async () => {
            const result = await actions.BrowseDownloadFolder()
            if (!result.cancelled) await onChanged?.(result.folder)
            return result
          })
        }
      >
        {children}
      </Button>
      <ActionError error={task.error} />
    </>
  )
}

export function UseDefaultDownloadFolder({ view }: { view: View }) {
  const task = useActionTask()
  if (view.download_folder.is_default) return null
  return (
    <>
      <Button
        variant="outline"
        size="sm"
        disabled={!view.download_folder.change.allowed || task.pending}
        onClick={() =>
          void task.run(() =>
            actions.PatchPreferences({ set: {}, clear: ['data_folder'] }),
          )
        }
      >
        Use default
      </Button>
      <ActionError error={task.error} />
    </>
  )
}
