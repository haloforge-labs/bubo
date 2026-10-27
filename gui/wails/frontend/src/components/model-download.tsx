import { useEffect, useState, type ReactNode } from 'react'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import {
  bytes,
  preparedRows,
  preparedStart,
  samePath,
  shortByShown,
  type ModelPlan,
} from '@/components/model-facts'
import { damagedFile } from '@/lib/gui-errors'
import { folderPath } from '@/lib/format'
import {
  cancelModelPlan,
  confirmModelPlan,
  openModelPlan,
} from '@/components/model-download-flow'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import {
  CurrentFileDetails,
  TransferProgress,
} from '@/components/transfer-progress'
import { GUIError } from '@/components/gui-error'
import { ErrorNote } from '@/components/error-note'
import { PathText } from '@/components/path-text'
import {
  ChangeDownloadFolder,
  driveName,
  FolderLocked,
} from '@/components/download-folder'

export function plannedTransferBytes(plan: ModelPlan) {
  return plan.files.reduce(
    (total, file) =>
      plan.kind === 'runtime' ||
      file.state === 'download' ||
      file.state === 'resume'
        ? total + file.size
        : total,
    0,
  )
}

const engineFileLabels = new Map([
  ['rocm_sdk_core', 'AMD engine core'],
  ['rocm_sdk_libraries', 'AMD engine libraries'],
  ['rocm_sdk_device_gfx1151', 'AMD GPU kernels (gfx1151)'],
])

export function PlanSummary({
  plan,
  licence,
  showSpace = true,
  checkOnly = false,
  changeFolder,
  downloadFolder,
  canChangeFolder = true,
}: {
  plan: ModelPlan
  licence?: ModelPlan['licence']
  showSpace?: boolean
  checkOnly?: boolean
  // The model consent shows where the files go in plain view, with the action
  // that changes it (and the line that says why it is locked, right under it).
  // Other plans (the engine files) keep the closed disclosure.
  changeFolder?: ReactNode
  // The download folder as Settings names it, not the package path.
  downloadFolder?: string
  canChangeFolder?: boolean
}) {
  const terms = plan.licence ?? licence
  const licenceTexts = plan.licences.filter((item) => item.text.trim())
  const destinations = Object.values(plan.paths ?? {})
  const runtimeDestinations =
    plan.kind === 'runtime'
      ? Object.entries(plan.paths)
          .filter(([key]) => key === 'cache' || key === 'runtime')
          .filter(
            ([, path], index, entries) =>
              !entries
                .slice(0, index)
                .some(([, previous]) => samePath(previous, path)),
          )
      : []
  // Display only human-readable CLI paths. A volume GUID is an identifier, not a destination label.
  const readable = (path: string) => !path.startsWith('\\\\?\\Volume{')
  const volumes = plan.volumes ?? []
  const namedDrive =
    plan.drive && readable(plan.drive)
      ? plan.drive
      : volumes.length === 1
        ? destinations.find((path) => /^[A-Za-z]:[\\/]/.test(path))?.slice(0, 3)
        : undefined
  // Drives are written "C:" everywhere.
  const drive = namedDrive ? driveName(namedDrive) : undefined
  // The CLI's own safety margin of the tightest volume (a fixed 2 GiB today).
  const spare = volumes.find(
    (volume) => volume.required_bytes === plan.required_bytes,
  )?.safety_margin_bytes
  const unidentified = volumes.filter((volume) => !readable(volume.name))
  const replacements = plan.files.filter((file) => file.replaces)
  const transferBytes = plannedTransferBytes(plan)
  const statesReported = plan.files.every((file) => !!file.state)
  const spareText = spare
    ? bytes(spare, {
        unit: 'GiB',
        digits: Number.isInteger(spare / 1024 ** 3) ? 0 : 1,
      })
    : ''
  // "Needs 110.6 GiB (108.6 GiB + 2 GiB spare)" only when the printed parts add
  // up to the printed total; otherwise the download figure keeps its own line.
  const tenths = (value: number) => Math.round((value / 1024 ** 3) * 10)
  const breakdown =
    plan.kind !== 'runtime' &&
    statesReported &&
    !!spare &&
    tenths(transferBytes) + tenths(spare) === tenths(plan.required_bytes)
  const files = !!plan.files?.length && (
    <details>
      <summary>Files ({plan.files.length})</summary>
      <ul aria-label="Download files" className="space-y-2 pt-2">
        {plan.files.map((file) => (
          <li key={file.file_id} className="flex justify-between gap-3">
            <span className="min-w-0 break-all">
              {plan.kind === 'runtime'
                ? (engineFileLabels.get(file.file_id) ?? file.file_id)
                : file.path?.split(/[\\/]/).at(-1) || file.file_id}
            </span>
            <span className="shrink-0 tabular-nums">{bytes(file.size)}</span>
          </li>
        ))}
      </ul>
    </details>
  )
  if (checkOnly)
    return (
      <div className="space-y-3 text-sm">
        <p>All files match. Nothing to download.</p>
        {files}
      </div>
    )
  return (
    <div className="min-w-0 space-y-3 text-sm [overflow-wrap:anywhere]">
      {showSpace && (
        <div className="space-y-1">
          {!breakdown && (
            <p>
              {plan.kind === 'runtime' || statesReported ? (
                <>
                  To download:{' '}
                  <span className="tabular-nums">{bytes(transferBytes)}</span>
                </>
              ) : (
                <>
                  Planned files:{' '}
                  <span className="tabular-nums">
                    {bytes(
                      plan.files.reduce((total, file) => total + file.size, 0),
                    )}
                  </span>
                </>
              )}
            </p>
          )}
          {changeFolder && downloadFolder && (
            <p>
              Downloads to <PathText path={downloadFolder} />
            </p>
          )}
          <p>
            Needs{' '}
            <span className="tabular-nums">{bytes(plan.required_bytes)}</span>
            {plan.kind === 'runtime' ? (
              ' (includes unpacking)'
            ) : breakdown ? (
              <>
                {' '}
                (<span className="tabular-nums">
                  {bytes(transferBytes)}
                </span> + {spareText} spare)
              </>
            ) : spare ? (
              ` (includes ${spareText} spare)`
            ) : null}{' '}
            · {drive || 'Destination'} has{' '}
            <span className="tabular-nums">{bytes(plan.free_bytes)}</span> free
          </p>
          {changeFolder}
          {!plan.enough_space && (
            <ErrorNote>
              <span>
                Short by{' '}
                <span className="tabular-nums">
                  {shortByShown(plan.required_bytes, plan.free_bytes)}
                </span>
                . Free up space on {drive || 'the destination drive'}
                {changeFolder && canChangeFolder
                  ? ', or choose another folder.'
                  : ' and try again.'}
              </span>
            </ErrorNote>
          )}
        </div>
      )}
      {replacements.length > 0 && (
        <ul aria-label="Files to replace" className="space-y-1">
          {replacements.map((file) => (
            <li key={file.file_id} className="break-all">
              {file.path?.split(/[\\/]/).at(-1) || file.file_id}: replaces a
              file that did not match
            </li>
          ))}
        </ul>
      )}
      {files}
      {!changeFolder && (
        <details>
          <summary>
            {plan.kind === 'runtime' ? 'Engine folder' : 'Download folder'}
          </summary>
          {runtimeDestinations.length ? (
            runtimeDestinations.map(([key, path]) => (
              <p key={key}>
                {key === 'cache' ? 'Downloads to ' : 'Installs to '}
                <PathText path={path} />
              </p>
            ))
          ) : destinations.length ? (
            destinations.map((path) => (
              <p key={path}>
                {readable(path) ? (
                  <PathText path={path} />
                ) : (
                  <span>Destination folder not reported</span>
                )}
              </p>
            ))
          ) : (
            <p data-code="not_reported">Not reported by Bubo.</p>
          )}
        </details>
      )}
      {volumes.length > 1 ? (
        volumes
          .filter((volume) => readable(volume.name))
          .map((volume) => (
            <p
              key={volume.name}
              data-code={
                volume.free_after_download_bytes == null
                  ? 'not_reported'
                  : undefined
              }
            >
              Free space after download on {volume.name}:{' '}
              {volume.free_after_download_bytes == null ? (
                'Not reported by Bubo'
              ) : (
                <span className="tabular-nums">
                  {bytes(volume.free_after_download_bytes)}
                </span>
              )}
            </p>
          ))
      ) : !volumes.length ? (
        <p data-code="not_reported">
          Free space after download is not reported by Bubo.
        </p>
      ) : null}
      {volumes.length > 1 && unidentified.length > 0 && (
        <p data-code="not_reported">
          Per-drive remaining space needs a destination label. See the download
          folders above.
        </p>
      )}
      {licenceTexts.length ? (
        <>
          <p>
            By downloading you accept the licence terms for these{' '}
            {plan.kind === 'runtime' ? 'engine' : 'model'} files.
            {terms?.name && terms.url && (
              <>
                {' '}
                You can also read the{' '}
                <a
                  href={terms.url}
                  className="underline underline-offset-4"
                  onClick={(event) => {
                    event.preventDefault()
                    void actions.OpenExternal(terms.url)
                  }}
                >
                  {terms.name}
                </a>{' '}
                online.
              </>
            )}
          </p>
          <details>
            <summary>Read licence texts ({licenceTexts.length})</summary>
            <div className="space-y-4 pt-2">
              {licenceTexts.map((item) => (
                <section key={item.path} className="space-y-1">
                  <h3 className="font-medium break-all">{item.path}</h3>
                  <pre className="whitespace-pre-wrap break-words font-sans text-xs">
                    {item.text}
                  </pre>
                </section>
              ))}
            </div>
          </details>
        </>
      ) : terms?.name && terms.url ? (
        <p>
          By downloading you accept the{' '}
          <a
            href={terms.url}
            className="underline underline-offset-4"
            onClick={(event) => {
              event.preventDefault()
              void actions.OpenExternal(terms.url)
            }}
          >
            {terms.name}
          </a>
          .
        </p>
      ) : (
        <p data-code="not_reported">
          Bubo did not report licence terms for these files.
        </p>
      )}
    </div>
  )
}

export function ModelDownloadFlow({
  view,
  children,
}: {
  view: View
  children: (
    open: (id: string, kind?: 'download' | 'repair', output?: string) => void,
    pending: boolean,
  ) => ReactNode
}) {
  const [plan, setPlan] = useState<ModelPlan | undefined>(() =>
    view.models?.download?.state === 'awaiting_confirm'
      ? view.models.download.plan
      : undefined,
  )
  const command = useActionTask()
  function open(
    id: string,
    kind: 'download' | 'repair' = 'download',
    output?: string,
  ) {
    command.mutate(async () => {
      setPlan(await openModelPlan(actions, id, kind, output))
    })
  }
  const entry = view.models?.catalogue?.find((entry) => entry.id === plan?.id)
  const terms = plan?.licence ?? entry?.licence
  const checkOnly =
    !!plan &&
    plan.files.every((file) => !!file.state) &&
    plannedTransferBytes(plan) === 0 &&
    !plan.files.some((file) => file.replaces)
  // A new download goes to the download folder: changing it re-plans, so the
  // destination, the space line and any shortfall follow at once.
  const [chosen, setChosen] = useState<string>()
  const replan = async (folder: string) => {
    if (!plan) return
    setPlan(await openModelPlan(actions, plan.id, 'download'))
    setChosen(folder)
  }
  function close() {
    command.mutate(async () => {
      await cancelModelPlan(actions)
      setPlan(undefined)
      setChosen(undefined)
    })
  }
  return (
    <>
      {children(open, command.isPending)}
      {!plan && <ActionError error={command.error} />}
      <Dialog
        open={!!plan}
        onOpenChange={(open) => {
          if (!open && !command.isPending) close()
        }}
      >
        <DialogContent
          showCloseButton={false}
          className="flex max-h-[90dvh] min-w-0 flex-col overflow-hidden focus:outline-hidden"
          // Open with the dialog itself focused, not its first disclosure: no
          // focus ring appears on "Files" when the dialog opens.
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            ;(event.currentTarget as HTMLElement | null)?.focus()
          }}
        >
          <DialogHeader className="shrink-0">
            <DialogTitle>
              {checkOnly
                ? 'Check'
                : plan?.kind === 'repair'
                  ? 'Repair'
                  : 'Download'}{' '}
              {entry?.display_name ?? 'model'}
            </DialogTitle>
            <DialogDescription className="sr-only">
              {checkOnly
                ? 'Review the planned files.'
                : 'Review the planned files and space needed.'}
            </DialogDescription>
          </DialogHeader>
          <div
            data-download-body
            className="min-h-0 min-w-0 space-y-4 overflow-y-auto [overflow-wrap:anywhere]"
          >
            {plan && (
              <PlanSummary
                plan={plan}
                checkOnly={checkOnly}
                licence={entry?.licence}
                downloadFolder={chosen ?? view.download_folder.path}
                canChangeFolder={view.download_folder.change.allowed}
                changeFolder={
                  plan.kind === 'download' ? (
                    <div className="space-y-1">
                      <ChangeDownloadFolder view={view} onChanged={replan}>
                        Change folder…
                      </ChangeDownloadFolder>
                      <FolderLocked folder={view.download_folder} />
                    </div>
                  ) : undefined
                }
              />
            )}
            <ActionError error={command.error} />
          </div>
          <DialogFooter className="shrink-0 bg-background pt-3">
            {!checkOnly && (
              <Button
                variant="outline"
                disabled={command.isPending}
                onClick={close}
              >
                Not now
              </Button>
            )}
            <Button
              disabled={
                command.isPending ||
                (!checkOnly &&
                  (!plan?.enough_space ||
                    !(
                      plan?.licences.some((item) => item.text.trim()) ||
                      (terms?.url && terms.name)
                    ) ||
                    !Object.values(plan?.paths ?? {}).length))
              }
              onClick={() => {
                if (plan)
                  command.mutate(async () => {
                    await confirmModelPlan(actions, plan.id)
                    setPlan(undefined)
                    setChosen(undefined)
                  })
              }}
            >
              {checkOnly ? 'OK' : 'Download'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

// The package folder of the job's model: the only folder the app opens for it
// (the damaged file can sit in a subfolder). With two folders holding the same
// package, the one that contains the damaged file; none when it is unclear.
function packageFolder(view: View, id: string, file?: string) {
  const folders = preparedRows(view)
    .filter((row) => row.id === id)
    .flatMap((row) => {
      const start = preparedStart(row)
      return start?.kind === 'manifest' ? [folderPath(start.path)] : []
    })
  const inside = (folder: string) =>
    !!file &&
    samePath(file.slice(0, folder.length), folder) &&
    file.length > folder.length
  return folders.find(inside) ?? (folders.length === 1 ? folders[0] : undefined)
}

export function ModelDownloadProgress({
  view,
  retry,
}: {
  view: View
  retry: (id: string, kind?: 'download' | 'repair', output?: string) => void
}) {
  const job = view.models?.download
  const command = useActionTask()
  if (job?.state === 'done')
    return (
      <ModelReadyNotice
        key={`${job.id}-${job.updated_at}`}
        view={view}
        id={job.id}
      />
    )
  if (!job || job.state === 'awaiting_confirm') return null
  const active =
    job.state === 'running' ||
    job.state === 'planning' ||
    job.state === 'cancelling'
  const entry = view.models?.catalogue?.find((entry) => entry.id === job.id)
  const name = entry?.display_name ?? 'Model download'
  // Only a file the CLI names for deletion; other integrity failures keep Try again.
  const damagedPath =
    job.state === 'failed' && job.error?.code === 'integrity_failed'
      ? damagedFile(job.error?.detail)
      : undefined
  const damaged = damagedPath && {
    file: damagedPath,
    folder: packageFolder(view, job.id, damagedPath.path),
  }
  return (
    <section
      aria-label="Model download progress"
      className="space-y-2 rounded-lg border p-4"
    >
      <h2 className="font-medium">
        {name} ·{' '}
        {job.state === 'failed'
          ? 'Download failed'
          : job.state === 'cancelled'
            ? 'Download cancelled'
            : job.phase?.includes('verif')
              ? 'Checking files'
              : active
                ? 'Downloading files'
                : job.state}
      </h2>

      {/* One bar and one line for the whole job; Go adds up the files. */}
      <TransferProgress
        identity={
          job.overall
            ? `model-${job.id}-${job.kind}`
            : `model-${job.id}-${job.kind}-${job.phase}-${job.file_id}`
        }
        done={job.overall?.bytes_done ?? job.bytes_done}
        total={job.overall?.bytes_total ?? job.bytes_total ?? undefined}
        scope={job.overall ? undefined : 'Current file'}
        whole={!!job.overall}
      />
      {job.overall && (
        <Progress
          aria-label="Download progress"
          value={
            (job.overall?.bytes_total ?? job.bytes_total)
              ? ((job.overall?.bytes_done ?? job.bytes_done) /
                  (job.overall?.bytes_total ?? job.bytes_total!)) *
                100
              : undefined
          }
        />
      )}
      {job.message &&
        (job.error ? (
          <details className="text-xs text-muted-foreground">
            <summary>Download details</summary>
            <p>{job.message}</p>
          </details>
        ) : (
          <p>{job.message}</p>
        ))}
      {job.stalled && (
        <p className="text-sm text-muted-foreground">
          <span>
            Download stalled
            {job.stall_seconds ? ` for ${job.stall_seconds} seconds` : ''}.
          </span>
        </p>
      )}
      <GUIError error={job.error} recovery={false} />
      {/* A file that fails its pin stays on disk, so running the job again fails
          the same way: no Try again. The words are in gui-errors.ts. */}
      {damaged && damaged.file && (
        <p>
          <PathText path={damaged.file.path} />
        </p>
      )}
      <div className="flex gap-2">
        {active ? (
          <Button
            variant="outline"
            disabled={command.isPending || job.state === 'cancelling'}
            onClick={() => command.mutate(actions.CancelDownload)}
          >
            Cancel download
          </Button>
        ) : (
          <>
            {damaged?.folder && (
              <Button
                variant="outline"
                disabled={command.isPending}
                onClick={() =>
                  command.mutate(() => actions.OpenModelFolder(damaged.folder!))
                }
              >
                Open folder
              </Button>
            )}
            {['failed', 'cancelled'].includes(job.state) && !damaged && (
              <Button
                onClick={() =>
                  retry(
                    job.id,
                    job.kind === 'repair' ? 'repair' : 'download',
                    job.repair_output,
                  )
                }
              >
                Try again
              </Button>
            )}
            <Button
              variant="outline"
              disabled={command.isPending}
              onClick={() => command.mutate(actions.DismissModelJob)}
            >
              Dismiss
            </Button>
          </>
        )}
      </div>
      <ActionError error={command.error} />
      <CurrentFileDetails
        position={job.file_position}
        done={job.bytes_done}
        total={job.bytes_total ?? undefined}
        plan={job.plan}
        catalogueFiles={entry?.files}
        fileId={job.file_id}
        phase={job.phase}
        folder={
          job.kind === 'download'
            ? view.download_folder.path
            : job.repair_output
        }
        active={active}
      />
    </section>
  )
}

function ModelReadyNotice({ view, id }: { view: View; id: string }) {
  const [visible, setVisible] = useState(true)
  const { mutate, error } = useActionTask()
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false)
      mutate(actions.DismissModelJob)
    }, 5000)
    return () => clearTimeout(timer)
  }, [mutate])
  const name =
    view.models?.catalogue.find((entry) => entry.id === id)?.display_name ??
    'The model'
  return (
    <>
      {visible && (
        <p role="status" className="text-sm">
          {name} is ready.
        </p>
      )}
      <ActionError error={error} />
    </>
  )
}
