import { Page } from '@/components/page'
import { DesktopView } from '@/components/desktop-view'
import { ActionError, useActionTask } from '@/components/action-feedback'
import {
  bytes,
  preparedRows,
  preparedStart,
  packageStatus,
  estimateSubject,
  modelPreferencePatch,
  samePath,
} from '@/components/model-facts'
import {
  ModelDownloadFlow,
  ModelDownloadProgress,
} from '@/components/model-download'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { actions, type View } from '@/lib/bindings'
import { requireActionSuccess } from '@/lib/bindings'
import { useState } from 'react'
import { useQueries } from '@tanstack/react-query'
import { folderPath } from '@/lib/format'
import { PathText } from '@/components/path-text'
import { CopyText } from '@/components/copy-text'
import { GUIError } from '@/components/gui-error'
import { guiError } from '@/lib/gui-errors'
import { Link } from '@tanstack/react-router'
import { ImportMenu } from './import-menu'
import { ModelInfo } from './model-info'

// What one Installed row's fit cell says. "none": nothing (the row cannot be
// selected, or the server holds the memory). "checking": an estimate is in
// flight. "unknown": no verdict, shown as "—" with one note under the table.
type RowFit =
  | { kind: 'none' }
  | { kind: 'checking' }
  | { kind: 'unknown' }
  | { kind: 'sentence'; text: string }

// The CLI's answer for one fit: its sentence, or no verdict.
function answered(fit?: { class: string; sentence: string }): RowFit {
  if (!fit) return { kind: 'unknown' }
  if (fit.class === 'unavailable') return { kind: 'checking' }
  if (fit.class === 'fit_unknown' || !fit.sentence) return { kind: 'unknown' }
  return { kind: 'sentence', text: fit.sentence }
}

export default function ModelsPage() {
  return (
    <Page title="Models">
      <DesktopView>{(view) => <ModelsContent view={view} />}</DesktopView>
    </Page>
  )
}

// A model that is on this computer has no status word: the row being in the
// Installed table says it. Other states (incomplete, in use) keep theirs.
function installedWord(state: string) {
  const { word } = packageStatus(state)
  return state === 'prepared' ? '' : word
}

function ModelsContent({ view }: { view: View }) {
  if (!view.models)
    return (
      <GUIError
        error={{ code: 'internal_error', remedy: 'Models View is missing.' }}
      />
    )
  return <ModelsLoaded view={view} models={view.models} />
}

function ModelsLoaded({
  view,
  models,
}: {
  view: View
  models: NonNullable<View['models']>
}) {
  const command = useActionTask()
  const [notice, setNotice] = useState('')
  const browse = (kind: 'file' | 'folder') =>
    command.mutate(async () => {
      const result = await (kind === 'file'
        ? actions.BrowseModelFile()
        : actions.BrowseModelFolder())
      requireActionSuccess(result)
      setNotice(result.message)
      document.getElementById('import')?.scrollIntoView({ block: 'nearest' })
    })
  const identity = view.current_model
  const serverRunning = view.serving
  const ownedRunning = view.owned_server
  // Every package the CLI found in the download folders and the Import folders,
  // each with its own folder. The same package in two folders is two rows. An
  // incomplete package is listed too: it says so and cannot be selected.
  const installed = [
    ...preparedRows(view)
      .filter((model) => model.prepared_state !== 'not_prepared')
      .map((model) => {
        const start = preparedStart(model)
        return {
          name: model.display_name,
          size: model.size_bytes,
          path: start?.path,
          state: model.prepared_state,
          id: model.id,
          kind: start?.kind ?? ('manifest' as const),
          catalogued: true,
          reason: '',
          estimate: models.catalogue?.find((entry) => entry.id === model.id)
            ?.estimate,
          runnable: model.prepared_state === 'prepared' && !!start,
          vision: model.capabilities.mmproj,
          repairOutput: model.repair_output,
        }
      }),
    // A loose model file the CLI found in an Import folder. Part files (sidecars:
    // a vision projector, an MTP head) are never models of their own.
    ...(models.scanned ?? [])
      .filter((model) => model.kind !== 'sidecar')
      .map((model) => ({
        name: model.model_name || model.file_name,
        size: model.size_bytes,
        path: model.path,
        state: '',
        id: model.path,
        kind: 'file' as const,
        catalogued: false,
        reason: model.reason,
        estimate: model.estimate,
        runnable: model.runnable,
        vision: false,
        repairOutput: undefined,
      })),
  ]
  type Installed = (typeof installed)[number]
  const runsFromPath = (model: Installed) =>
    !!model.path &&
    (samePath(model.path, identity?.manifest_path) ||
      samePath(model.path, identity?.model_path))
  const runningListed = installed.some(
    (model) => model.catalogued && runsFromPath(model),
  )
  const flags = (model: Installed) => ({
    selected:
      !!model.path &&
      (samePath(model.path, view.saved.model_path) ||
        samePath(model.path, view.saved.manifest_path)),
    inUse:
      serverRunning &&
      !!identity &&
      (model.catalogued
        ? // The running server names its package by path: two folders can hold
          // the same package, and only the one that runs is in use. The
          // catalogue id decides only when no listed package has that path.
          runningListed
          ? runsFromPath(model)
          : model.id === identity.catalogue_id
        : samePath(model.path, identity.model_path)),
  })
  // The package the app reports as current: by path when a listed package has
  // it (two folders can hold the same package), else by catalogue id.
  const isCurrent = (model: Installed) =>
    !!identity &&
    (runningListed ? runsFromPath(model) : model.id === identity.catalogue_id)
  // One word for a package that is not whole, and one size rule: no size.
  const incomplete = (model: Installed) =>
    model.state ? model.state !== 'prepared' : !model.runnable
  // A download in progress for this package: the row is not whole yet.
  const downloading = (model: Installed) =>
    models.download?.id === model.id &&
    ['running', 'planning', 'cancelling'].includes(models.download.state)
  // A row that cannot be selected shows no fit text. The saved setup's fit
  // belongs to the current or selected row; any other row has its own estimate
  // (not while the server holds the memory).
  const rowSubject = (model: Installed) => {
    const { selected, inUse } = flags(model)
    if (inUse || downloading(model) || !model.runnable || !model.path)
      return undefined
    if (isCurrent(model) || selected || serverRunning) return undefined
    return estimateSubject({ kind: model.kind, path: model.path })
  }
  const estimates = useQueries({
    queries: installed.map((model, index) => {
      const subject = rowSubject(model)
      return {
        queryKey: ['row-fit', subject ?? `off-${index}`],
        enabled: subject !== undefined,
        queryFn: async () => {
          const result = await actions.Estimate(subject!, 0, 0)
          requireActionSuccess(result)
          return {
            class: result.fit?.class ?? '',
            sentence: result.fit?.sentence ?? '',
          }
        },
        staleTime: 30_000,
        retry: false,
      }
    }),
  })
  const fits = installed.map((model, index): RowFit => {
    const { selected, inUse } = flags(model)
    if (inUse || downloading(model) || !model.runnable || !model.path)
      return { kind: 'none' }
    if (isCurrent(model) || selected) return answered(view.fit)
    if (serverRunning) return { kind: 'none' }
    const estimate = estimates[index]!
    if (estimate.isPending) return { kind: 'checking' }
    return answered(estimate.data)
  })
  // A column with no cell to show is not drawn.
  const showFit = fits.some((fit) => fit.kind !== 'none')
  const showNote = fits.some((fit) => fit.kind === 'unknown')
  // One word under the name: the row's own state.
  const statusOf = (model: Installed, inUse: boolean, selected: boolean) =>
    downloading(model)
      ? 'Downloading'
      : inUse
        ? 'In use'
        : selected
          ? 'Selected for next start'
          : model.state
            ? installedWord(model.state)
            : model.runnable
              ? ''
              : 'Incomplete'
  const showAction = installed.some((model) => !flags(model).inUse)
  return (
    <ModelDownloadFlow view={view}>
      {(open, pending) => (
        <div className="space-y-8">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-sm text-muted-foreground">
              {(!models.reason_code && models.reason) ||
                (models.refreshing
                  ? 'Refreshing models…'
                  : 'Models on this computer and supported downloads.')}
            </p>
            <Button
              variant="outline"
              disabled={command.isPending || models.refreshing}
              onClick={() => command.mutate(actions.RefreshModels)}
            >
              Check for model files
            </Button>
            <ImportMenu disabled={command.isPending} browse={browse} />
          </div>
          {models.reason_code && (
            <GUIError
              error={{
                code: models.reason_code,
                remedy: models.reason || 'No further details were reported.',
              }}
            />
          )}
          <ModelDownloadProgress view={view} retry={open} />
          <ActionError error={command.error} />
          <section aria-label="Installed models" className="space-y-3">
            <h2 className="text-xl">Installed</h2>
            {!installed.length ? (
              <p className="text-sm text-muted-foreground">
                No models installed yet.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model / Path</TableHead>
                    <TableHead className="text-right">Size</TableHead>
                    {showFit && <TableHead>Memory fit</TableHead>}
                    {showAction && <TableHead>Action</TableHead>}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {installed.map((model, index) => {
                    const { selected, inUse } = flags(model)
                    const status = statusOf(model, inUse, selected)
                    const fit = fits[index]!
                    return (
                      <TableRow key={`${model.kind}-${model.path ?? model.id}`}>
                        <TableCell className="whitespace-normal">
                          <p>{model.name}</p>
                          {status && (
                            <p className="text-xs text-muted-foreground">
                              {status}
                            </p>
                          )}
                          {model.reason && (
                            <div className="text-xs text-muted-foreground">
                              <details>
                                <summary>Details</summary>
                                <p>{model.reason}</p>
                              </details>
                            </div>
                          )}
                          {model.vision && (
                            <p className="text-xs text-muted-foreground">
                              Reads images
                            </p>
                          )}
                          <div className="mt-1 space-y-1 text-muted-foreground">
                            <div className="max-w-lg">
                              {model.path ? (
                                <PathText path={folderPath(model.path)} />
                              ) : (
                                <p className="text-xs">Folder not reported</p>
                              )}
                            </div>
                            {model.path && (
                              <div className="-ml-2 flex flex-wrap items-center">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="px-2"
                                  disabled={command.isPending}
                                  onClick={() =>
                                    command.mutate(() =>
                                      actions.OpenModelFolder(
                                        folderPath(model.path!),
                                      ),
                                    )
                                  }
                                >
                                  Open folder
                                </Button>
                                <CopyText
                                  label="Copy folder path"
                                  text={folderPath(model.path)}
                                  compact
                                  variant="ghost"
                                />
                                {/* The CLI names the --output of a repair; no row
                                    name, no button. */}
                                {model.repairOutput && !inUse && (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="px-2"
                                    disabled={command.isPending || pending}
                                    onClick={() =>
                                      open(
                                        model.id,
                                        'repair',
                                        model.repairOutput,
                                      )
                                    }
                                  >
                                    Check or repair files
                                  </Button>
                                )}
                              </div>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {/* An incomplete package has not the size it reports. */}
                          {incomplete(model) ? '—' : bytes(model.size)}
                        </TableCell>
                        {showFit && (
                          <TableCell className="whitespace-normal text-sm">
                            {fit.kind === 'checking' ? (
                              <span className="text-muted-foreground">
                                Checking memory…
                              </span>
                            ) : fit.kind === 'unknown' ? (
                              'Not checked'
                            ) : fit.kind === 'sentence' ? (
                              fit.text
                            ) : null}
                          </TableCell>
                        )}
                        {showAction && (
                          <TableCell>
                            {!inUse && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-auto min-h-8 whitespace-nowrap px-2 py-1"
                                disabled={
                                  command.isPending ||
                                  downloading(model) ||
                                  !model.runnable ||
                                  !model.path ||
                                  selected
                                }
                                onClick={() =>
                                  command.mutate(() =>
                                    actions.PatchPreferences(
                                      modelPreferencePatch(
                                        model.kind,
                                        model.path!,
                                      ),
                                    ),
                                  )
                                }
                              >
                                Select for Server
                              </Button>
                            )}
                          </TableCell>
                        )}
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            )}
            {showNote && (
              <p className="text-sm text-muted-foreground">
                Bubo has no memory estimate for these models yet; you can still
                select them and start the server.
              </p>
            )}
            {installed.some((model) => model.reason) && (
              <p className="text-sm text-muted-foreground">
                {guiError({ code: 'model_unusable', remedy: '' }).message}
              </p>
            )}
          </section>
          <section
            id="import"
            aria-label="Import models"
            className="space-y-3 scroll-mt-6"
          >
            <h2 className="text-xl">Import</h2>
            <p className="text-sm text-muted-foreground">
              Use a model file or a model folder already on your drive.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                disabled={command.isPending}
                onClick={() => browse('file')}
              >
                Browse model file…
              </Button>
              <Button
                variant="outline"
                disabled={command.isPending}
                onClick={() => browse('folder')}
              >
                Browse model folder…
              </Button>
            </div>
            {notice && (
              <p className="text-sm" role="status">
                {notice}{' '}
                <Link to="/" className="underline underline-offset-4">
                  Open Server
                </Link>
              </p>
            )}
            {(models.folders ?? []).map((folder) => (
              <div key={folder} className="flex items-center gap-2">
                <code className="min-w-0 break-all text-xs">{folder}</code>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={command.isPending}
                  onClick={() =>
                    command.mutate(() => actions.RemoveModelFolder(folder))
                  }
                >
                  Remove folder
                </Button>
              </div>
            ))}
            {models.scan_warnings?.map((warning, index) => (
              <div key={index} className="text-sm">
                <p>
                  {
                    guiError({
                      code: 'model_folder_warning',
                      remedy: warning.reason,
                    }).message
                  }
                </p>
                <details>
                  <summary>Details</summary>
                  <p>
                    {warning.folder}: {warning.reason}
                  </p>
                </details>
              </div>
            ))}
          </section>
          <section aria-label="Discover models" className="space-y-3">
            <h2 className="text-xl">Discover</h2>
            {ownedRunning && (
              <p className="text-sm text-muted-foreground">
                Bubo's server holds the memory now. Stop it to switch models.
              </p>
            )}
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead>Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(models.catalogue ?? []).map((entry) => {
                  const status = packageStatus(entry.prepared_state)
                  // Prepared status and catalogue identity come from the CLI;
                  // the catalogue and local scan may have different refresh ages.
                  const local = installed.find(
                    (model) =>
                      model.catalogued &&
                      model.id === entry.id &&
                      model.runnable,
                  )
                  // Installed reads the list of prepared packages only.
                  const isInstalled = !!local
                  const inUse =
                    serverRunning &&
                    !!identity &&
                    (identity.catalogue_id === entry.id ||
                      (!!local?.path &&
                        (identity.manifest_path === local.path ||
                          identity.model_path === local.path)))
                  const selected =
                    !!local?.path &&
                    (local.path === view.saved.manifest_path ||
                      local.path === view.saved.model_path)
                  // The model Server already uses or has selected is not offered again.
                  const current = entry.id === identity?.catalogue_id
                  const downloading =
                    models.download?.id === entry.id &&
                    ['running', 'planning', 'cancelling'].includes(
                      models.download.state,
                    )
                  return (
                    <TableRow key={entry.id}>
                      <TableCell className="min-w-44 whitespace-normal">
                        {entry.display_name}
                        <ModelInfo entry={entry} />
                        {!entry.download_enabled && entry.disabled_reason && (
                          <p className="text-xs text-muted-foreground">
                            {entry.disabled_reason}
                          </p>
                        )}
                        {isInstalled && (
                          <p className="text-xs text-muted-foreground">
                            Installed
                          </p>
                        )}
                        {/* "Not downloaded" is what the Download button says. */}
                        {!isInstalled &&
                          (downloading || status.action !== 'download') && (
                            <p className="text-xs text-muted-foreground">
                              {downloading ? 'Downloading' : status.word}
                            </p>
                          )}
                        {entry.role === 'recommended' &&
                          entry.default_status === 'default' && (
                            <p className="text-xs text-muted-foreground">
                              Default
                            </p>
                          )}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {bytes(entry.download_bytes)}
                      </TableCell>
                      <TableCell>
                        {isInstalled &&
                        (inUse || current || selected) ? null : isInstalled ? (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={command.isPending || !local?.path}
                            onClick={() =>
                              command.mutate(async () => {
                                requireActionSuccess(
                                  await actions.PatchPreferences(
                                    modelPreferencePatch(
                                      local!.kind,
                                      local!.path!,
                                    ),
                                  ),
                                )
                                setNotice(
                                  `${entry.display_name} selected for next start.`,
                                )
                              })
                            }
                          >
                            Select for Server
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={
                              pending ||
                              !entry.download_enabled ||
                              command.isPending ||
                              downloading ||
                              status.action === 'none'
                            }
                            onClick={() =>
                              open(
                                entry.id,
                                status.action === 'check' ||
                                  status.action === 'repair'
                                  ? 'repair'
                                  : 'download',
                              )
                            }
                          >
                            {downloading ? 'Downloading' : status.actionLabel}
                          </Button>
                        )}
                        {isInstalled && !local?.path && (
                          <p
                            data-code="not_reported"
                            className="text-xs text-muted-foreground"
                          >
                            Model path not reported. Check for model files.
                          </p>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
            {!models.catalogue?.length && (
              <p className="text-sm text-muted-foreground">
                No supported downloads reported.
              </p>
            )}
          </section>
        </div>
      )}
    </ModelDownloadFlow>
  )
}
