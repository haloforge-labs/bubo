import { links } from '@/lib/links'
import { Page } from '@/components/page'
import { DesktopView } from '@/components/desktop-view'
import { bytes, recommendedModel } from '@/components/model-facts'
import {
  ModelDownloadFlow,
  ModelDownloadProgress,
} from '@/components/model-download'
import { Preparation } from '@/components/preparation'
import { Button } from '@/components/ui/button'
import type { View } from '@/lib/bindings'
import { Link } from '@tanstack/react-router'
import { actions } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Lifecycle } from '@/components/server-lifecycle'
import { GUIError } from '@/components/gui-error'
import {
  ChangeDownloadFolder,
  driveName,
  FolderLocked,
} from '@/components/download-folder'
import { AboutPopup } from '@/components/about-popup'
import { Disclosure } from '@/components/disclosure'
import { downloadFolderScope } from '@/lib/presentation'
import { useState } from 'react'

type MemoryEstimate = NonNullable<
  NonNullable<
    NonNullable<View['models']>['catalogue']
  >[number]['estimate']['memory_estimate']
>

// What a choice needs of the GPU, said once per place it is shown.
function needsGpuMemory(memory: MemoryEstimate) {
  if (!memory.gpu) return ''
  const need = bytes(memory.gpu.need.bytes, { unit: 'GiB', digits: 0 })
  return `Needs about ${need} of GPU memory at ${memory.context_tokens.toLocaleString()} tokens`
}

export default function OnboardingPage() {
  return (
    <Page title="Get started">
      <DesktopView>{(view) => <OnboardingContent view={view} />}</DesktopView>
    </Page>
  )
}

function OnboardingContent({ view }: { view: View }) {
  const check = useActionTask()
  const recommended = recommendedModel(view)
  const choices = (view.models?.catalogue ?? []).filter((entry) =>
    recommended?.family_group
      ? entry.family_group === recommended.family_group
      : recommended &&
        entry.architecture === recommended.architecture &&
        !entry.family_group,
  )
  const [chosenId, setChosenId] = useState<string>()
  const model =
    choices.find(
      (entry) => entry.id === (chosenId ?? view.models?.download?.id),
    ) ?? recommended
  const selectorShown = !view.runtime.readiness.model && choices.length > 0
  const chosenMemory = model?.estimate.memory_estimate
  const runtimeReady = view.runtime.readiness.runtime
  const modelReady = view.runtime.readiness.model
  const activeComponent =
    view.phase === 'downloading_model'
      ? 'model'
      : view.phase === 'preparing_engine'
        ? 'runtime'
        : undefined
  const busy =
    view.phase === 'preparing_engine' || view.phase === 'downloading_model'
  // The engine's size is known once the CLI has made its plan.
  const engineBytes = view.preparation
    .filter((plan) => plan.kind === 'runtime')
    .flatMap((plan) => plan.files)
    .reduce((total, file) => total + file.size, 0)
  const disk = view.models?.disk
  const neededBytes = model ? model.download_bytes + engineBytes : undefined
  const shortOfSpace =
    !busy &&
    !modelReady &&
    neededBytes != null &&
    disk?.free_bytes != null &&
    neededBytes > disk.free_bytes
  if (view.phase === 'cli_missing')
    return (
      <GUIError
        error={{ code: 'cli_missing', remedy: view.failure?.remedy ?? '' }}
      />
    )
  return (
    <ModelDownloadFlow view={view}>
      {(open, pending) => {
        return (
          <div className="max-w-3xl space-y-6">
            {!(runtimeReady && modelReady) && (
              <p>
                Needs a Strix Halo computer running Windows. Download the model
                below, or try other models on the{' '}
                <Link to="/models" className="underline">
                  Models page
                </Link>
                .
              </p>
            )}
            <GUIError
              error={
                view.failure ||
                (view.phase === 'failed' && view.reason && view.remedy
                  ? { code: view.reason, remedy: view.remedy }
                  : undefined)
              }
              recovery={view.failure?.code === 'download_stalled'}
              retry={
                view.failure?.code === 'download_stalled'
                  ? actions.Prepare
                  : undefined
              }
            />
            <ol aria-label="Setup steps" className="space-y-3 border-y py-4">
              <li className="space-y-3">
                <p>
                  <span className="tabular-nums">1</span> Engine files ·{' '}
                  {activeComponent === 'runtime'
                    ? 'In progress'
                    : runtimeReady
                      ? 'Done ✓'
                      : 'Next'}
                  {engineBytes > 0 && (
                    <>
                      {' '}
                      ·{' '}
                      <span className="tabular-nums">{bytes(engineBytes)}</span>
                    </>
                  )}
                </p>
                <Preparation
                  view={view}
                  primary={!runtimeReady}
                  hideRepair
                  progressKind="runtime"
                />
              </li>
              <li className="space-y-3">
                <p>
                  <span className="tabular-nums">2</span>{' '}
                  {/* The selector below names the choice. */}
                  {!selectorShown && model
                    ? `${model.display_name}${model.id === recommended?.id ? ' (default model)' : ''}`
                    : 'Model'}{' '}
                  ·{' '}
                  {activeComponent === 'model'
                    ? 'In progress'
                    : modelReady
                      ? 'Done ✓'
                      : runtimeReady
                        ? 'Next'
                        : 'After step 1'}
                  {model && (
                    <>
                      {' '}
                      ·{' '}
                      <span className="tabular-nums">
                        {bytes(model.download_bytes)}
                      </span>
                    </>
                  )}
                </p>
                {selectorShown && (
                  <div className="space-y-2">
                    <label htmlFor="setup-model" className="sr-only">
                      Model to download
                    </label>
                    <select
                      id="setup-model"
                      className="w-full min-w-0 border bg-background px-3 py-2 text-sm"
                      value={model?.id ?? ''}
                      disabled={busy || pending}
                      onChange={(event) => setChosenId(event.target.value)}
                    >
                      {choices.map((entry) => (
                        <option
                          key={entry.id}
                          value={entry.id}
                          disabled={!entry.download_enabled}
                        >
                          {entry.selection_label || entry.display_name}
                          {!entry.download_enabled && entry.disabled_reason
                            ? ` — ${entry.disabled_reason}`
                            : ''}
                        </option>
                      ))}
                    </select>
                    {chosenMemory?.gpu && (
                      <p className="text-sm text-muted-foreground">
                        {needsGpuMemory(chosenMemory)}
                      </p>
                    )}
                    {choices.length > 1 && (
                      <Disclosure
                        summary="Compare models"
                        className="text-sm text-muted-foreground"
                      >
                        <ul aria-label="Download choices" className="space-y-2">
                          {choices.map((entry) => (
                            <li key={entry.id}>
                              {entry.selection_label || entry.display_name}:
                              Download {bytes(entry.download_bytes)}
                              {entry.estimate.memory_estimate?.gpu && (
                                <>
                                  {' '}
                                  ·{' '}
                                  {needsGpuMemory(
                                    entry.estimate.memory_estimate,
                                  )}
                                </>
                              )}
                            </li>
                          ))}
                        </ul>
                      </Disclosure>
                    )}
                  </div>
                )}
                <ModelDownloadProgress view={view} retry={open} />
                {!view.models?.download &&
                  view.phase === 'downloading_model' && (
                    <Preparation
                      view={view}
                      progressKind="model"
                      progressOnly
                    />
                  )}
                {runtimeReady &&
                  !modelReady &&
                  model?.download_enabled &&
                  !busy && (
                    <Button
                      disabled={
                        pending || !view.actions.download_default.allowed
                      }
                      onClick={() => open(model.id)}
                    >
                      Download model…
                    </Button>
                  )}
                {!modelReady && (
                  <Link
                    to="/models"
                    hash="import"
                    className="block text-sm underline"
                  >
                    I already have a model
                  </Link>
                )}
                {runtimeReady && view.models?.status === 'not_reported' && (
                  <div className="space-y-2">
                    <GUIError
                      error={{
                        code: view.models.reason_code,
                        remedy: view.models.reason,
                      }}
                      recovery={false}
                    />
                    <Button
                      variant="outline"
                      disabled={check.isPending || view.models.refreshing}
                      onClick={() => check.mutate(actions.RefreshModels)}
                    >
                      Check for model files
                    </Button>
                    <ActionError error={check.error} />
                  </div>
                )}
                {shortOfSpace && (
                  <div className="space-y-2 text-sm">
                    <p>
                      Needs {bytes(neededBytes!)} free;{' '}
                      {driveName(disk!.drive) || 'This drive'} has{' '}
                      {bytes(disk!.free_bytes!)}.
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      <ChangeDownloadFolder view={view}>
                        Choose download folder…
                      </ChangeDownloadFolder>
                      <AboutPopup label="download folder">
                        <p>{downloadFolderScope}</p>
                      </AboutPopup>
                    </div>
                    <FolderLocked folder={view.download_folder} />
                  </div>
                )}
              </li>
            </ol>
            {runtimeReady && modelReady && (
              <section aria-label="Memory right now" className="space-y-2">
                <h3 className="text-lg">Memory right now</h3>
                <p aria-label="Memory fit">{view.fit.sentence}</p>
                <div className="flex flex-wrap items-start gap-3">
                  {runtimeReady && modelReady && <Lifecycle view={view} />}
                  {view.fit.class !== 'memory_does_not_fit_any' && (
                    <Button
                      variant="outline"
                      disabled={check.isPending || view.models?.refreshing}
                      onClick={() => check.mutate(actions.RefreshModels)}
                    >
                      Check again
                    </Button>
                  )}
                </div>
                {runtimeReady && modelReady && (
                  <Link className="block underline" to="/">
                    Show server setup and addresses
                  </Link>
                )}
                <ActionError error={check.error} />
              </section>
            )}
            <a
              href={links.docs}
              className="block text-sm underline"
              onClick={(event) => {
                event.preventDefault()
                void actions.OpenExternal(links.docs)
              }}
            >
              Install guide
            </a>
          </div>
        )
      }}
    </ModelDownloadFlow>
  )
}
