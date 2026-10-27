import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { bytes, recommendedModel, shortByShown } from '@/components/model-facts'
import { plannedTransferBytes, PlanSummary } from '@/components/model-download'
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
import { GUIError } from '@/components/gui-error'
import {
  CurrentFileDetails,
  TransferProgress,
} from '@/components/transfer-progress'

export function Preparation({
  view,
  primary = false,
  hideRepair = false,
  progressKind = view.phase === 'downloading_model' ? 'model' : 'runtime',
  progressOnly = false,
}: {
  view: View
  primary?: boolean
  hideRepair?: boolean
  progressKind?: 'runtime' | 'model'
  progressOnly?: boolean
}) {
  const command = useActionTask()
  const plans = view.preparation ?? []
  const orderedPlans = [...plans].sort((a, b) =>
    a.kind === b.kind ? 0 : a.kind === 'runtime' ? -1 : 1,
  )
  const combined = plans.length > 1
  const files = plans.flatMap((plan) => plan.files)
  const statesReported = plans.every(
    (plan) =>
      plan.kind === 'runtime' || plan.files.every((file) => !!file.state),
  )
  const transferBytes = plans.reduce(
    (total, plan) => total + plannedTransferBytes(plan),
    0,
  )
  const plannedBytes = files.reduce((total, file) => total + file.size, 0)
  // The CLI reports writes and one safety reserve per component. A combined
  // download shares the largest reserve on each destination volume.
  const combinedSpace = new Map<
    string,
    { writes: number; reserve: number; free: number }
  >()
  const spaceReported = plans.every(
    (plan) =>
      plan.volumes.length > 0 &&
      plan.volumes.every(
        (volume) =>
          volume.bytes_to_write != null && volume.safety_margin_bytes != null,
      ),
  )
  if (spaceReported)
    for (const plan of plans)
      for (const volume of plan.volumes) {
        const previous = combinedSpace.get(volume.name)
        combinedSpace.set(volume.name, {
          writes: (previous?.writes ?? 0) + volume.bytes_to_write!,
          reserve: Math.max(
            previous?.reserve ?? 0,
            volume.safety_margin_bytes!,
          ),
          free: Math.min(
            previous?.free ?? volume.free_bytes,
            volume.free_bytes,
          ),
        })
      }
  const combinedShort = [...combinedSpace.values()].some(
    (volume) => volume.writes + volume.reserve > volume.free,
  )
  const model = recommendedModel(view)
  const licence = model?.licence
  const busy =
    (progressKind === 'runtime'
      ? view.phase === 'preparing_engine'
      : view.phase === 'downloading_model') && !view.approval
  if (progressOnly && !busy) return null
  return (
    <section aria-label="Prepare or repair Bubo" className="space-y-3">
      <ActionError error={command.error} />
      {busy ? (
        <>
          {/* The status label already says "Downloading …". */}
          {view.progress?.phase?.includes('verif') && (
            <h2 className="font-medium">Checking the downloaded files</h2>
          )}
          {view.progress && (
            <>
              {/* One bar and one line for the whole job; Go adds up the files. */}
              <TransferProgress
                identity={
                  view.progress.overall
                    ? `prepare-${view.generation}`
                    : `prepare-${view.generation}-${view.progress.component}-${view.progress.phase}-${view.progress.file_id}`
                }
                done={
                  view.progress.overall?.bytes_done ?? view.progress.bytes_done
                }
                total={
                  view.progress.overall?.bytes_total ??
                  view.progress.bytes_total
                }
                scope={view.progress.overall ? undefined : 'Current file'}
                whole={!!view.progress.overall}
              />
              {/* Without Go's whole-job number the line is the whole statement: no per-file bar. */}
              {view.progress.overall && (
                <>
                  <Progress
                    aria-label="Preparation progress"
                    value={
                      (view.progress.overall?.bytes_total ??
                      view.progress.bytes_total)
                        ? ((view.progress.overall?.bytes_done ??
                            view.progress.bytes_done) /
                            (view.progress.overall?.bytes_total ??
                              view.progress.bytes_total!)) *
                          100
                        : undefined
                    }
                  />
                </>
              )}
            </>
          )}
          <Button
            variant="outline"
            disabled={command.isPending}
            onClick={() => command.mutate(actions.Cancel)}
          >
            Cancel download
          </Button>
          {view.progress && (
            <CurrentFileDetails
              position={view.preparation_file}
              done={view.progress.bytes_done}
              total={view.progress.bytes_total}
              plan={plans.find((plan) => plan.kind === progressKind)}
              fileId={view.progress.file_id}
              phase={view.progress.phase}
            />
          )}
        </>
      ) : !progressOnly &&
        !view.approval &&
        (!hideRepair || !view.runtime.readiness.runtime) ? (
        <Button
          variant={primary ? 'default' : 'outline'}
          disabled={
            command.isPending ||
            view.preparation_pending ||
            !view.actions.prepare.allowed
          }
          onClick={() => command.mutate(actions.Prepare)}
        >
          {view.preparation_pending
            ? 'Reading preparation plan…'
            : !view.runtime.readiness.runtime
              ? 'Download the engine files'
              : 'Check or repair Bubo files'}
        </Button>
      ) : null}
      {!busy &&
        !progressOnly &&
        !view.approval &&
        !view.actions.prepare.allowed &&
        view.actions.prepare.reason_code !== 'busy' &&
        view.actions.prepare.reason_code &&
        view.actions.prepare.reason_code !== 'not_applicable' && (
          <GUIError
            error={{
              code: view.actions.prepare.reason_code,
              remedy: view.actions.prepare.reason,
            }}
            recovery={false}
          />
        )}
      {view.plan_error && <GUIError error={view.plan_error} recovery={false} />}
      <Dialog
        open={!progressOnly && !!view.approval}
        onOpenChange={(open) => {
          if (!open && !command.isPending) command.mutate(actions.Decline)
        }}
      >
        <DialogContent
          showCloseButton={false}
          className="flex max-h-[90dvh] min-w-0 flex-col overflow-hidden"
        >
          <DialogHeader className="shrink-0">
            <DialogTitle>
              {combined
                ? 'Download engine files and model'
                : plans.some((plan) => plan.kind === 'model') && model
                  ? `Download ${model.display_name}`
                  : 'Download the engine files'}
            </DialogTitle>
            {/* Read by screen readers only: the dialog shows the same facts. */}
            <DialogDescription className="sr-only">
              {combined
                ? 'This starts the engine files, then the model.'
                : 'Review the download and space needed.'}
            </DialogDescription>
          </DialogHeader>
          <div
            data-download-body
            className="min-h-0 min-w-0 space-y-4 overflow-y-auto [overflow-wrap:anywhere]"
          >
            {combined && (
              <div className="space-y-1 text-sm">
                <p>
                  {statesReported ? 'To download:' : 'Planned files:'}{' '}
                  <span className="tabular-nums">
                    {bytes(statesReported ? transferBytes : plannedBytes)}
                  </span>
                </p>
                {spaceReported ? (
                  [...combinedSpace].map(([name, volume]) => {
                    const needed = volume.writes + volume.reserve
                    const shortfall = Math.max(0, needed - volume.free)
                    return (
                      <div key={name}>
                        <p>
                          {/^[A-Za-z]:[\\/]/.test(name) ? name : 'Destination'}:
                          free space needed{' '}
                          <span className="tabular-nums">
                            {bytes(volume.writes + volume.reserve)}
                          </span>
                          {' · '}
                          <span className="tabular-nums">
                            {bytes(volume.free)}
                          </span>{' '}
                          free
                        </p>
                        {shortfall > 0 && (
                          <p>
                            Free up space on{' '}
                            {/^[A-Za-z]:/.test(name)
                              ? name.slice(0, 2)
                              : 'the destination drive:'}{' '}
                            {shortByShown(needed, volume.free)} more is needed.
                            Then try again.
                          </p>
                        )}
                      </div>
                    )
                  })
                ) : (
                  <p data-code="not_reported">
                    Combined free space needed was not reported.
                  </p>
                )}
              </div>
            )}
            {orderedPlans.map((plan, index) => (
              <section
                key={`${plan.plan_digest}-${index}`}
                className="space-y-2"
              >
                {combined && (
                  <h3 className="font-medium">
                    {plan.kind === 'runtime' ? 'Engine files' : 'Model files'}
                  </h3>
                )}
                <PlanSummary
                  plan={plan}
                  licence={plan.kind === 'model' ? licence : undefined}
                  showSpace={!combined}
                />
              </section>
            ))}
            {!plans.length && (
              <p data-code="not_reported">
                Bubo did not return a preparation plan.
              </p>
            )}
            <ActionError error={command.error} />
          </div>
          <DialogFooter className="shrink-0 bg-background pt-3">
            <Button
              variant="outline"
              disabled={command.isPending}
              onClick={() => command.mutate(actions.Decline)}
            >
              Not now
            </Button>
            <Button
              disabled={
                command.isPending ||
                !view.approval?.plan_digest ||
                !plans.length ||
                plans.some(
                  (plan) =>
                    !plan.enough_space ||
                    !Object.values(plan.paths ?? {}).length ||
                    !(
                      plan.licences.some((item) => item.text.trim()) ||
                      ((plan.licence ?? licence)?.url &&
                        (plan.licence ?? licence)?.name)
                    ),
                ) ||
                (combined && (!spaceReported || combinedShort))
              }
              onClick={() => command.mutate(actions.Approve)}
            >
              {combined
                ? 'Download engine and model'
                : plans[0]?.kind === 'runtime'
                  ? 'Download engine files'
                  : 'Download and install'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}
