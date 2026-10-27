import { useEffect, useRef, useState } from 'react'
import { ActionButton, ActionError } from '@/components/action-feedback'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { actions, type View } from '@/lib/bindings'
import { bytes, updatePresentation } from '@/lib/format'
import { TransferProgress } from '@/components/transfer-progress'
import { GUIError } from '@/components/gui-error'
import { checkFailureSentence, guiError } from '@/lib/gui-errors'
import { ErrorNote } from '@/components/error-note'
import { useActionTask } from '@/lib/action-task'
import { Button } from '@/components/ui/button'

export function UpdateCard({
  update,
  availability,
  serverRunning = true,
}: {
  update: View['update']
  availability?: View['actions']
  serverRunning?: boolean
}) {
  const state = updatePresentation(update)
  // The update runs: Later would leave a half-finished choice.
  const [busy, setBusy] = useState({ download: false, install: false })
  const progress = update.progress
  // The updater could not start, or a check failed. The user started no update.
  const checkFailed = state === 'failed' && update.check_failed
  const checkTask = useActionTask()
  const resetCheck = checkTask.reset
  const previousError = useRef(update.error)
  useEffect(() => {
    if (previousError.current && !update.error) resetCheck()
    previousError.current = update.error
  }, [update.error, resetCheck])
  // The action and the refreshed update view describe the same check. Show
  // its cause in one place, even while the view is catching up with the action.
  const checkError =
    checkTask.error ??
    (checkFailed || state === 'not_checked' ? update.error : undefined)
  // A check that cannot run (the updater did not start) shows no dead button;
  // the one sentence above says why.
  const checkButton =
    (state === 'none' ||
      state === 'current' ||
      state === 'failed' ||
      state === 'not_checked') &&
    !(checkFailed && availability && !availability.check.allowed)
  const updated = !!update.last_result_notice && state === 'current'
  // A failed result whose saved notice is the error box's own sentence (an
  // interrupted update) is said once, in the box.
  const notice =
    update.last_result_notice &&
    state !== 'rollback_pending' &&
    !(
      state === 'failed' &&
      update.error &&
      guiError(update.error).message === update.last_result_notice
    )
      ? update.last_result_notice
      : ''
  const result = !!update.last_result_notice && state !== 'rollback_pending'
  const offer = ['available', 'ready', 'failed'].includes(state) && !checkFailed
  const dismiss =
    result && (state === 'failed' || !offer) ? 'Dismiss' : offer ? 'Later' : ''
  const labels: Record<string, string> = {
    none: '',
    current: 'Bubo is up to date.',
    available: '',
    progress: 'Downloading and checking the update.',
    ready: 'The update is downloaded and checked.',
    failed: '',
    rollback_pending: "Update didn't finish.",
    not_checked: 'Updates have not been checked.',
  }
  return (
    <Card data-update={state}>
      <CardHeader>
        <CardTitle>Updates</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* After an update the one line is the result: "Updated to 0.1.2." */}
        {checkError ? (
          <ErrorNote>
            <p>{checkFailureSentence(checkError)}</p>
          </ErrorNote>
        ) : labels[state] !== '' && !updated ? (
          <p>{labels[state] ?? 'Update state unavailable.'}</p>
        ) : null}
        {/* Two versions are shown only when they are being compared. */}
        {update.available_version && state !== 'rollback_pending' && (
          <p className="text-sm text-muted-foreground">
            Installed:{' '}
            <span className="tabular-nums">{update.current_version}</span> ·
            Available:{' '}
            <span className="tabular-nums">{update.available_version}</span>
          </p>
        )}
        {notice && (
          <p role="status">{notice.replace(/candidate/gi, 'new version')}</p>
        )}
        {(offer ||
          state === 'progress' ||
          update.runtime_followup_required) && (
          <div className="space-y-1 text-sm">
            {update.runtime_download_bytes === 0 && (
              <p>Engine files: not redownloaded.</p>
            )}
            {update.models_download_bytes === 0 && (
              <p>Models: not redownloaded.</p>
            )}
            {update.size_bytes != null && (
              <p>Download size: {bytes(update.size_bytes)}.</p>
            )}
          </div>
        )}
        {update.runtime_followup_required && (
          <GUIError
            error={{
              code: 'runtime_missing',
              remedy:
                'Prepare the engine files for this Bubo version before starting.',
            }}
          />
        )}
        {state !== 'not_checked' &&
          state !== 'rollback_pending' &&
          !checkFailed &&
          !checkError && (
            <GUIError error={update.error} recovery={state !== 'failed'} />
          )}
        {checkError && (
          <details className="text-xs text-muted-foreground">
            <summary>Details</summary>
            <p>{checkError.remedy}</p>
          </details>
        )}
        {!(state in labels) && (
          <ActionError
            error={{
              code: 'internal_error',
              remedy: `Unknown update state: ${state}`,
            }}
          />
        )}
        {state === 'progress' && (
          <div className="space-y-2">
            <Progress
              aria-label="Update download progress"
              value={
                progress?.bytes_total
                  ? (progress.bytes_done / progress.bytes_total) * 100
                  : null
              }
            />
            {progress && (
              <TransferProgress
                identity={`update-${update.available_version}-${progress.phase}`}
                done={progress.bytes_done}
                total={progress.bytes_total}
              />
            )}
          </div>
        )}
        {state === 'ready' && (
          <p>
            Installing stops the server and restarts this app. Your models and
            settings stay.
          </p>
        )}
        {/* The sentence and its one Recover button share one box, as on Server. */}
        {state === 'rollback_pending' && (
          <GUIError
            error={{
              code: 'rollback_pending',
              remedy: update.error?.remedy ?? '',
              label: update.previous_version || undefined,
              serverStopped: !serverRunning,
            }}
            recoverAction={actions.RecoverUpdate}
          />
        )}
        <div className="flex flex-wrap gap-3">
          {state === 'failed' &&
            ['stale_manifest', 'filesystem'].includes(
              update.error?.code || '',
            ) && (
              <ActionButton
                variant="outline"
                action={actions.ClearReleaseHistory}
              >
                Clear saved update history
              </ActionButton>
            )}
          {checkButton && (
            <Button
              variant={state === 'current' ? 'outline' : 'default'}
              onClick={() => void checkTask.run(actions.CheckUpdate)}
              disabled={
                checkTask.pending ||
                (availability ? !availability.check.allowed : false)
              }
            >
              {checkTask.pending
                ? 'Checking…'
                : state === 'failed'
                  ? 'Try again'
                  : 'Check for updates'}
            </Button>
          )}
          {state === 'available' && (
            <ActionButton
              variant="default"
              action={actions.DownloadUpdate}
              busyLabel="Downloading…"
              onPendingChange={(download) =>
                setBusy((previous) => ({ ...previous, download }))
              }
            >
              Download update
            </ActionButton>
          )}
          {state === 'progress' && (
            <ActionButton action={actions.CancelUpdate}>
              Cancel download
            </ActionButton>
          )}
          {state === 'ready' && (
            <ActionButton
              variant="default"
              action={actions.InstallUpdate}
              busyLabel="Installing…"
              onPendingChange={(install) =>
                setBusy((previous) => ({ ...previous, install }))
              }
              disabled={
                availability ? !availability.install.allowed : undefined
              }
            >
              Install and restart
            </ActionButton>
          )}
          {/* Later and Dismiss do the same thing: one of them shows. A failed
              update's saved result is dismissed, an offer is put off. */}
          {dismiss === 'Later' && (
            <ActionButton
              variant="outline"
              action={actions.DismissUpdate}
              disabled={busy.download || busy.install}
            >
              Later
            </ActionButton>
          )}
          {dismiss === 'Dismiss' && (
            <ActionButton variant="outline" action={actions.DismissUpdate}>
              Dismiss
            </ActionButton>
          )}
        </div>
        {checkButton && (
          <p className="text-sm text-muted-foreground">
            Checking for updates contacts Bubo's GitHub release feed. The
            request sends your installed version in its User-Agent and no other
            app data. You can turn automatic checks off in Settings.
          </p>
        )}
        {availability &&
          !availability.check.allowed &&
          !checkFailed &&
          ['none', 'current', 'failed', 'not_checked'].includes(state) && (
            <GUIError
              error={{
                code: availability.check.reason_code,
                remedy: availability.check.reason,
              }}
              recovery={false}
            />
          )}
        {update.notes && state !== 'rollback_pending' && (
          <details className="border-t pt-3">
            <summary className="min-h-9 cursor-pointer">
              {update.available_version
                ? `Update to ${update.available_version}`
                : 'Update details'}
            </summary>
            <p className="pt-3 whitespace-pre-wrap">{update.notes}</p>
          </details>
        )}
      </CardContent>
    </Card>
  )
}
