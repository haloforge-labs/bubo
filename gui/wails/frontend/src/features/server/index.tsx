import { useEffect, useRef, useState } from 'react'
import { Page } from '@/components/page'
import { DesktopView } from '@/components/desktop-view'
import { Preparation } from '@/components/preparation'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { actions, type View } from '@/lib/bindings'
import { ServerSetup, type ServerSetupHandle } from './components/server-setup'
import { Lifecycle } from '@/components/server-lifecycle'
import { RejectedOverride } from '@/components/rejected-override'
import { GUIError } from '@/components/gui-error'
import { guiError, warningSentence } from '@/lib/gui-errors'
import { CopyText } from '@/components/copy-text'
import { PathText } from '@/components/path-text'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { effectiveValue } from '@/lib/registry-view'
import { liveSessionSummary } from '@/lib/live-sessions'
import { modelAlias } from '@/lib/server-values'
import {
  serverAddresses,
  activeModelName,
  number,
  rate,
  duration,
  folderPath,
} from '@/lib/format'

export default function ServerPage() {
  return (
    <Page title="Server">
      <DesktopView>{(view) => <ServerContent view={view} />}</DesktopView>
    </Page>
  )
}

function BusyPort({ view }: { view: View }) {
  const [port, setPort] = useState(() =>
    String(effectiveValue(view, '--port') ?? ''),
  )
  const command = useActionTask()
  const validPort =
    /^\d+$/.test(port) && Number(port) >= 1 && Number(port) <= 65535
  const row = view.switches.find((row) => row.name === '--port')
  if (!row) return null
  return (
    <div className="flex flex-wrap items-end gap-2">
      <label className="space-y-1 text-sm">
        New server port
        <Input
          value={port}
          onChange={(event) => setPort(event.target.value)}
          inputMode="numeric"
          aria-invalid={!validPort}
          aria-describedby={!validPort ? 'busy-port-error' : undefined}
        />
      </label>
      <Button
        variant="outline"
        disabled={command.isPending || !validPort}
        onClick={() =>
          validPort &&
          command.mutate(() =>
            actions.PatchPreferences({
              set: {},
              clear: [],
              switch_values: [{ name: row.name, value: port }],
            }),
          )
        }
      >
        Save port
      </Button>
      {!validPort && (
        <p id="busy-port-error" className="text-sm">
          Enter a whole-number port from 1 to 65535.
        </p>
      )}
      <ActionError error={command.error} />
    </div>
  )
}

function ServerContent({ view }: { view: View }) {
  const [setupDirty, setSetupDirty] = useState(false)
  const setupAction = useRef<ServerSetupHandle>(null)
  const [suggestionSignal, setSuggestionSignal] = useState(0)
  const [startupRequest, setStartupRequest] = useState<{
    generation: number
  }>()
  useEffect(() => {
    if (!startupRequest) return
    // Cover the next View poll after Start/Restart. Once the supervisor starts
    // the new generation, its starting phase carries the waiting indication.
    const timer = setTimeout(() => setStartupRequest(undefined), 5000)
    return () => clearTimeout(timer)
  }, [startupRequest])
  const waiting =
    view.phase === 'starting' ||
    (startupRequest != null && startupRequest.generation === view.generation)
  const running = view.serving
  const addresses = serverAddresses(view)
  const modelName = activeModelName(view)
  const identity = view.current_model
  const modelPath = identity?.model_path || identity?.manifest_path
  const servedName = view.instance?.served_model || view.connection?.model_id
  const needsPreparation =
    view.actions.prepare.allowed ||
    view.actions.download_default.allowed ||
    view.phase === 'preparing_engine' ||
    view.phase === 'downloading_model'
  const request = view.requests?.findLast((row) => row.outcome === 'completed')
  const metrics = [
    {
      key: 'decode',
      label: 'Decode tok/s',
      value: request?.decode_tps,
    },
    {
      key: 'prefill',
      label: 'Prefill tok/s',
      value: request?.prefill_tps,
    },
  ] as const

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-2"
          aria-label="Server status"
        >
          <p className="text-lg" role="status">
            {view.status_label}
            {waiting
              ? ' · Please wait'
              : running && view.stale
                ? ' · Not updating'
                : ''}
          </p>
          {running && (
            <p
              aria-label="Live server activity"
              className="text-sm text-muted-foreground"
            >
              {liveSessionSummary(view)}
            </p>
          )}
          <Lifecycle
            view={view}
            serverPage
            onStartRequested={() =>
              setStartupRequest({
                generation: view.generation,
              })
            }
            onStartFailed={() => setStartupRequest(undefined)}
            onUseSuggestedContext={() =>
              setSuggestionSignal((value) => value + 1)
            }
            hasUnsavedSetup={setupDirty}
            beforeStart={async () => {
              if (setupDirty) {
                if (!setupAction.current)
                  throw {
                    code: 'setting_rejected',
                    remedy: 'Wait for server setup to be ready.',
                  }
                await setupAction.current.save()
              }
            }}
          />
        </div>
        {view.auto_restarted && (
          <p role="status" className="text-sm text-muted-foreground">
            {guiError({ code: 'device_lost_restarted', remedy: '' }).message}
          </p>
        )}
        {view.phase === 'stalled' && (
          <GUIError
            error={{
              code: 'server_stalled',
              remedy: view.stall?.message ?? '',
            }}
          />
        )}
        {/* A failure shows its cause, its own button and Details. Only a state
            that is not a failure (setup needed) is left to the status row. */}
        {!view.failure?.switch &&
          (view.failure ||
            ![
              view.actions.start.reason_code,
              view.actions.restart.reason_code,
            ].includes(view.reason)) && (
            <GUIError
              retry={
                view.failure?.code === 'insufficient_space' ||
                view.failure?.code === 'download_stalled'
                  ? actions.Prepare
                  : undefined
              }
              error={
                view.failure ||
                (view.reason && view.remedy
                  ? { code: view.reason, remedy: view.remedy }
                  : undefined)
              }
            />
          )}
        {view.server_warnings.map((warning) => {
          // A known code gets one plain sentence; an unknown code shows no line.
          const sentence = warningSentence(warning.code)
          return (
            sentence && (
              <div key={warning.code} className="text-sm text-muted-foreground">
                <p>{sentence}</p>
                <details className="text-xs">
                  <summary className="cursor-pointer">Details</summary>
                  <p className="font-mono">
                    {warning.code}: {warning.remedy}
                  </p>
                </details>
              </div>
            )
          )
        })}
        <RejectedOverride failure={view.failure} />
        {(view.failure?.code === 'port_occupied' ||
          view.reason === 'port_occupied') && <BusyPort view={view} />}
      </div>
      {/* The setup form says nothing while the CLI cannot run, as on cli-missing. */}
      {view.failure?.code !== 'vcredist_missing' && (
        <section aria-label="Server setup" className="space-y-3 border-t pt-4">
          <h2 className="text-xl">Server setup</h2>
          <ServerSetup
            view={view}
            onDirtyChange={setSetupDirty}
            actionRef={setupAction}
            suggestionSignal={suggestionSignal}
            addresses={
              <>
                {' '}
                {(addresses.openai_base_url ||
                  addresses.anthropic_base_url) && (
                  <section aria-label="Addresses" className="space-y-2">
                    <h2 className="text-lg">Addresses</h2>
                    {(
                      [
                        ['OpenAI', addresses.openai_base_url],
                        ['Anthropic', addresses.anthropic_base_url],
                      ] as const
                    ).map(
                      ([label, url]) =>
                        url && (
                          <div
                            key={label}
                            className="flex min-w-0 flex-wrap items-center gap-3 text-sm"
                          >
                            <span>{label}</span>
                            <code className="min-w-0 break-all">{url}</code>
                            <CopyText
                              iconOnly
                              label={`Copy ${label} address`}
                              text={url}
                            />
                          </div>
                        ),
                    )}
                  </section>
                )}
              </>
            }
          />
        </section>
      )}
      {running && (
        <section aria-label="Model in use" className="space-y-1">
          <p>
            Model: {modelName || 'not identified'}
            {!modelName && servedName ? ` (server name: ${servedName})` : ''}
          </p>
          {modelPath && (
            <details className="text-sm text-muted-foreground">
              <summary>Model folder</summary>
              <PathText path={folderPath(modelPath)} />
            </details>
          )}
          {servedName && (
            <p className="text-sm text-muted-foreground">
              Clients see this model as <code>{servedName}</code>.
              {servedName !== modelAlias && (
                <>
                  {' '}
                  (<code>{modelAlias}</code> also works)
                </>
              )}
            </p>
          )}
        </section>
      )}
      {running && !waiting && view.status_feed?.state === 'not_updating' && (
        <p className="text-sm text-muted-foreground">
          Live numbers are not updating.
          {view.status_feed.age_seconds != null ? (
            <>
              {' '}
              ·{' '}
              <span className="tabular-nums">
                {number(view.status_feed.age_seconds, 0)}
              </span>{' '}
              seconds old
            </>
          ) : (
            ''
          )}
        </p>
      )}
      {(view.approval ||
        (needsPreparation &&
          !view.failure &&
          ![
            'runtime_missing',
            'model_missing',
            'source_missing',
            'not_prepared',
          ].includes(view.actions.start.reason_code))) && (
        <Preparation view={view} primary />
      )}
      {running && (
        <>
          <section aria-label="Reported request rates" className="space-y-2">
            <p className="text-sm text-muted-foreground">
              {request ? (
                <>
                  Last request ·{' '}
                  <span className="tabular-nums">
                    {new Date(request.time).toLocaleTimeString(undefined, {
                      hour12: false,
                    })}
                  </span>
                  {view.last_completed_request?.age_ms != null
                    ? ` (${duration(view.last_completed_request.age_ms / 1000)} ago)`
                    : ''}
                </>
              ) : (
                'No requests yet.'
              )}
            </p>
            {request && (
              <div className="grid gap-4 sm:grid-cols-3">
                {metrics.map((metric) => (
                  <figure key={metric.key} aria-label={metric.label}>
                    {metric.key === 'prefill' &&
                    request.prompt_tokens != null &&
                    request.prompt_tokens > 0 &&
                    request.cached_tokens === request.prompt_tokens ? (
                      <p className="text-lg tabular-nums">
                        Prefill: all {number(request.prompt_tokens, 0)} tokens
                        from cache
                      </p>
                    ) : (
                      <>
                        <p className="text-3xl tabular-nums">
                          {metric.value == null ? '—' : rate(metric.value)}{' '}
                          <span className="text-sm">tok/s</span>
                        </p>
                        <figcaption className="text-sm text-muted-foreground">
                          {metric.key === 'decode' ? 'Decode' : 'Prefill'}
                        </figcaption>
                        {metric.key === 'prefill' &&
                          request.prompt_tokens != null &&
                          request.cached_tokens != null &&
                          request.cached_tokens > 0 &&
                          request.cached_tokens < request.prompt_tokens && (
                            <p className="text-sm text-muted-foreground">
                              {number(
                                request.prompt_tokens - request.cached_tokens,
                                0,
                              )}{' '}
                              new + {number(request.cached_tokens, 0)} cached
                            </p>
                          )}
                      </>
                    )}
                  </figure>
                ))}
                {request.ttft_ms != null && (
                  <figure aria-label="Time to first token">
                    <p className="text-3xl tabular-nums">
                      {request.ttft_ms < 1000
                        ? `${number(request.ttft_ms, 0)} ms`
                        : duration(request.ttft_ms / 1000)}
                    </p>
                    <figcaption className="text-sm text-muted-foreground">
                      Time to first token
                    </figcaption>
                  </figure>
                )}
              </div>
            )}
          </section>
        </>
      )}
      {running && (
        <section aria-label="Sessions" className="space-y-2">
          <h2 className="text-lg">
            Sessions
            {view.sessions != null ? (
              <>
                {' '}
                ·{' '}
                <span className="tabular-nums">
                  {number(view.sessions, 0)}
                </span>{' '}
                active
              </>
            ) : (
              ''
            )}
            {view.queue != null ? (
              <>
                {' '}
                · <span className="tabular-nums">
                  {number(view.queue, 0)}
                </span>{' '}
                queued
              </>
            ) : (
              ''
            )}
          </h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-right">Session</TableHead>
                <TableHead>State</TableHead>
                <TableHead className="text-right">Context used</TableHead>
                <TableHead className="text-right">Generated (tokens)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {view.slots?.map((slot) => (
                <TableRow key={slot.id}>
                  <TableCell className="text-right tabular-nums">
                    {number(slot.id + 1, 0)}
                  </TableCell>
                  <TableCell>{slot.state_label}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {slot.tokens_used?.toLocaleString() ?? '—'}
                    {slot.context_tokens != null
                      ? ` / ${slot.context_tokens.toLocaleString()}`
                      : ''}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {slot.generated_tokens?.toLocaleString() ?? '—'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {!view.slots?.length && (
            <p className="text-sm text-muted-foreground">
              No sessions reported.
            </p>
          )}
        </section>
      )}
    </div>
  )
}
