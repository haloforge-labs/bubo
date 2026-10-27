import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { useEffect, useRef, useState } from 'react'
import { useForm } from 'react-hook-form'
import {
  ActionButton,
  ActionError,
  useActionTask,
} from '@/components/action-feedback'
import { CopyText } from '@/components/copy-text'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { actions, useView } from '@/lib/bindings'
import { copyLabel, logText, shownLines } from './lines'
import { GUIError } from '@/components/gui-error'
import { GPUHistory } from './gpu-history'

export default function LogsPage() {
  const query = useView()
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const log = useRef<HTMLPreElement>(null)
  const bundle = useForm({ defaultValues: { path: '' } })
  const save = useActionTask()
  const view = query.data
  const serverLogPath = view?.instance?.log_path || view?.log_path
  const hasServerLog = !!serverLogPath || !!view?.logs?.length
  const text = logText(view?.logs, filter)
  const lines = shownLines(view?.logs, filter)
  useEffect(() => {
    if (follow && log.current) log.current.scrollTop = log.current.scrollHeight
  }, [follow, text])
  return (
    <Page title="Logs">
      {query.error && (
        <ErrorNote>
          <span>{query.error.message}</span>
        </ErrorNote>
      )}
      <div className="space-y-5">
        <GUIError error={view?.failure} />
        {view && <GPUHistory view={view} updatedAt={query.dataUpdatedAt} />}
        {view && hasServerLog && (
          <div className="space-y-2 text-sm">
            {(view.phase === 'running_external' || serverLogPath) && (
              <p>
                {view.phase === 'running_external'
                  ? 'This server was started outside Bubo. '
                  : ''}
                {serverLogPath ? 'Server log path:' : ''}
              </p>
            )}
            {serverLogPath && (
              <div className="flex flex-wrap items-center gap-3">
                <code className="break-all">{serverLogPath}</code>
                <CopyText label="Copy log path" text={serverLogPath} />
              </div>
            )}
          </div>
        )}
        {view?.update.log_path && (
          <section aria-label="Updater log" className="space-y-2 text-sm">
            <h2 className="text-xl">Updater log</h2>
            <code className="break-all">{view.update.log_path}</code>
            <pre className="max-h-52 overflow-auto border bg-secondary p-4 text-xs whitespace-pre-wrap break-all">
              {view.update.log_lines.join('\n')}
            </pre>
          </section>
        )}
        {!hasServerLog && view && <p>Bubo has no server log to show.</p>}
        {hasServerLog && (
          <>
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-48 flex-1 space-y-2">
                <Label htmlFor="log-filter">Filter lines</Label>
                <Input
                  id="log-filter"
                  placeholder="Contains…"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                />
              </div>
              {filter && (
                <Button variant="outline" onClick={() => setFilter('')}>
                  Clear filter
                </Button>
              )}
              {(view?.logs?.length ?? 0) > 0 && (
                <CopyText label={copyLabel(filter)} text={lines.join('\n')} />
              )}
              <div className="flex min-h-9 items-center gap-2">
                <Label htmlFor="log-follow">Follow</Label>
                <Switch
                  id="log-follow"
                  checked={follow}
                  onCheckedChange={setFollow}
                />
              </div>
            </div>
            {(view?.logs?.length ?? 0) > 0 && (
              <p className="text-sm text-muted-foreground">
                <span className="tabular-nums">
                  {lines.length.toLocaleString()}
                </span>{' '}
                shown of{' '}
                <span className="tabular-nums">
                  {(view?.logs?.length ?? 0).toLocaleString()}
                </span>{' '}
                lines
              </p>
            )}
            <pre
              aria-label="Log lines"
              ref={log}
              tabIndex={0}
              className="h-52 overflow-auto border bg-secondary p-4 text-xs whitespace-pre-wrap break-all"
            >
              {text}
            </pre>
          </>
        )}
        <div className="flex flex-wrap gap-3">
          <ActionButton
            action={actions.Diagnose}
            disabled={!view?.actions.diagnose.allowed}
          >
            Run diagnostics
          </ActionButton>
          <ActionButton
            action={actions.PreviewBundle}
            disabled={!view?.actions.support_bundle.allowed}
          >
            Preview support bundle
          </ActionButton>
        </div>
        {view &&
          [
            ...new Map(
              [view.actions.diagnose, view.actions.support_bundle]
                .filter((action) => !action.allowed && action.reason_code)
                .map((action): [string, string] => [
                  action.reason_code,
                  action.reason,
                ]),
            ),
          ].map(([code, reason]) => (
            <GUIError
              key={code}
              error={{ code, remedy: reason }}
              recovery={false}
            />
          ))}
        {view?.diagnostic && (
          <pre className="overflow-auto border p-4 text-xs whitespace-pre-wrap">
            {view.diagnostic}
          </pre>
        )}
        {view?.bundle_preview && (
          <section className="space-y-4" aria-label="Support bundle">
            <h2 className="text-2xl">Support bundle</h2>
            <pre className="max-h-60 overflow-auto border p-4 text-xs whitespace-pre-wrap break-all">
              {view.bundle_preview}
            </pre>
            <form
              className="space-y-3"
              onSubmit={bundle.handleSubmit(({ path }) =>
                save.run(() => actions.SaveBundle(path)),
              )}
            >
              <Label htmlFor="bundle-path">Local ZIP path</Label>
              <Input
                id="bundle-path"
                placeholder="Absolute local path"
                {...bundle.register('path', {
                  required: 'Enter a local ZIP path.',
                })}
              />
              {bundle.formState.errors.path && (
                <ErrorNote>
                  <span>{bundle.formState.errors.path.message}</span>
                </ErrorNote>
              )}
              <Button
                type="submit"
                variant="outline"
                disabled={save.pending || !view.actions.support_bundle.allowed}
              >
                Save support bundle locally
              </Button>
              <ActionError error={save.error} />
              {save.completed && (
                <p role="status">Support bundle saved locally.</p>
              )}
            </form>
          </section>
        )}
      </div>
    </Page>
  )
}
