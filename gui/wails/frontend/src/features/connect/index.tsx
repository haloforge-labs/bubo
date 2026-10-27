import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { CopyText } from '@/components/copy-text'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Switch } from '@/components/ui/switch'
import { actions, useView } from '@/lib/bindings'
import { KeyControls } from './key-controls'
import { ClientSnippets } from './snippets'
import { Button } from '@/components/ui/button'
import { serverAddresses } from '@/lib/format'
import { GUIError } from '@/components/gui-error'
import { Link } from '@tanstack/react-router'

export default function ConnectPage() {
  const query = useView()
  const lan = useActionTask()
  const restart = useActionTask()
  const view = query.data
  const addresses = view ? serverAddresses(view) : undefined
  return (
    <Page title="Connect">
      {query.error && (
        <ErrorNote>
          <span>{query.error.message}</span>
        </ErrorNote>
      )}
      {view ? (
        <div className="space-y-6">
          <section aria-label="Server addresses" className="space-y-3">
            {!view.serving &&
              view.phase !== 'starting' &&
              !addresses?.openai_base_url &&
              !addresses?.anthropic_base_url && (
                <div className="space-y-2">
                  <p>
                    No server is running, so there is nothing to connect to yet.
                  </p>
                  <p>
                    Start the server on the{' '}
                    <Link to="/" className="underline">
                      Server page
                    </Link>
                    .
                  </p>
                </div>
              )}
            {(view.phase === 'starting' || view.serving) &&
              !addresses?.openai_base_url &&
              !addresses?.anthropic_base_url && (
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <p>
                    {view.phase === 'starting'
                      ? 'The server is starting. Addresses appear when it is ready.'
                      : 'Waiting for the server address.'}
                  </p>
                  {view.serving && (
                    <Button
                      variant="outline"
                      disabled={query.isFetching}
                      onClick={() => void query.refetch()}
                    >
                      Refresh addresses
                    </Button>
                  )}
                </div>
              )}
            {view.serving &&
              (
                [
                  ['OpenAI', addresses?.openai_base_url],
                  ['Anthropic', addresses?.anthropic_base_url],
                ] as const
              ).map(([label, url]) =>
                url ? (
                  <div
                    key={label}
                    className="grid min-w-0 grid-cols-[6rem_minmax(0,1fr)_auto] items-start gap-4 border-b pb-3"
                  >
                    <span>{label}</span>
                    <code className="break-all text-sm">{url}</code>
                    <CopyText
                      iconOnly
                      label={`Copy ${label} address`}
                      text={url}
                    />
                  </div>
                ) : null,
              )}
          </section>
          <section aria-label="Local network access" className="space-y-3">
            <div className="flex items-center gap-4">
              <h2 id="lan-label" className="text-2xl">
                Local network access
              </h2>
              <Switch
                id="lan-access"
                aria-labelledby="lan-label"
                checked={view.key.lan_access}
                disabled={lan.pending}
                onCheckedChange={(lan_access) =>
                  void lan.run(() =>
                    actions.PatchPreferences({
                      set: { lan_access },
                      clear: [],
                    }),
                  )
                }
              />
              <span>{view.key.lan_access ? 'On' : 'Off'}</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {view.key.lan_access
                ? 'Every client must send the key, including clients on this computer.'
                : 'Only this computer can connect. Turn on to allow other devices; every client will need a key.'}
            </p>
            <ActionError error={lan.error} />
            {view.key.unavailable && (
              // The key store failed: the one row sentence, the raw text under
              // Details. Retrying is turning the switch again, so no button.
              <GUIError
                error={{
                  code: 'key_store_failed',
                  remedy: view.key.unavailable,
                }}
                recovery={false}
              />
            )}
            {(view.key.restart_needed ||
              view.key.lan_access !== view.key.lan_active) && (
              <div className="space-y-2">
                <p role="status">
                  Network or key changes are used after restart.
                </p>
                <Button
                  variant="outline"
                  disabled={restart.pending || !view.actions.restart.allowed}
                  onClick={() => void restart.run(actions.Restart)}
                >
                  Restart server
                </Button>
                {!view.actions.restart.allowed && (
                  <GUIError
                    error={{
                      code: view.actions.restart.reason_code,
                      remedy: view.actions.restart.reason,
                    }}
                    recovery={false}
                  />
                )}
                <ActionError error={restart.error} />
              </div>
            )}
            {view.key.lan_access && (
              <>
                {(view.key.addresses ?? []).map((address) => (
                  <div
                    key={address.url}
                    className="flex flex-wrap items-start justify-between gap-3"
                  >
                    <div>
                      <p>{address.adapter}</p>
                      <code className="break-all text-sm">{address.url}</code>
                    </div>
                    <CopyText
                      iconOnly
                      label="Copy network address"
                      text={address.url}
                    />
                  </div>
                ))}
                {view.key.addresses_unavailable && (
                  <p>{view.key.addresses_unavailable}</p>
                )}
                {view.key.copy_with_key && view.key.present && (
                  <KeyControls
                    key={`${view.generation}-${view.key.restart_needed}`}
                    view={view}
                  />
                )}
              </>
            )}
          </section>
          {view.serving && <ClientSnippets view={view} />}
          {view.serving && (
            <section aria-label="Client snippet" className="space-y-4">
              <h2 className="text-2xl">Pi (coding agent)</h2>
              <p className="text-sm text-muted-foreground">
                Copy this configuration into your client.
              </p>
              {view.pi_snippet ? (
                <>
                  <CopyText label="Copy snippet" text={view.pi_snippet} />
                  <pre className="max-h-80 overflow-auto border bg-secondary p-4 text-xs whitespace-pre-wrap break-all">
                    {view.pi_snippet}
                  </pre>
                </>
              ) : (
                <p>Client snippet not reported.</p>
              )}
            </section>
          )}
        </div>
      ) : (
        !query.error && <p>Loading connection details…</p>
      )}
    </Page>
  )
}
