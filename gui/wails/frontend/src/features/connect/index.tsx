import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { AddressRow } from '@/components/address-row'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { Switch } from '@/components/ui/switch'
import { actions, useView, type View } from '@/lib/bindings'
import { KeyControls } from './key-controls'
import { ClientSnippets } from './snippets'
import { Button } from '@/components/ui/button'
import { serverAddresses } from '@/lib/format'
import { GUIError } from '@/components/gui-error'
import { Link } from '@tanstack/react-router'
import { pageGrid } from '@/components/page-grid'

// Saved and active differ until the restart: say what is saved as pending, and
// describe current access from what is active.
function lanStatus(key: View['key']) {
  if (key.lan_access === key.lan_active) return key.lan_access ? 'On' : 'Off'
  return key.lan_access ? 'On after restart' : 'Off after restart'
}

function lanSentence(key: View['key']) {
  if (key.lan_active)
    return key.lan_access
      ? 'Every client must send the key, including clients on this computer.'
      : 'Other devices can connect now.'
  return key.lan_access
    ? 'Only this computer can connect now.'
    : 'Only this computer can connect. Turn on to allow other devices; every client will need a key.'
}

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
        <div className="space-y-4">
          <section aria-label="Server addresses" className="space-y-2">
            {!view.serving &&
              view.phase !== 'starting' &&
              !addresses?.openai_base_url &&
              !addresses?.anthropic_base_url && (
                <p>
                  <Link to="/" className="underline">
                    Start the server to connect.
                  </Link>
                </p>
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
            {view.serving && (
              <div className={pageGrid}>
                {(
                  [
                    ['OpenAI', addresses?.openai_base_url],
                    ['Anthropic', addresses?.anthropic_base_url],
                  ] as const
                ).map(([label, url]) =>
                  url ? (
                    <AddressRow
                      key={label}
                      label={label}
                      url={url}
                      copyLabel={`Copy ${label} address`}
                    />
                  ) : null,
                )}
              </div>
            )}
          </section>
          <section aria-label="Local network access" className="space-y-2">
            <div className="flex items-center gap-3">
              <h2 id="lan-label" className="text-lg">
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
              <span>{lanStatus(view.key)}</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {lanSentence(view.key)}
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
                <div className={pageGrid}>
                  {(view.key.addresses ?? []).map((address) => (
                    <AddressRow
                      key={address.url}
                      label={address.adapter}
                      url={address.url}
                      copyLabel="Copy network address"
                    />
                  ))}
                </div>
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
        </div>
      ) : (
        !query.error && <p>Loading connection details…</p>
      )}
    </Page>
  )
}
