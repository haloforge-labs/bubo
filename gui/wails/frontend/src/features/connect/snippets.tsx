import { useState } from 'react'
import { ActionButton } from '@/components/action-feedback'
import { CopyText } from '@/components/copy-text'
import { Label } from '@/components/ui/label'
import { requireActionSuccess } from '@/lib/bindings'
import { actions, type View } from '@/lib/bindings'

export function fillSnippet(
  view: View,
  template: string,
  address = '',
  key?: string,
  // The CLI's loopback dummy key, for a snippet that needs no key.
  defaultKey?: string | null,
) {
  let result = template
  for (const [name, value] of Object.entries(view.endpoints ?? {})) {
    if (!result.includes(`{${name}}`)) continue
    if (!value)
      throw {
        code: 'internal_error',
        remedy: 'This client address has not been reported.',
      }
    const endpoint = new URL(value)
    if (address) {
      // Reuse the CLI endpoint's path with the reachable origin selected by the user.
      const reachable = new URL(address)
      endpoint.protocol = reachable.protocol
      endpoint.hostname = reachable.hostname
      endpoint.port = reachable.port
    }
    result = result.replaceAll(
      `{${name}}`,
      address ? endpoint.href.replace(/\/$/, '') : value,
    )
  }
  if (/\{(?:openai|anthropic)_base_url\}/.test(result))
    throw {
      code: 'internal_error',
      remedy: 'This client address has not been reported.',
    }
  // An explicit keyed copy fills the real key; a keyless server's default key
  // fills the template as shown and as copied; otherwise the placeholder stays.
  if (key !== undefined) return result.replaceAll('{api_key}', key)
  return defaultKey ? result.replaceAll('{api_key}', defaultKey) : result
}

function Snippet({
  view,
  template,
  title,
  address,
  defaultKey,
}: {
  view: View
  template: string
  title: string
  address: string
  defaultKey?: string | null
}) {
  let text: string | undefined
  try {
    text = fillSnippet(view, template, address, undefined, defaultKey)
  } catch {
    /* Missing endpoints stay visibly unavailable. */
  }
  return (
    <section aria-label={title} className="space-y-3">
      <h3 className="text-xl">{title}</h3>
      {text === undefined ? (
        <p data-code="not_reported">
          {view.serving
            ? 'This client address has not been reported.'
            : 'Start the server to see the client address.'}
        </p>
      ) : (
        <>
          <pre className="max-h-80 overflow-auto border bg-secondary p-4 text-xs whitespace-pre-wrap break-all">
            {text}
          </pre>
          <div className="flex gap-3">
            <CopyText label="Copy" text={text} />
            {view.key.lan_access &&
              view.key.present &&
              view.key.copy_with_key && (
                <ActionButton
                  action={async () => {
                    const { value: key } = requireActionSuccess(
                      await actions.RevealKey(),
                    )
                    const keyed = fillSnippet(view, template, address, key)
                    await actions.CopyText(keyed)
                  }}
                  done="Copied snippet with key."
                >
                  Copy address and key
                </ActionButton>
              )}
          </div>
        </>
      )}
    </section>
  )
}

export function ClientSnippets({ view }: { view: View }) {
  const [selected, setSelected] = useState('')
  const addresses = view.key.lan_access ? (view.key.addresses ?? []) : []
  const address = addresses.some((entry) => entry.url === selected)
    ? selected
    : ''
  if (!view.client_snippets?.length) return null
  return (
    <section aria-label="Client templates" className="space-y-6">
      <h2 className="text-2xl">Client templates</h2>
      {!!addresses.length && (
        <div className="space-y-2">
          <Label htmlFor="client-address">Address for client templates</Label>
          <select
            id="client-address"
            className="h-9 w-full border bg-background px-2 text-sm"
            value={address}
            onChange={(event) => setSelected(event.target.value)}
          >
            <option value="">Reported server endpoints</option>
            {addresses.map((entry) => (
              <option key={entry.url} value={entry.url}>
                {entry.adapter} · {entry.url}
              </option>
            ))}
          </select>
        </div>
      )}
      {view.client_snippets.map((snippet) => (
        <Snippet
          key={snippet.kind}
          view={view}
          address={address}
          title={snippet.title}
          template={snippet.template}
          defaultKey={
            snippet.requires_key ? undefined : snippet.default_api_key
          }
        />
      ))}
    </section>
  )
}
