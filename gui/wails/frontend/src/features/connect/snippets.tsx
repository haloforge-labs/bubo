import { useState, type ReactNode } from 'react'
import { ActionButton } from '@/components/action-feedback'
import { CopyText } from '@/components/copy-text'
import { Label } from '@/components/ui/label'
import { Disclosure } from '@/components/disclosure'
import { pageGrid } from '@/components/page-grid'
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

// One example: a collapsed Disclosure named for the client, with its Copy
// actions on the same line. Every example, including the Pi configuration, is
// built from this. An opened example takes the whole row to read the code.
function Example({
  title,
  text,
  missing,
  children,
}: {
  title: string
  text: string | undefined
  missing: string
  children?: ReactNode
}) {
  return (
    <section
      aria-label={title}
      className="min-w-0 has-[details[open]]:col-span-full"
    >
      <div className="flex flex-wrap items-start gap-x-3 gap-y-1">
        <Disclosure
          className="min-w-0 flex-1 basis-40"
          summaryClassName="h-9"
          summary={title}
        >
          {text === undefined ? (
            <p data-code="not_reported" className="pb-2 text-sm">
              {missing}
            </p>
          ) : (
            <pre className="mb-2 max-h-80 overflow-auto border bg-secondary p-3 text-xs whitespace-pre-wrap break-all">
              {text}
            </pre>
          )}
        </Disclosure>
        {text !== undefined && (
          <div className="flex flex-wrap gap-2">
            <CopyText compact label="Copy" text={text} />
            {children}
          </div>
        )}
      </div>
    </section>
  )
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
    <Example
      title={title}
      text={text}
      missing={
        view.serving
          ? 'This client address has not been reported.'
          : 'Start the server to see the client address.'
      }
    >
      {view.key.lan_access && view.key.present && view.key.copy_with_key && (
        <ActionButton
          size="sm"
          action={async () => {
            const { value: key } = requireActionSuccess(
              await actions.RevealKey(),
            )
            const keyed = fillSnippet(view, template, address, key)
            await actions.CopyText(keyed)
          }}
          done="Copied snippet with key."
        >
          Copy with key
        </ActionButton>
      )}
    </Example>
  )
}

// The address the examples use. The default is what the server reports; it is
// "This computer" only when that address really is loopback.
function reportsLoopback(view: View) {
  const urls = Object.values(view.endpoints ?? {}).filter(Boolean)
  return (
    urls.length > 0 &&
    urls.every((url) => {
      try {
        const host = new URL(url).hostname
        return (
          /^127(?:\.\d+){3}$/.test(host) ||
          ['localhost', '[::1]'].includes(host)
        )
      } catch {
        return false
      }
    })
  )
}

export function ClientSnippets({ view }: { view: View }) {
  const [selected, setSelected] = useState('')
  const addresses = view.key.lan_access ? (view.key.addresses ?? []) : []
  const address = addresses.some((entry) => entry.url === selected)
    ? selected
    : ''
  return (
    <section aria-label="Client configuration" className="space-y-2">
      <h2 className="text-lg">Client configuration</h2>
      {!!addresses.length && (
        <div className="flex items-center gap-3">
          <Label htmlFor="client-address">Client address</Label>
          <select
            id="client-address"
            className="h-9 min-w-0 flex-1 border bg-background px-2 text-sm"
            value={address}
            onChange={(event) => setSelected(event.target.value)}
          >
            <option value="">
              {reportsLoopback(view) ? 'This computer' : 'Server address'}
            </option>
            {addresses.map((entry) => (
              <option key={entry.url} value={entry.url}>
                {entry.adapter} · {entry.url}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className={pageGrid}>
        {(view.client_snippets ?? []).map((snippet) => (
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
        <Example
          title="Pi (coding agent)"
          text={view.pi_snippet || undefined}
          missing="Client configuration not reported."
        />
      </div>
    </section>
  )
}
