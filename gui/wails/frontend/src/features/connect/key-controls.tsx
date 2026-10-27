import { useEffect, useRef, useState } from 'react'
import { z } from 'zod'
import {
  ActionButton,
  ActionError,
  useActionTask,
} from '@/components/action-feedback'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from '@/components/ui/dialog'
import { requireActionSuccess } from '@/lib/bindings'
import { actions, type View } from '@/lib/bindings'

// Transform only the CLI's supplied JSON snippet. Never build an endpoint or
// another client configuration in the frontend, or retain the secret in View.
export function snippetWithKey(snippet: string, key: string): string {
  let json: unknown
  try {
    json = JSON.parse(snippet) as unknown
  } catch {
    throw {
      code: 'internal_error',
      remedy: 'The client snippet is invalid JSON.',
    }
  }
  const parsed = z.json().safeParse(json)
  if (!parsed.success)
    throw {
      code: 'internal_error',
      remedy: 'The client snippet cannot include a key.',
    }
  let replaced = false
  function replace(
    value: z.infer<ReturnType<typeof z.json>>,
  ): z.infer<ReturnType<typeof z.json>> {
    if (Array.isArray(value)) return value.map(replace)
    if (value && typeof value === 'object')
      return Object.fromEntries(
        Object.entries(value).map(([name, item]) => {
          if (name === 'apiKey' && typeof item === 'string') {
            replaced = true
            return [name, key]
          }
          return [name, replace(item)]
        }),
      )
    return value
  }
  const result = replace(parsed.data)
  if (!replaced)
    throw {
      code: 'internal_error',
      remedy: 'The client snippet has no key placeholder.',
    }
  return JSON.stringify(result, null, 2)
}

export function KeyControls({ view }: { view: View }) {
  const [shown, setShown] = useState('')
  const [seconds, setSeconds] = useState(30)
  const [keep, setKeep] = useState(false)
  const [resetOpen, setResetOpen] = useState(false)
  const lifetime = useRef(0)
  const mounted = useRef(false)
  const show = useActionTask()
  const reset = useActionTask()
  function hide() {
    lifetime.current++
    setShown('')
    setKeep(false)
  }
  useEffect(() => {
    mounted.current = true
    function onVisibility() {
      if (document.hidden) {
        lifetime.current++
        setShown('')
        setKeep(false)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      mounted.current = false
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])
  useEffect(() => {
    if (!shown || keep) return
    const timeout = setTimeout(() => {
      lifetime.current++
      setShown('')
    }, 30_000)
    const interval = setInterval(
      () => setSeconds((value) => Math.max(0, value - 1)),
      1_000,
    )
    return () => {
      clearTimeout(timeout)
      clearInterval(interval)
    }
  }, [shown, keep])
  return (
    <section className="space-y-3" aria-label="API key">
      <h3 className="text-xl">API key</h3>
      <p className="font-mono break-all text-sm" data-testid="key-value">
        {shown || view.key.mask || '••••'}
      </p>
      <div className="flex flex-wrap gap-3">
        {shown ? (
          <>
            <Button variant="outline" onClick={hide}>
              Hide{keep ? '' : ` · ${seconds} s`}
            </Button>
            {!keep && (
              <Button variant="outline" onClick={() => setKeep(true)}>
                Keep showing
              </Button>
            )}
          </>
        ) : (
          <Button
            variant="outline"
            disabled={show.pending}
            onClick={() => {
              const token = ++lifetime.current
              void show.run(async () => {
                const { value: key } = requireActionSuccess(
                  await actions.RevealKey(),
                )
                if (
                  mounted.current &&
                  token === lifetime.current &&
                  !document.hidden
                ) {
                  setSeconds(30)
                  setKeep(false)
                  setShown(key)
                }
              })
            }}
          >
            Show
          </Button>
        )}
        <ActionButton
          disabled={!view.pi_snippet}
          action={async () => {
            const { value: key } = requireActionSuccess(
              await actions.RevealKey(),
            )
            const snippet = snippetWithKey(view.pi_snippet, key)
            await actions.CopyText(snippet)
          }}
          done="Copied snippet with key."
        >
          Copy address and key
        </ActionButton>
        <Button
          variant="outline"
          onClick={() => {
            hide()
            setResetOpen(true)
          }}
        >
          Reset key…
        </Button>
      </div>
      <ActionError error={show.error} />
      {shown && (
        <p className="text-sm text-muted-foreground">
          {keep
            ? 'Shown until you hide the key or leave Connect.'
            : 'Hidden after 30 seconds or when you leave Connect.'}
        </p>
      )}
      <Dialog open={resetOpen} onOpenChange={setResetOpen}>
        <DialogContent>
          <DialogTitle>Reset the key?</DialogTitle>
          <DialogDescription>
            Clients using the old key keep working until the restart, then stop.
            Give them the new key before you restart.
          </DialogDescription>
          <ActionError error={reset.error} />
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" autoFocus>
                Cancel
              </Button>
            </DialogClose>
            <Button
              disabled={reset.pending}
              onClick={() =>
                void reset.run(actions.ResetKey).then((ok) => {
                  if (ok) setResetOpen(false)
                })
              }
            >
              Reset key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}
