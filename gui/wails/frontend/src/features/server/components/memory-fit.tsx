import { Link } from '@tanstack/react-router'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import { number } from '@/lib/format'
import { Button } from '@/components/ui/button'

export function MemoryFit({
  view,
  fit,
  estimate,
  dirty,
  onContext,
  onSessions,
  modelId,
}: {
  view: View
  fit?: View['fit']
  estimate?: View['memory_estimate']
  dirty: boolean
  onContext: (value: number) => void
  onSessions: (value: number) => void
  modelId?: string
}) {
  const override = useActionTask()
  if (!fit || ['unavailable', 'in_use'].includes(fit.class)) return null
  const tight = fit.class === 'memory_tight'
  const noFit = ['memory_does_not_fit', 'memory_does_not_fit_any'].includes(
    fit.class,
  )
  const anyContext = fit.class === 'memory_does_not_fit_any'
  if (!tight && !noFit && fit.class !== 'fits')
    return (
      <p aria-label="Memory fit" className="text-sm">
        {fit.sentence}
      </p>
    )
  const fewer = estimate?.recommended_context?.by_sessions?.find(
    (row) =>
      row.sessions < fit.sessions &&
      row.recommended === true &&
      row.context_per_session >= fit.context_tokens,
  )
  const catalogue = view.models?.catalogue ?? []
  const model = catalogue.find((row) => row.id === modelId)
  const lowerMemory =
    model &&
    catalogue.some(
      (row) =>
        row.id !== model.id &&
        row.architecture === model.architecture &&
        /lower.memory/i.test(row.display_name),
    )
  const numbers = [
    ['GPU left after cache growth', estimate?.free_after_growth_gib],
    ['System memory left after spill', estimate?.host_free_after_spill_gib],
    ['Free system memory', estimate?.host?.available?.gib],
  ] as const
  return (
    <section aria-label="Memory fit" className="space-y-3 rounded border p-4">
      <h3 className="text-lg">
        {tight
          ? 'Tight'
          : anyContext
            ? "Doesn't fit at any context"
            : noFit
              ? "Doesn't fit"
              : 'Fits'}
      </h3>
      <p className="text-sm">{fit.sentence}</p>
      {numbers.some(([, value]) => value != null) && (
        <table className="text-sm">
          <tbody>
            {numbers.map(([label, value]) =>
              value == null ? null : (
                <tr key={label}>
                  <th className="pr-6 text-left font-normal">{label}</th>
                  <td className="text-right tabular-nums">
                    {number(value, 0)} GiB
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      )}
      {tight && (
        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            className="mt-1 accent-primary"
            checked={!!view.saved.allow_tight_setup && !dirty}
            disabled={dirty || override.isPending}
            onChange={(event) => {
              const allowed = event.currentTarget.checked
              override.mutate(() =>
                actions.PatchPreferences({
                  set: { allow_tight_setup: allowed },
                  clear: [],
                }),
              )
            }}
          />
          Override: allow this tight setup. It may run out of memory, stop the
          server or freeze your computer.
        </label>
      )}
      {tight && dirty && (
        <p className="text-sm text-muted-foreground">
          Save this setup before allowing the override.
        </p>
      )}
      {tight && !dirty && (
        <Button
          type="button"
          variant="outline"
          disabled={override.isPending}
          onClick={() => override.mutate(actions.RefreshModels)}
        >
          Check again
        </Button>
      )}
      {noFit && (
        <ul className="list-inside list-disc space-y-2 text-sm">
          {!anyContext && fit.suggested_context != null && (
            <li>
              <Button
                type="button"
                variant="outline"
                onClick={() => onContext(fit.suggested_context!)}
              >
                Use {fit.suggested_context.toLocaleString()} tokens
              </Button>
            </li>
          )}
          {fewer && (
            <li>
              <Button
                type="button"
                variant="outline"
                onClick={() => onSessions(fewer.sessions)}
              >
                Use fewer sessions ({fewer.sessions})
              </Button>
            </li>
          )}
          {!anyContext &&
            estimate?.per_session_vision &&
            estimate.per_session_vision.bytes > 0 &&
            view.switches.some(
              (row) => row.name === '--no-vision' && row.gui_control,
            ) && <li>Turn off image input in Memory and caches.</li>}
          {catalogue.length > 0 && (
            <li>
              <Link to="/models" className="underline">
                {lowerMemory
                  ? 'The lower-memory version of this model'
                  : 'Choose a lower-memory model'}
              </Link>
            </li>
          )}
          <li>
            More GPU memory: Variable Graphics Memory in AMD Software or the
            BIOS.
          </li>
          {!anyContext && estimate?.host?.available && (
            <li>Close other apps.</li>
          )}
        </ul>
      )}
      <ActionError error={override.error} />
    </section>
  )
}
