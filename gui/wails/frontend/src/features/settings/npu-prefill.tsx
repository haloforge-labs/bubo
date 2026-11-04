import { bytes, number, percent, rate } from '@/lib/format'
import type { View } from '@/lib/bindings'
import { guiError } from '@/lib/gui-errors'
import { Disclosure } from '@/components/disclosure'
import { valueWord } from '@/lib/presentation'

// A power window under a second is told in milliseconds, never rounded to zero.
function windowText(seconds: number) {
  return seconds < 1
    ? `${Math.round(seconds * 1000)} ms`
    : `${number(seconds, Number.isInteger(seconds) ? 0 : 2)} s`
}

export function NPUPrefill({
  runtime,
}: {
  runtime: View['npu_prefill_runtime']
}) {
  const paused = runtime?.state === 'tripped' || runtime?.state === 'faulted'
  // Only on and paused say anything. Off is the normal default and unavailable
  // has no reason or next step, so neither draws a line or a Debug panel.
  if (!runtime || !(paused || runtime.state === 'on')) return null
  return (
    <section
      aria-label="NPU prefill status"
      className="space-y-1 text-sm text-muted-foreground"
    >
      {!paused && (
        <p>NPU prefill: {valueWord('--npu-prefill', runtime.state)}</p>
      )}
      {paused && (
        <p>
          {
            guiError({ code: 'npu_paused', remedy: runtime.reason ?? '' })
              .message
          }
        </p>
      )}
      {runtime && (
        <Disclosure summary="Details" className="text-xs">
          {runtime.reason && <p>{runtime.reason}</p>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 pt-3 text-sm">
            <dt>Guard trips</dt>
            <dd className="tabular-nums">{number(runtime.trips, 0)}</dd>
            <dt title="Faults of the NPU's hardware abstraction layer">
              HAL faults
            </dt>
            <dd className="tabular-nums">{number(runtime.hal_faults, 0)}</dd>
            <dt title="Windows hardware error reports">WHEA events</dt>
            <dd className="tabular-nums">{number(runtime.whea_events, 0)}</dd>
            <dt>Invalid chunks</dt>
            <dd className="tabular-nums">
              {number(runtime.invalid_chunks, 0)}
            </dd>
            <dt>Last NPU request chip power</dt>
            <dd>
              {runtime.chip_power.w_mean == null ? (
                runtime.chip_power.state
              ) : (
                <>
                  <span className="tabular-nums">
                    {rate(runtime.chip_power.w_mean)} W
                  </span>
                </>
              )}
            </dd>
            <dt>Power window</dt>
            <dd className="tabular-nums">
              {windowText(runtime.chip_power.window_s)}
            </dd>
            <dt>Power scheme</dt>
            <dd title={runtime.scheme?.guid}>
              {runtime.scheme ? runtime.scheme.name : 'Unavailable'}
            </dd>
            <dt>Minimum available memory</dt>
            <dd>
              {runtime.memory ? (
                <span className="tabular-nums">
                  {bytes(runtime.memory.min_available_mib * 1024 ** 2)}
                </span>
              ) : (
                'Unavailable'
              )}
            </dd>
            <dt>Maximum commit</dt>
            <dd>
              {runtime.memory ? (
                <span className="tabular-nums">
                  {percent(runtime.memory.max_commit_pct)}
                </span>
              ) : (
                'Unavailable'
              )}
            </dd>
          </dl>
        </Disclosure>
      )}
    </section>
  )
}
