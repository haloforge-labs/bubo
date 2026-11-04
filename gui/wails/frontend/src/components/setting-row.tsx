import {
  useLayoutEffect,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
} from 'react'
import { AboutPopup } from '@/components/about-popup'
import { Label } from '@/components/ui/label'
import { pageGrid } from '@/components/page-grid'
import { cn } from '@/lib/utils'

// The one setting row, used for every setting on every page (Server, Settings
// App / Advanced / Debug, the file pickers, the byte fields, the unset and
// read-only states): label, control, a hint of at most three words, then the ⓘ
// that holds the About text and the default. A Reset sits beside the hint, and
// only while the value is changed or saved.
//
// One rule sizes every row, from the width of its own cell (a container), never
// from the window: from 22rem the row is label, control, hint and ⓘ on one line;
// narrower, the visible hint is dropped (it stays behind the ⓘ) and the label and
// control columns shrink. So two columns of rows fit down to about 33rem of page.
// The width below which the visible hint moves behind the ⓘ: 22rem, the same
// width the CSS columns switch at (the app's root font size is not 16 px, so the
// rem is read from the page). The hint is rendered in one place only.
const narrowBelowRem = 22
function useNarrow() {
  const ref = useRef<HTMLDivElement>(null)
  const [narrow, setNarrow] = useState(false)
  useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    const measure = () =>
      setNarrow(
        node.clientWidth <
          narrowBelowRem *
            parseFloat(getComputedStyle(document.documentElement).fontSize),
      )
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, narrow] as const
}

export function SettingRow({
  name,
  label,
  htmlFor,
  control,
  unit,
  hint,
  about,
  reset,
  wide,
  full,
  error,
}: {
  name?: string
  label: string
  htmlFor?: string
  control: ReactNode
  // After the control, in the control's cell (ms, MiB, tokens).
  unit?: string
  // At most three words. The CLI's reason for an unavailable setting goes here
  // only as a short phrase; the full reason is part of `about`.
  hint?: ReactNode
  // The full explanation, the default and the range: everything behind the ⓘ.
  about?: ReactNode
  // A Reset / Undo button, present only while there is something to reset.
  reset?: ReactNode
  // The control grows into the width the hint would have (a path, a model name,
  // a long choice, three numbers). The hint then sits under it.
  wide?: boolean
  // The row spans every column of its group (only a row that needs the width of
  // the whole page).
  full?: boolean
  error?: ReactNode
}) {
  const [ref, narrow] = useNarrow()
  return (
    <div
      ref={ref}
      data-setting={name}
      role="group"
      aria-label={label}
      className={cn('@container min-w-0', full && 'col-span-full')}
    >
      <div
        className={cn(
          'grid items-center gap-x-2 gap-y-0.5 [--cw:6.5rem] [--lw:6.5rem] @min-[22rem]:[--lw:8rem]',
          '@min-[22rem]:[--cw:8rem]',
          wide
            ? 'grid-cols-[var(--lw)_minmax(0,1fr)_auto]'
            : 'grid-cols-[var(--lw)_minmax(0,var(--cw))_auto] @min-[22rem]:grid-cols-[var(--lw)_minmax(0,var(--cw))_minmax(0,1fr)_auto]',
        )}
      >
        <Label htmlFor={htmlFor} className="leading-snug">
          {label}
        </Label>
        <div
          className={cn(
            'flex min-w-0 items-center gap-x-3 gap-y-1',
            wide && 'flex-wrap',
          )}
        >
          {control}
          {unit && (
            <span className="shrink-0 text-xs text-muted-foreground">
              {unit}
            </span>
          )}
          {wide && reset}
        </div>
        {wide ? (
          hint &&
          !narrow && (
            <span className="col-start-2 row-start-2 min-w-0 text-xs text-muted-foreground">
              {hint}
            </span>
          )
        ) : (
          <span className="col-span-2 col-start-2 row-start-2 min-w-0 text-xs text-muted-foreground empty:hidden @min-[22rem]:col-span-1 @min-[22rem]:col-start-3 @min-[22rem]:row-start-1">
            {hint && !narrow && <span>{hint}</span>}
            {reset}
          </span>
        )}
        {(about || hint) && (
          <AboutPopup
            label={label}
            className={
              wide
                ? 'col-start-3'
                : 'col-start-3 row-start-1 @min-[22rem]:col-start-4'
            }
          >
            {/* The hint is dropped from the row when the cell is narrow. */}
            {hint && narrow && <p>{hint}</p>}
            {about}
          </AboutPopup>
        )}
        {error && <div className="col-span-full">{error}</div>}
      </div>
    </div>
  )
}

// A group of rows. Columns follow the page's own width (pageGrid); a group that
// reads better as a list (the files and cache group) passes one column.
export function SettingGroup({
  className,
  one,
  ...props
}: ComponentProps<'div'> & { one?: boolean }) {
  return (
    <div
      {...props}
      className={cn(one ? 'grid grid-cols-1 gap-y-2' : pageGrid, className)}
    />
  )
}
