import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { useEffect, useRef, useState } from 'react'
import { Link } from '@tanstack/react-router'
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { useView, type View } from '@/lib/bindings'

import { number, rate, milliseconds } from '@/lib/format'
import {
  cacheWord,
  requestColumnTitle,
  requestResult,
  requestResultRaw,
} from '@/lib/presentation'
import { EmptyState } from '@/components/empty-state'
import { RequestCharts } from './charts'
import { PromptProgress } from './prompt-progress'

type Request = View['requests'][number]
const count = (value: number | undefined) =>
  value == null ? '—' : number(value, 0)
// True when one cell of this row prints a dash (the same absent values the
// columns below show as "—").
const hasDash = (row: Request) =>
  [
    row.slot,
    row.prompt_tokens,
    row.completion_tokens,
    row.ttft_ms,
    row.decode_tps,
    row.prefill_tps,
  ].some((value) => value == null) || !row.cache_kind

const columns: ColumnDef<Request>[] = [
  {
    accessorKey: 'time',
    header: 'Time',
    cell: ({ row }) => (
      <time dateTime={row.original.time} title={row.original.time}>
        {new Date(row.original.time).toLocaleTimeString(undefined, {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false,
        })}
      </time>
    ),
  },
  {
    accessorKey: 'slot',
    header: 'Session',
    cell: ({ row }) =>
      row.original.slot == null ? '—' : count(row.original.slot + 1),
  },
  {
    accessorKey: 'prompt_tokens',
    header: 'Prompt',
    cell: ({ row }) => count(row.original.prompt_tokens),
  },
  {
    accessorKey: 'completion_tokens',
    header: 'Output',
    cell: ({ row }) => count(row.original.completion_tokens),
  },
  {
    accessorKey: 'ttft_ms',
    header: 'TTFT (ms)',
    cell: ({ row }) =>
      row.original.ttft_ms == null ? '—' : milliseconds(row.original.ttft_ms),
  },
  {
    accessorKey: 'decode_tps',
    header: 'Decode (tok/s)',
    cell: ({ row }) =>
      row.original.decode_tps == null ? '—' : rate(row.original.decode_tps),
  },
  {
    accessorKey: 'prefill_tps',
    header: 'Prefill (tok/s)',
    cell: ({ row }) =>
      row.original.prefill_tps == null ? '—' : rate(row.original.prefill_tps),
  },
  {
    accessorKey: 'cache_kind',
    header: 'Cache',
    cell: ({ row }) => cacheWord(row.original.cache_kind),
  },
  {
    accessorKey: 'finish_reason',
    header: 'Finish',
    cell: ({ row }) => (
      <span title={requestResultRaw(row.original) || undefined}>
        {requestResult(row.original)}
      </span>
    ),
  },
  {
    accessorKey: 'id',
    header: 'Request',
    cell: ({ row }) => (
      <span className="font-mono text-xs">{row.original.id}</span>
    ),
  },
]

export default function RequestsPage() {
  const query = useView()
  const pending =
    query.data?.serving &&
    (query.data.slots.some((slot) => slot.processing) ||
      (query.data.queue ?? 0) > 0)
  const hasHistory = (query.data?.requests.length ?? 0) > 0
  const region = useRef<HTMLDivElement>(null)
  const [overflow, setOverflow] = useState(false)
  useEffect(() => {
    const viewport = region.current?.querySelector(
      '[data-slot="table-container"]',
    )
    const table = viewport?.querySelector('table')
    if (!viewport || !table) return
    const observer = new ResizeObserver(() =>
      setOverflow(viewport.scrollWidth > viewport.clientWidth),
    )
    observer.observe(viewport)
    observer.observe(table)
    return () => observer.disconnect()
  }, [hasHistory])
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'time', desc: true },
  ])
  const table = useReactTable({
    data: query.data?.requests ?? [],
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: (row) => row.id,
  })
  return (
    <Page title="Requests">
      <div className="space-y-3">
        {query.data && <PromptProgress view={query.data} />}
        {!hasHistory && (
          <EmptyState>
            {pending ? 'No completed requests yet.' : 'No requests yet.'}
          </EmptyState>
        )}
        {hasHistory && query.data && <RequestCharts view={query.data} />}
        {table.getRowModel().rows.some((row) => hasDash(row.original)) && (
          <p className="text-sm text-muted-foreground">— means no value.</p>
        )}
        {query.error && (
          <ErrorNote>
            <span>{query.error.message}</span>
          </ErrorNote>
        )}
        {hasHistory && (
          <div
            ref={region}
            role="region"
            aria-label="Requests table"
            tabIndex={0}
            className="overflow-auto"
          >
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((group) => (
                  <TableRow key={group.id}>
                    {group.headers.map((header) => (
                      <TableHead
                        scope="col"
                        key={header.id}
                        aria-sort={
                          header.column.getIsSorted() === 'asc'
                            ? 'ascending'
                            : header.column.getIsSorted() === 'desc'
                              ? 'descending'
                              : 'none'
                        }
                        className={
                          ['finish_reason', 'cache_kind'].includes(header.id)
                            ? 'px-1.5'
                            : 'px-1.5 text-right'
                        }
                      >
                        <button
                          title={requestColumnTitle[header.id]}
                          className="min-h-10 max-w-36 whitespace-normal leading-tight"
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                          {header.column.getIsSorted() === 'asc'
                            ? ' ↑'
                            : header.column.getIsSorted() === 'desc'
                              ? ' ↓'
                              : ''}
                        </button>
                      </TableHead>
                    ))}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.map((row) => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className={
                          cell.column.id === 'finish_reason'
                            ? 'w-40 min-w-32 px-1.5 whitespace-normal'
                            : `px-1.5 whitespace-nowrap ${cell.column.id === 'cache_kind' ? '' : 'text-right font-mono'}`
                        }
                      >
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext(),
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
        {hasHistory && (
          <p className="text-sm text-muted-foreground">
            <span>
              <span className="tabular-nums">
                {number(table.getRowModel().rows.length, 0)}
              </span>{' '}
              {table.getRowModel().rows.length === 1 ? 'request' : 'requests'}
              {pending ? ' in history' : ''}
            </span>
            {overflow ? ' · Scroll sideways for all columns.' : ''}
          </p>
        )}
        <Link to="/logs" className="underline underline-offset-4">
          View logs
        </Link>
      </div>
    </Page>
  )
}
