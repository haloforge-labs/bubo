import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { bytes, number, percent } from '@/lib/format'
import type { ModelPlan } from '@/components/model-facts'
import type { View } from '@/lib/bindings'
import { PathText } from '@/components/path-text'

type Sample = { bytes: number; at: number }

export function transferSummary(
  done: number,
  total: number | undefined,
  samples: readonly Sample[],
  whole = false,
) {
  const parts = [
    total
      ? `${bytes(done)} of ${bytes(total)} · ${percent((done / total) * 100)}`
      : bytes(done),
  ]
  const first = samples[0]
  const last = samples.at(-1)
  if (first && last && last.at > first.at && last.bytes > first.bytes) {
    const speed = (last.bytes - first.bytes) / ((last.at - first.at) / 1000)
    parts.push(`${bytes(speed, { unit: 'MiB' })}/s`)
    if (total && total > done) {
      const seconds = Math.ceil((total - done) / speed)
      const left =
        seconds < 60
          ? `${number(seconds, 0)} s`
          : `${number(Math.ceil(seconds / 60), 0)} min`
      parts.push(whole ? `about ${left} left` : `${left} left for this file`)
    }
  }
  return parts.join(' · ')
}

export function TransferProgress({
  identity,
  done,
  total,
  scope,
  whole = false,
}: {
  identity: string
  done: number
  total?: number
  scope?: string
  // The numbers describe the whole job, not the current file.
  whole?: boolean
}) {
  const client = useQueryClient()
  const { data } = useQuery<Sample[]>({
    queryKey: ['transfer-samples', identity],
    queryFn: async () => [],
    initialData: [],
    enabled: false,
  })
  useEffect(() => {
    client.setQueryData<Sample[]>(
      ['transfer-samples', identity],
      (previous = []) => {
        const last = previous.at(-1)
        if (last?.bytes === done) return previous
        const sample = { bytes: done, at: Date.now() }
        return last && done < last.bytes
          ? [sample]
          : [...previous, sample].slice(-20)
      },
    )
  }, [client, identity, done])
  return (
    <p className="text-sm">
      {scope && <span className="tabular-nums">{scope}: </span>}
      <span className="font-mono tabular-nums">
        {transferSummary(done, total, data, whole)}
      </span>
    </p>
  )
}

// The current file of a job, for the Details of the one whole-job line.
export function CurrentFileDetails({
  position,
  done,
  total,
  plan,
  catalogueFiles,
  fileId,
  phase,
  folder,
  active = true,
}: {
  position?: { index: number; total: number }
  done: number
  total?: number
  plan?: ModelPlan
  catalogueFiles?: NonNullable<View['models']>['catalogue'][number]['files']
  fileId?: string
  phase?: string
  folder?: string
  active?: boolean
}) {
  const files = plan
    ? plan.files.map((file) => ({
        name: file.path?.split(/[\\/]/).at(-1) || file.file_id,
        size: file.size,
        source: [file.repository, file.revision].filter(Boolean).join(' · '),
        id: file.file_id,
        reused: file.state === 'reuse',
      }))
    : (catalogueFiles ?? []).map((file) => ({
        name: file.name,
        size: file.bytes,
        source: undefined,
        id: undefined,
        reused: false,
      }))
  const current = plan
    ? files.findIndex((file) => file.id === fileId)
    : position?.total === files.length
      ? position.index - 1
      : -1
  const folders = [...new Set(Object.values(plan?.paths ?? {}))]
  if (!folders.length && folder) folders.push(folder)
  return (
    <details className="text-sm text-muted-foreground">
      <summary className="cursor-pointer">Details</summary>
      <p className="tabular-nums">
        {position
          ? `File ${position.index} of ${position.total}`
          : 'Current file'}
        : {total ? `${bytes(done)} of ${bytes(total)}` : bytes(done)}
      </p>
      {folders.map((path) => (
        <p key={path} className="pt-2">
          Save folder: <PathText path={path} />
        </p>
      ))}
      {files.length > 0 && (
        <ul aria-label="Download file progress" className="space-y-3 pt-3">
          {files.map((file, index) => {
            // The CLI processes each plan in order. Without a file identity or
            // an ordinal for this list, no per-file state can be inferred.
            const state =
              current === index && active
                ? phase?.includes('verif')
                  ? 'checking'
                  : phase === 'download'
                    ? `downloading ${bytes(done)} of ${bytes(total ?? file.size)}`
                    : undefined
                : file.reused || (current >= 0 && index < current)
                  ? 'done'
                  : current >= 0 && index > current
                    ? 'waiting'
                    : undefined
            return (
              <li
                key={`${file.id ?? file.name}-${index}`}
                className="min-w-0 space-y-1 break-all"
              >
                <p className="text-foreground">{file.name}</p>
                <p>
                  <span className="tabular-nums">{bytes(file.size)}</span>
                  {state && ` · ${state}`}
                </p>
                {file.source && <p>Source: {file.source}</p>}
              </li>
            )
          })}
        </ul>
      )}
    </details>
  )
}
