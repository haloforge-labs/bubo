import { InfoIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { actions, type View } from '@/lib/bindings'
import { bytes } from '@/lib/format'

export type CatalogueInfo = NonNullable<View['models']>['catalogue'][number]

export function ModelInfo({ entry }: { entry: CatalogueInfo }) {
  if (!entry.about?.trim()) return null
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`About ${entry.display_name}`}
        >
          <InfoIcon />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[80vh] overflow-y-auto">
        <DialogTitle className="pr-6 text-xl font-normal">
          {entry.display_name}
        </DialogTitle>
        <DialogDescription className="sr-only">Model details</DialogDescription>
        {entry.about && <p className="text-sm">{entry.about}</p>}
        {!!entry.about_links?.length && (
          <div className="flex flex-wrap gap-3 text-sm">
            {entry.about_links.map((url) => (
              <a
                key={url}
                href={url}
                className="underline"
                onClick={(event) => {
                  event.preventDefault()
                  void actions.OpenExternal(url)
                }}
              >
                {url.replace('https://arxiv.org/abs/', 'arXiv ')}
              </a>
            ))}
          </div>
        )}
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
          {entry.technical_name && (
            <>
              <dt className="text-muted-foreground">Technical name</dt>
              <dd className="break-words">{entry.technical_name}</dd>
            </>
          )}
          <dt className="text-muted-foreground">Download</dt>
          <dd className="tabular-nums">{bytes(entry.download_bytes)}</dd>
          {entry.gpu_bytes != null && (
            <>
              <dt className="text-muted-foreground">On GPU</dt>
              <dd className="tabular-nums">{bytes(entry.gpu_bytes)}</dd>
            </>
          )}
          {entry.system_memory_bytes != null && (
            <>
              <dt className="text-muted-foreground">In system memory</dt>
              <dd className="tabular-nums">
                {bytes(entry.system_memory_bytes)}
              </dd>
            </>
          )}
        </dl>
        {entry.system_memory_note && entry.system_memory_bytes != null && (
          <p className="text-sm text-muted-foreground">
            {entry.system_memory_note}
          </p>
        )}
        {(entry.licence.prompt_line || entry.licence.name) && (
          <p className="text-sm">
            {entry.licence.prompt_line || entry.licence.name}
          </p>
        )}
        {entry.licence.note &&
          entry.licence.note !== entry.licence.prompt_line && (
            <p className="text-sm text-muted-foreground">
              {entry.licence.note}
            </p>
          )}
        {entry.disabled_reason && (
          <p className="text-sm">{entry.disabled_reason}</p>
        )}
      </DialogContent>
    </Dialog>
  )
}
