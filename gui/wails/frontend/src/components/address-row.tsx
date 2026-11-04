import { CopyText } from '@/components/copy-text'

// The one address row: name, address (wraps inside its column), copy. Used for
// the server addresses on Server and Connect and for the network adapters.
export function AddressRow({
  label,
  url,
  copyLabel,
}: {
  label: string
  url: string
  copyLabel: string
}) {
  return (
    <div className="grid min-w-0 grid-cols-[6rem_minmax(0,1fr)_auto] items-start gap-x-4 border-b pb-3">
      <span>{label}</span>
      <code className="text-sm break-all">{url}</code>
      <CopyText iconOnly label={copyLabel} text={url} />
    </div>
  )
}
