// The release-owned file is bundled into the GUI, like the third-party
// inventory. A missing file produces no section. Never substitute upstream
// Gufo's changelog or keep notes from another installed version on screen.
const files = import.meta.glob<string>('../../../../../WHATSNEW.md', {
  query: '?raw',
  import: 'default',
  eager: true,
})
const text = Object.values(files)[0]?.trim() ?? ''
const [heading, ...body] = text.split(/\r?\n/)
const release = heading?.match(/^# What's new in (\S+)$/)?.[1]

export function WhatsNew({ version }: { version?: string }) {
  if (!release || release !== version) return null
  const lines = body.map((line) => line.trim()).filter(Boolean)
  if (!lines.length) return null
  return (
    <section aria-label={`What's new in ${release}`} className="space-y-4">
      <h2 className="text-2xl">What's new in {release}</h2>
      <ul className="list-disc space-y-3 pl-5">
        {lines.map((line, index) => (
          <li key={index}>
            {line
              .replace(/^- /, '')
              .split(/\*\*(.*?)\*\*/)
              .map((part, index) =>
                index % 2 ? <strong key={index}>{part}</strong> : part,
              )}
          </li>
        ))}
      </ul>
    </section>
  )
}
