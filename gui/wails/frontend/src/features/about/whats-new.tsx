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
// The release the bundled file describes. Empty when the file is missing or malformed.
export const whatsNewRelease = heading?.match(/^# What's new in (\S+)$/)?.[1]

// True when the bundled text describes this version, so it can be shown.
export function hasWhatsNew(version?: string) {
  return (
    !!whatsNewRelease &&
    whatsNewRelease === version &&
    body.some((line) => line.trim())
  )
}

function inline(text: string) {
  return text
    .split(/\*\*(.*?)\*\*/)
    .map((part, index) =>
      index % 2 ? <strong key={index}>{part}</strong> : part,
    )
}

// The file is a short numbered list: "1. **Title.** text", optional indented
// "- detail" lines under an item, and plain lines (shown as paragraphs). A
// line with the older "- text" form, without indentation, is an item too.
type Item = { text: string; details: string[] }
const items: Item[] = []
const notes: string[] = []
for (const raw of body) {
  const line = raw.replace(/\s+$/, '')
  if (!line.trim()) continue
  const numbered = line.match(/^\d+\.\s+(.*)$/) ?? line.match(/^- (.*)$/)
  const detail = line.match(/^\s+[-*]\s+(.*)$/)
  if (numbered) items.push({ text: numbered[1]!, details: [] })
  else if (detail && items.length)
    items[items.length - 1]!.details.push(detail[1]!)
  else notes.push(line.trim())
}

export function WhatsNewList() {
  return (
    <div className="space-y-3">
      <ol className="list-decimal space-y-3 pl-5">
        {items.map((item, index) => (
          <li key={index}>
            {inline(item.text)}
            {item.details.length > 0 && (
              <ul className="mt-1 list-disc space-y-1 pl-5 text-sm">
                {item.details.map((detail, detailIndex) => (
                  <li key={detailIndex}>{inline(detail)}</li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
      {notes.map((note, index) => (
        <p key={index} className="text-sm">
          {inline(note)}
        </p>
      ))}
    </div>
  )
}

export function WhatsNew({ version }: { version?: string }) {
  if (!hasWhatsNew(version)) return null
  return (
    <section
      aria-label={`What's new in ${whatsNewRelease}`}
      className="space-y-4"
    >
      <h2 className="text-2xl">What's new in {whatsNewRelease}</h2>
      <WhatsNewList />
    </section>
  )
}
