import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import { coreView } from '@/components/core-page-test-support'
import { SettingAbout } from '@/components/registry-form'
import { fieldFromRow } from '@/lib/registry'
import { registryRow } from '@/lib/registry-view'

// The owner read the effort map's (i): "Suggested range: -2 to 0." and, below it,
// "Suggested: from -2 up to 0." A range is said once, in every row's (i).
test('no row says its range twice in its About text', () => {
  const rows = coreView('stopped-saved-estimate').switches
  let said = 0
  for (const row of rows) {
    let field
    try {
      field = fieldFromRow(registryRow(row))
    } catch {
      continue
    }
    const text = renderToStaticMarkup(
      <SettingAbout
        paragraphs={[]}
        range={field.range}
        suggestion={field.suggestion}
      />,
    )
    const lines = text.match(/<p>[^<]*<\/p>/g) ?? []
    const ranges = lines.filter((line) => /Suggested/.test(line))
    expect(ranges.length, `${row.name}: ${text}`).toBeLessThanOrEqual(1)
    said += ranges.length
  }
  expect(said).toBeGreaterThan(0)
})
