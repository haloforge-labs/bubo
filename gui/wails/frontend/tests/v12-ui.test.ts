import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test } from 'vitest'

function uiFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    return entry.isDirectory()
      ? uiFiles(path)
      : /\.(tsx?|css)$/.test(path) && !/\.(test|shots)\./.test(path)
        ? [path]
        : []
  })
}

test('no UI source links to gufo-org', () => {
  for (const file of uiFiles('src'))
    expect(readFileSync(file, 'utf8'), file).not.toMatch(
      /https?:\/\/[^\s'"]*gufo-org/i,
    )
})
