import { readdir, readFile } from 'node:fs/promises'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'
import { expect, test } from 'vitest'

const forbiddenNumbers = new Set([8080, 8081, 8082, 8083, 262144, 131072])
const forbiddenStrings = new Set([
  ...[...forbiddenNumbers].map(String),
  '/v1/chat/completions',
  '/v1/messages',
  '127.0.0.1',
  '0.0.0.0',
  'bubo-loopback',
  'exact',
  'stock',
  'subagents',
])
const forbiddenFragments = [
  '/v1/chat/completions',
  '/v1/messages',
  '127.0.0.1',
  '0.0.0.0',
  'bubo-loopback',
]
const forbiddenNumberInString =
  /(?:^|\D)(8080|8081|8082|8083|262144|131072)(?=\D|$)/

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = await Promise.all(
    entries.map(async (entry) => {
      if (['scenarios', 'generated', 'wailsjs'].includes(entry.name)) return []
      const path = join(directory, entry.name)
      if (entry.isDirectory()) return sourceFiles(path)
      return /\.(?:[cm]?[jt]sx?|json|css)$/.test(entry.name) ? [path] : []
    }),
  )
  return files.flat()
}

export function findFactLiterals(filename: string, text: string) {
  const source = ts.createSourceFile(
    filename,
    text,
    ts.ScriptTarget.Latest,
    true,
    filename.endsWith('.tsx')
      ? ts.ScriptKind.TSX
      : filename.endsWith('.json')
        ? ts.ScriptKind.JSON
        : ts.ScriptKind.TS,
  )
  const matches: string[] = []
  function visit(node: ts.Node) {
    const forbidden = ts.isNumericLiteral(node)
      ? forbiddenNumbers.has(Number(node.text))
      : (ts.isStringLiteral(node) ||
          ts.isNoSubstitutionTemplateLiteral(node) ||
          ts.isTemplateHead(node) ||
          ts.isTemplateMiddle(node) ||
          ts.isTemplateTail(node)) &&
        (forbiddenStrings.has(node.text) ||
          forbiddenFragments.some((fragment) => node.text.includes(fragment)) ||
          forbiddenNumberInString.test(node.text))
    if (forbidden) {
      const location = source.getLineAndCharacterOfPosition(node.getStart())
      matches.push(
        `${filename}:${location.line + 1}:${location.character + 1} ${node.getText(source)}`,
      )
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return matches
}

test('src contains no CLI-owned fact literals outside generated fixtures/types', async () => {
  const root = fileURLToPath(new URL('../src', import.meta.url))
  const files = await sourceFiles(root)
  const matches = (
    await Promise.all(
      files.map(async (file) =>
        findFactLiterals(relative(root, file), await readFile(file, 'utf8')),
      ),
    )
  ).flat()
  expect(matches, matches.join('\n')).toEqual([])
  console.log(
    `Fact guard checked ${files.length} source files; ${matches.length} forbidden literals.`,
  )
})

test('fact detection recognizes numeric, string and template literals without flagging identifiers', () => {
  expect(
    findFactLiterals(
      'probe.ts',
      'const port = 8_081; const path = "/v1/messages"; const profile = `exact`',
    ),
  ).toHaveLength(3)
  expect(
    findFactLiterals('probe.ts', 'export default function Page() {}'),
  ).toHaveLength(0)
  expect(
    findFactLiterals(
      'probe.tsx',
      '<Button variant="default" size="standard" />',
    ),
  ).toHaveLength(0)
  expect(
    findFactLiterals('probe.ts', '["exact", "stock", "subagents"]'),
  ).toHaveLength(3)
})
