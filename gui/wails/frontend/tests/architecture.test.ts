import { ESLint } from 'eslint'
import { expect, test } from 'vitest'

const eslint = new ESLint()
test.each([
  [
    'src/features/server/probe.tsx',
    'import Page from "@/features/models"; void Page',
    'no-restricted-imports',
  ],
  [
    'src/features/server/probe.tsx',
    'import Page from "../models"; void Page',
    'no-restricted-imports',
  ],
  [
    'src/components/probe.ts',
    'import { GetView } from "../../wailsjs/go/main/App"; void GetView',
    'no-restricted-imports',
  ],
  [
    'src/lib/probe.ts',
    'const value: any = 1; void value',
    '@typescript-eslint/no-explicit-any',
  ],
  ['src/lib/probe.ts', 'window.state = 1', 'no-restricted-globals'],
])('architecture lint rejects %s (%s)', async (filePath, code, rule) => {
  const results = await eslint.lintText(code, { filePath })
  expect(
    results
      .flatMap((result) => result.messages)
      .some((message) => message.ruleId === rule),
  ).toBe(true)
})

test('generated Wails imports are allowed from the bindings adapter', async () => {
  const results = await eslint.lintText(
    'import { GetView } from "../../wailsjs/go/main/App"; void GetView',
    { filePath: 'src/lib/bindings.ts' },
  )
  expect(
    results
      .flatMap((result) => result.messages)
      .filter((message) => message.ruleId === 'no-restricted-imports'),
  ).toEqual([])
})
