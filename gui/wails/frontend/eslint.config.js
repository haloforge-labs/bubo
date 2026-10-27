import { readdirSync } from 'node:fs'
import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import hooks from 'eslint-plugin-react-hooks'
import query from '@tanstack/eslint-plugin-query'
import globals from 'globals'

const features = readdirSync(new URL('./src/features', import.meta.url), {
  withFileTypes: true,
})
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
const bindings = 'src/lib/bindings.ts'
const wailsRestriction = {
  regex: '(^|/)wailsjs(/|$)',
  message: 'Import generated Wails code only from src/lib/bindings.ts.',
}
const files = ['**/*.{js,mjs,ts,tsx}']

export default tseslint.config(
  {
    ignores: [
      'node_modules/**',
      '.vite/**',
      'dist/**',
      'wailsjs/**',
      'src/generated/**',
    ],
  },
  js.configs.recommended,
  tseslint.configs.recommended,
  {
    files,
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      'no-restricted-globals': [
        'error',
        {
          name: 'window',
          message: 'Keep application state in Query or component state.',
        },
        {
          name: 'globalThis',
          message: 'Do not use globals as an application state store.',
        },
        {
          name: 'self',
          message: 'Do not use globals as an application state store.',
        },
      ],
    },
  },
  {
    files: ['**/*.{ts,tsx}'],
    plugins: { 'react-hooks': hooks },
    rules: {
      ...hooks.configs.recommended.rules,
      // React Compiler is not enabled in Vite. Its memoization compatibility
      // warning would reject the template's supported TanStack Table v8 API.
      'react-hooks/incompatible-library': 'off',
    },
  },
  ...query.configs['flat/recommended'],
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: [
      'src/**/*.test.{ts,tsx}',
      'src/**/*.shots.{ts,tsx}',
      'src/scenarios/**',
    ],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector:
            'MemberExpression[object.name="view"][property.name="state"]',
          message:
            'Read View.phase, View.serving or a View.actions field instead of the raw lifecycle state.',
        },
        {
          selector: 'BinaryExpression[left.name="serverState"]',
          message:
            'Read the Go status label or phase instead of comparing a lifecycle state.',
        },
      ],
    },
  },
  {
    files,
    ignores: [bindings],
    rules: {
      'no-restricted-imports': ['error', { patterns: [wailsRestriction] }],
    },
  },
  ...features.map((feature) => ({
    files: [`src/features/${feature}/**/*.{ts,tsx}`],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            wailsRestriction,
            {
              regex: `(^@/features/(?:${features.filter((other) => other !== feature).join('|')})(/|$)|^\\.\\./)`,
              message:
                'A feature cannot import another feature. Use @/ for shared code and ./ for its own children.',
            },
          ],
        },
      ],
    },
  })),
)
