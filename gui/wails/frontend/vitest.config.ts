import { fileURLToPath, URL } from 'node:url'
import { mkdir } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { defineConfig } from 'vitest/config'
import { playwright } from '@vitest/browser-playwright'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { BrowserCommandContext } from 'vitest/node'
import type { View } from './src/lib/bindings'

const shots = process.env.BUBO_SHOTS === '1'
const chartShotsDirectory = process.env.BUBO_SHOTS_DIRECTORY
if (shots && !chartShotsDirectory)
  throw new Error(
    'Set BUBO_SHOTS_DIRECTORY to an evidence folder outside the repo',
  )
const evidenceDirectory = process.env.BUBO_SHOTS_INPUT_DIRECTORY
const startupDirectory = process.env.BUBO_STARTUP_SHOTS_INPUT_DIRECTORY
const liveDirectory = process.env.BUBO_LIVE_SHOTS_INPUT_DIRECTORY
const liveRequestViews: Record<string, View> = {}
if (shots && liveDirectory) {
  for (const state of ['reading', 'writing', 'completed']) {
    liveRequestViews[state] = JSON.parse(
      readFileSync(join(liveDirectory, `live-${state}.json`), 'utf8'),
    ) as View
  }
}
const startupViews: Record<string, View> = {}
if (shots && startupDirectory) {
  for (const state of ['starting', 'loading', 'ready']) {
    startupViews[state] = JSON.parse(
      readFileSync(join(startupDirectory, `workflow-${state}.json`), 'utf8'),
    ) as View
  }
}
const tightRestartViews: Record<string, View> = {}
if (shots && evidenceDirectory) {
  for (const state of ['ready-tight-blocked', 'ready-tight-allowed']) {
    tightRestartViews[state] = JSON.parse(
      readFileSync(join(evidenceDirectory, `${state}.json`), 'utf8'),
    ) as View
  }
}
const viewport = { width: 1100, height: 760 }
const variants = ['light', 'dark'].flatMap((theme) =>
  [1, 1.5].map((scale) => ({
    theme,
    scale,
    name: `${theme}-${scale * 100}`,
  })),
)
const selectedVariant = process.env.BUBO_SHOTS_VARIANT
if (selectedVariant && !variants.some(({ name }) => name === selectedVariant))
  throw new Error(`Unknown screenshot variant: ${selectedVariant}`)
const shotVariants = selectedVariant
  ? variants.filter(({ name }) => name === selectedVariant)
  : variants

async function captureFeature(
  { frame }: BrowserCommandContext,
  feature: string,
  scenario: string,
) {
  // Inputs are test identifiers, never arbitrary filesystem paths.
  if (![feature, scenario].every((value) => /^[a-z0-9_-]+$/i.test(value)))
    throw new Error('Invalid screenshot identifier')
  const current = await frame()
  const theme = await current.locator('html').getAttribute('data-theme')
  if (theme !== 'light' && theme !== 'dark')
    throw new Error('Missing screenshot theme')
  const scale = await current.evaluate(() => {
    for (const value of [100, 150]) {
      if (matchMedia(`(resolution: ${value / 100}dppx)`).matches) return value
    }
    throw new Error('Unexpected screenshot device scale')
  })
  const lane = process.env.BUBO_SHOTS_LANE ?? 'D3'
  if (!/^[a-z0-9_-]+$/i.test(lane)) throw new Error('Invalid screenshot lane')
  const directory = join(process.env.BUBO_SHOTS_DIRECTORY ?? '', lane, feature)
  await mkdir(directory, { recursive: true })
  const path = join(directory, `${scenario}-${theme}-${scale}.png`)
  const capture = current.locator('#shell-capture')
  // Pages can scroll beyond the shell. Keep captures at the existing viewport
  // contract, and allow scenario specs to scroll the content between captures.
  await capture.evaluate((element, height) => {
    element.style.height = `${height}px`
    element.style.overflow = 'auto'
  }, viewport.height)
  const png = await capture.screenshot({
    path,
    type: 'png',
    animations: 'disabled',
  })
  const width = png.readUInt32BE(16)
  const height = png.readUInt32BE(20)
  if (
    width !== (viewport.width * scale) / 100 ||
    height !== (viewport.height * scale) / 100
  )
    throw new Error(`Unexpected PNG dimensions: ${width} x ${height}`)
  console.log(`PNG ${path} (${width} x ${height})`)
  return path
}

export default defineConfig({
  cacheDir: '.vite',
  plugins: [react(), tailwindcss()],
  server: {
    fs: {
      allow: [
        fileURLToPath(new URL('./', import.meta.url)),
        ...(chartShotsDirectory ? [chartShotsDirectory] : []),
      ],
    },
  },
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  test: {
    reporters: ['default'],
    // Each browser worker holds a ~0.65 GB headless renderer, and two jobs may
    // run this suite at once beside the 96 GB GPU carve (about 32 GB left).
    maxWorkers: 4,
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'browser',
          include: [shots ? 'src/**/*.shots.tsx' : 'src/**/*.test.tsx'],
          setupFiles: ['./tests/browser-setup.ts'],
          browser: {
            enabled: true,
            headless: true,
            // Vitest 4 scales its iframe into the Playwright context. Match both
            // viewports and disable runner UI for pixel-accurate PNGs.
            ui: false,
            provider: playwright({ contextOptions: { viewport } }),
            viewport,
            screenshotFailures: false,
            commands: { captureFeature },
            instances: shots
              ? shotVariants.map(({ name, theme, scale }) => ({
                  browser: 'chromium' as const,
                  name: `browser-${name}`,
                  provider: playwright({
                    contextOptions: {
                      viewport,
                      deviceScaleFactor: scale,
                      colorScheme: theme === 'dark' ? 'dark' : 'light',
                    },
                  }),
                  provide: {
                    theme,
                    chartShotsDirectory,
                    tightRestartViews,
                    startupViews,
                    liveRequestViews,
                  },
                }))
              : [{ browser: 'chromium', name: 'browser-chromium' }],
          },
        },
      },
    ],
  },
})

declare module 'vitest/browser' {
  interface BrowserCommands {
    captureFeature: (feature: string, scenario: string) => Promise<string>
  }
}
declare module 'vitest' {
  interface ProvidedContext {
    theme: string
    chartShotsDirectory: string
    tightRestartViews?: Record<string, View>
    startupViews?: Record<string, View>
    liveRequestViews?: Record<string, View>
  }
}
