import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { existsSync, readFileSync } from 'node:fs'

const lock = process.env.BUBO_GPU_LOCK
if (!lock) throw new Error('Set BUBO_GPU_LOCK to the shared timing lock path')
if (existsSync(lock) && /timing|owner/i.test(readFileSync(lock, 'utf8')))
  throw new Error(
    'GPU timing owner is active. Wait before running browser screenshots.',
  )

const result = spawnSync(
  process.execPath,
  [
    fileURLToPath(
      new URL('../node_modules/vitest/vitest.mjs', import.meta.url),
    ),
    'run',
    '--project',
    'browser',
    ...process.argv.slice(2),
  ],
  {
    stdio: 'inherit',
    env: { ...process.env, BUBO_SHOTS: '1' },
    windowsHide: true,
  },
)
if (result.error) throw result.error
process.exitCode = result.status ?? 1
