// Scratch product breaks for W22-W25. Every source byte is restored and the
// tagged exe rebuilt in finally; no product change belongs in the A5 branch.
import { execFileSync, spawnSync } from 'node:child_process'
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { v8Config } from './config.mjs'

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const paths = v8Config()
const name = process.env.FLOW_BREAK
const specs = {
  W22: ['gui/wails/frontend/src/features/settings/registry-section.tsx',
    '.filter((entry) => clears.includes(entry.row.name))',
    '.filter((entry) => clears.includes(entry.row.name) && false)', false],
  W23: ['gui/supervisor/app_tray_windows.go',
    'Label: "Show window", Action: "show"',
    'Label: "Show window", Action: ""', true],
  W24: ['gui/supervisor/login_windows.go',
    'key.SetStringValue(loginRunValue, `"`+exe+`"`)',
    'key.SetStringValue(loginRunValue, "scratch wrong test path")', true],
  W25: ['gui/wails/frontend/src/features/connect/key-controls.tsx',
    'const snippet = snippetWithKey(view.pi_snippet, key)',
    'const snippet = snippetWithKey(view.pi_snippet, key.slice(1))', false],
}
if (!specs[name]) throw new Error('set FLOW_BREAK to W22, W23, W24, or W25')
const [source, from, to, skipFrontend] = specs[name]
const target = join(root, source)
const original = readFileSync(target, 'utf8')
const normalized = original.replaceAll('\r\n', '\n')
if (normalized.split(from).length !== 2) throw new Error(`${name} scratch anchor changed in ${source}`)
const artifact = join(paths.evidence, 'flows', `${name}.json`)
function build() {
  execFileSync('node', [join(root, 'gui', 'tests', 'v8', 'build.mjs'), '--job', paths.job, '--port', String(paths.port), ...(skipFrontend ? ['--skip-frontend'] : [])],
    { cwd: root, stdio: 'inherit', windowsHide: true })
}
function run() {
  return spawnSync('node', [join(root, 'gui', 'tests', 'v8', 'flows.mjs'), '--job', paths.job, '--port', String(paths.port)],
    { cwd: root, env: { ...process.env, FLOW_CASE: name }, encoding: 'utf8', windowsHide: true, timeout: 180000 })
}
let broken = false
try {
  const base = run()
  if (base.error || base.status !== 0) throw new Error(`${name} base must be GREEN before its break: ${base.error ?? base.stderr}`)
  copyFileSync(artifact, join(paths.evidence, 'flows', `base-${name}.json`))
  writeFileSync(target, normalized.replace(from, to))
  broken = true
  build()
  const result = run()
  const summary = JSON.parse(readFileSync(artifact, 'utf8'))
  copyFileSync(artifact, join(paths.evidence, 'flows', `break-${name}.json`))
  const failed = summary.results[name]?.failed ?? 0
  console.log(JSON.stringify({ case: name, break_source: source, base_exit: base.status,
    break_exit: result.status, failed, fatal_step: summary.fatal_step, output: result.stdout?.trim() }))
  if (result.error || result.status !== 1 || failed < 1 || summary.fatal_step) {
    throw new Error(`${name} scratch break did not produce a clean RED gate`)
  }
} finally {
  if (broken) {
    writeFileSync(target, original)
    build()
    if (readFileSync(target, 'utf8') !== original) throw new Error(`${name} scratch source was not restored`)
  }
}
