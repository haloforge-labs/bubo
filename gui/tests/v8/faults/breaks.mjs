import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { v8Config } from '../config.mjs'

const root = resolve(process.cwd())
const { job, port, evidence, tree, build } = v8Config()
const folder = join(evidence, 'faults')
const modelsSource = join(root, 'gui', 'supervisor', 'models_windows.go')
const viewSource = join(root, 'gui', 'supervisor', 'view_windows.go')
const exe = join(tree, 'Bubo.Control.Web.exe')
const savedExe = join(build, 'Bubo.Control.Web.A4-base.exe')
const go = (process.env.BUBO_GO ?? 'go')
const node = 'C:\\Program Files\\nodejs\\node.exe'
const replacements = {
  F6: [viewSource, 'v.Failure = &ViewFailure{Code: f.Code, Source: f.Source,',
    'v.Failure = &ViewFailure{Code: "unexpected_failure", Source: f.Source,'],
  F18: [modelsSource, 'if m.job != nil && m.job.State == "cancelling" && j.State == "running" {',
    'if false && m.job != nil && m.job.State == "cancelling" && j.State == "running" {'],
  F24: [modelsSource, 'case cancelling && res.ExitCode != 0:', 'case false && cancelling && res.ExitCode != 0:'],
}
const selected = (process.env.FAULT_BREAK_CASES ?? Object.keys(replacements).join(',')).split(',')
mkdirSync(folder, { recursive: true })
if (existsSync(savedExe)) throw new Error('previous A4 scratch GUI backup remains; inspect before breaking again')
copyFileSync(exe, savedExe)
const originalHash = createHash('sha256').update(readFileSync(savedExe)).digest('hex')
try {
  for (const id of selected) {
    const [source, before, after] = replacements[id] ?? []
    if (!source) throw new Error(`unknown scratch break ${id}`)
    const sourceText = readFileSync(source, 'utf8')
    if (sourceText.split(before).length !== 2) throw new Error(`${id} scratch target is not unique`)
    const baseCase = join(folder, `${id}.json`)
    if (!existsSync(baseCase) || !JSON.parse(readFileSync(baseCase, 'utf8')).good) {
      throw new Error(`${id} must pass on base before its scratch break`)
    }
    const preserved = readFileSync(baseCase)
    writeFileSync(join(folder, `base-${id}.json`), preserved)
    const overlaySource = join(build, `break-${id}-${source.split(/[\\/]/).at(-1)}`)
    const overlay = join(build, `break-${id}-overlay.json`)
    writeFileSync(overlaySource, sourceText.replace(before, after))
    writeFileSync(overlay, JSON.stringify({ Replace: { [source]: overlaySource } }))
    execFileSync(go, ['build', `-modfile=${join(build, 'wails-test.mod')}`, `-overlay=${overlay}`, '-trimpath',
      '-tags=production,wv2runtime.error,bubo_test', '-ldflags=-H=windowsgui', '-o', exe, '.'],
    { cwd: join(root, 'gui', 'wails'), windowsHide: true, env: { ...process.env, CGO_ENABLED: '0', GOFLAGS: '-mod=readonly' }, stdio: 'pipe' })
    let output = '', status = 0
    try {
      output = execFileSync(node, ['gui/tests/v8/faults/run.mjs', '--job', job, '--port', String(port)],
        { cwd: root, windowsHide: true, encoding: 'utf8', env: { ...process.env, FAULT_CASES: id } })
    } catch (error) {
      status = error.status
      output = `${error.stdout ?? ''}${error.stderr ?? ''}`
    }
    const broken = JSON.parse(readFileSync(baseCase, 'utf8'))
    writeFileSync(join(folder, `break-${id}.json`), JSON.stringify({ id, status, output, broken,
      base_sha256: originalHash, break_sha256: createHash('sha256').update(readFileSync(exe)).digest('hex') }, null, 2))
    writeFileSync(baseCase, preserved)
    if (status !== 1 || broken.good) throw new Error(`${id} scratch break was not caught: status=${status} ${output}`)
    console.log(`RED scratch ${id}: ${output.trim()}`)
    unlinkSync(overlay)
    unlinkSync(overlaySource)
  }
} finally {
  copyFileSync(savedExe, exe)
  unlinkSync(savedExe)
  const restored = createHash('sha256').update(readFileSync(exe)).digest('hex')
  if (restored !== originalHash) throw new Error('A4 test exe was not restored after scratch breaks')
  console.log(`A4 test exe restored: ${restored}`)
}
