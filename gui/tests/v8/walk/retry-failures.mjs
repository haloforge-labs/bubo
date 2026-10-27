import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { controlCases } from './cases.mjs'
import { stateOrder } from './oracle.mjs'
import { loopbackManifest, paths, readCapture } from './session.mjs'
import { leaveState, reachState } from './state.mjs'

const baselinePath = join(paths.evidence, 'walk', 'summary.json')
const baselineBytes = readFileSync(baselinePath)
const baseline = JSON.parse(baselineBytes)
if (!baseline.complete) throw new Error('W2 retry requires a complete baseline summary')
const targets = new Map()
for (const [key, result] of Object.entries(baseline.results)) {
  if (!key.startsWith('W2:')) continue
  for (const check of result.checks.filter(check => !check.passed)) {
    if (!Number.isInteger(check.detail?.index) || !check.detail?.route) continue
    const items = targets.get(check.state) ?? []
    items.push({ key, route: check.detail.route, index: check.detail.index })
    targets.set(check.state, items)
  }
}
const manifest = await loopbackManifest()
const replay = { baseline: baselinePath, baseline_sha256: createHash('sha256').update(baselineBytes).digest('hex'), states: {} }
try {
  for (const name of stateOrder.filter(name => targets.has(name))) {
    let state
    const wanted = targets.get(name)
    const results = {}
    try {
      state = await reachState(name, manifest)
      if (!['S1', 'S3'].includes(name)) await state.session.page.waitForTimeout(2100)
      await controlCases(state, results, item => item.kind === 'button' &&
        wanted.some(target => target.route === item.hash && target.index === item.index))
      replay.states[name] = { targets: wanted, results, config: state.run.configPath, capture: readCapture(state.run),
        console: state.session.consoleLines }
      console.log(`${name}: ${wanted.map(target => `${target.key}[${target.index}]=${
        results[target.key]?.checks.find(check => check.detail?.index === target.index)?.passed ?? 'missing'}`).join(', ')}`)
    } catch (error) {
      replay.states[name] = { targets: wanted, error: String(error) }
      console.error(`${name}: ${error}`)
    } finally { if (state) await leaveState(state) }
  }
} finally {
  await manifest.close()
  writeFileSync(join(paths.evidence, 'walk', 'W2-retries.json'), JSON.stringify(replay, null, 2))
}
for (const { targets: wanted, results } of Object.values(replay.states)) {
  for (const target of wanted) {
    if (results?.[target.key]?.checks.find(check => check.detail?.index === target.index)?.passed !== true) process.exitCode = 1
  }
}
