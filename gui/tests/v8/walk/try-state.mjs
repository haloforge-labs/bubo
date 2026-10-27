import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { loopbackManifest, paths, readCapture, view } from './session.mjs'
import { leaveState, reachState } from './state.mjs'

const name = process.env.WALK_STATE ?? 'S5'
const manifest = await loopbackManifest()
let state
try {
  state = await reachState(name, manifest)
  if (name === 'S6') await state.session.page.waitForTimeout(2100)
  const v = await view(state.session)
  const file = join(paths.evidence, 'walk', `try-${name}.json`)
  writeFileSync(file, JSON.stringify({ name, view: v, capture: readCapture(state.run), console: state.session.consoleLines }, null, 2))
  console.log(JSON.stringify({ name, state: v.state, phase: v.phase, status_label: v.status_label, prepared: v.prepared, update: v.update?.state, capture_count: readCapture(state.run).length, file }))
} finally {
  if (state) await leaveState(state)
  await manifest.close()
}
