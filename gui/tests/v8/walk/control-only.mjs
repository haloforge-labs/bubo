import { controlCases } from './cases.mjs'
import { loopbackManifest } from './session.mjs'
import { leaveState, reachState } from './state.mjs'

const name = process.env.WALK_STATE ?? 'S4'
const route = process.env.WALK_CONTROL_ROUTE ?? 'About'
const text = process.env.WALK_CONTROL_TEXT ?? 'Copy version'
const occurrence = process.env.WALK_CONTROL_OCCURRENCE
const manifest = await loopbackManifest()
let state
try {
  state = await reachState(name, manifest)
  await state.session.page.waitForTimeout(2100)
  const results = {}
  await controlCases(state, results, item => item.route === route && item.text === text &&
    (occurrence == null || item.occurrence === Number(occurrence)))
  const after = await state.session.page.evaluate(() => ({ hash: location.hash,
    buttons: [...document.querySelectorAll('button')].map(b => ({ text: b.textContent?.trim(), aria: b.getAttribute('aria-label') })),
    dialogs: [...document.querySelectorAll('[role=dialog]')].map(d => d.textContent?.slice(0, 300)) }))
  console.log(JSON.stringify({ name, route, text, results, after }))
} finally {
  if (state) await leaveState(state)
  await manifest.close()
}
