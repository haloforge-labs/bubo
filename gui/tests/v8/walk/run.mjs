import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { controlCases, linkCase, liveLogCase, logCase, nativeCloseCase, panicCase, passiveCases, record, renderRateCase, settingsRestartCase } from './cases.mjs'
import { stateOrder } from './oracle.mjs'
import { hashInputFiles, loopbackManifest, paths, readCapture, route, view } from './session.mjs'
import { leaveState, reachState } from './state.mjs'

const liveStates = stateOrder.filter(name => name !== 'S17')
const controlsOnly = process.env.WALK_CASES === 'controls'
const linksOnly = process.env.WALK_CASES === 'links'
const focused = controlsOnly || linksOnly
// w8b runs only the updater-error log case, which otherwise waits for a full S4 walk.
const w8bOnly = process.env.WALK_CASES === 'w8b'
if (process.env.WALK_CASES && !focused && !w8bOnly) throw new Error('WALK_CASES must be controls, links or w8b')
const selected = w8bOnly ? ['S4'] : process.env.WALK_STATES ? process.env.WALK_STATES.split(',') : liveStates
for (const name of selected) if (!stateOrder.includes(name)) throw new Error(`unknown oracle state ${name}`)
const subset = selected.length !== liveStates.length || selected.some((name, index) => name !== liveStates[index])
const suffix = `${focused || w8bOnly ? `-${process.env.WALK_CASES}` : ''}${subset ? `-${selected.join('-')}` : ''}`
const results = {}
const states = {}
const consoleLines = []
const captures = []
const screenshots = []
let fatalStep = null
const manifest = await loopbackManifest()
mkdirSync(paths.shots, { recursive: true })
const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex')
const inputHashes = Object.fromEntries(hashInputFiles().filter(existsSync).map(file => [file, hash(file)]))
try {
  for (const name of w8bOnly ? [] : selected) {
    let state
    let deferred = []
    try {
      state = await reachState(name, manifest)
      // Allow the 2 s View query to paint the state that was reached in Go.
      if (!['S1', 'S3'].includes(name)) await state.session.page.waitForTimeout(2100)
      let stateView
      if (focused) {
        stateView = await view(state.session)
        states[name] = { reached: true, go_state: stateView.state, go_phase: stateView.phase,
          status_label: stateView.status_label ?? null, model_download_state: stateView.models?.download?.state ?? null,
          update_state: stateView.update?.state ?? null, stalled: Boolean(stateView.stall) }
      } else {
        await route(state.session, '#/')
        const screenshot = join(paths.shots, `${name}${suffix}.png`)
        await state.session.page.screenshot({ path: screenshot })
        screenshots.push(screenshot)
        const audit = await passiveCases(state, results)
        stateView = audit.view
        if (name === 'S12') liveLogCase(state, audit.frames, results)
        states[name] = { reached: true, go_state: audit.view.state, go_phase: audit.view.phase,
          status_label: audit.view.status_label ?? null, model_download_state: audit.view.models?.download?.state ?? null,
          update_state: audit.view.update?.state ?? null, stalled: Boolean(audit.view.stall),
          frame: audit.frames, offers: audit.offers, screenshot }
        if (name === 'S4') await linkCase(state, results)
        if (name === 'S2') await logCase(state, results, false)
        if (name === 'S12') await settingsRestartCase(state, results)
      }
      deferred = linksOnly ? await controlCases(state, results, item => item.kind === 'link', true, true) :
        await controlCases(state, results)
      console.log(`${name}: reached ${stateView.state}; ${Object.entries(results).filter(([, r]) => r.checks.some(c => c.state === name)).map(([k, r]) => `${k} ${r.checks.filter(c => c.state === name && c.passed).length}/${r.checks.filter(c => c.state === name).length}`).join(', ')}`)
    } catch (error) {
      states[name] = { reached: false, error: String(error), stack: error?.stack }
      record(results, 'W11', name, false, { error: String(error) })
      console.error(`${name}: ${error}`)
      if (error?.walkFailure) fatalStep = { state: name, folder: error.failureFolder, error: String(error) }
    } finally {
      if (state) {
        captures.push({ state: name, config: state.run.configPath, capture: readCapture(state.run) })
        consoleLines.push(...state.session.consoleLines.map(line => `${name}: ${line}`))
        await leaveState(state)
      }
    }
    if (fatalStep) break
    for (const item of deferred) {
      let replay
      const key = `W2:${item.route}:${item.text || `button-${item.index}`}`
      const count = results[key]?.checks.length ?? 0
      try {
        replay = await reachState(name, manifest)
        if (!['S1', 'S3'].includes(name)) await replay.session.page.waitForTimeout(2100)
        await controlCases(replay, results, candidate => candidate.route === item.route && candidate.kind === item.kind &&
          candidate.text === item.text && candidate.occurrence === item.occurrence)
        if ((results[key]?.checks.length ?? 0) === count) record(results, key, name, false,
          { route: item.hash, index: item.index, error: 'initial control absent on isolated replay' })
        captures.push({ state: `${name}-retry-${item.route}-${item.index}`, config: replay.run.configPath,
          capture: readCapture(replay.run) })
        consoleLines.push(...replay.session.consoleLines.map(line => `${name}-retry: ${line}`))
      } catch (error) {
        record(results, key, name, false, { route: item.hash, index: item.index, error: String(error) })
        if (error?.walkFailure) fatalStep = { state: name, folder: error.failureFolder, error: String(error) }
      } finally { if (replay) await leaveState(replay) }
      if (fatalStep) break
    }
    if (fatalStep) break
  }
  if (!fatalStep && !focused && selected.includes('S4')) {
    let updaterState
    try {
      const refusedManifest = { ...manifest, url: manifest.url.replace('/haloforge-labs/bubo/releases/latest/download/bubo-release.json', '/manifest.json') }
      updaterState = await reachState('S4', refusedManifest)
      await updaterState.session.page.waitForTimeout(500)
      await logCase(updaterState, results, true)
      captures.push({ state: 'S4-updater-error', config: updaterState.run.configPath, capture: readCapture(updaterState.run) })
      consoleLines.push(...updaterState.session.consoleLines.map(line => `S4-updater-error: ${line}`))
    } catch (error) {
      if (error?.walkFailure) throw error
      record(results, 'W8b', 'S4', false, { fixture_error: String(error) })
    } finally { if (updaterState) await leaveState(updaterState) }
  }
  if (!fatalStep && !focused && selected.includes('S5')) await renderRateCase(manifest, results)
  if (!fatalStep && !focused && selected.includes('S12')) {
    let closeState
    try {
      closeState = await reachState('S12', manifest)
      await nativeCloseCase(closeState, results)
      captures.push({ state: 'S12-native-close', config: closeState.run.configPath, capture: readCapture(closeState.run) })
    } catch (error) {
      if (error?.walkFailure) throw error
      record(results, 'W15', 'S12', false, { fixture_error: String(error) })
    } finally { if (closeState) await leaveState(closeState) }
  }
  if (!fatalStep && !focused && selected.includes('S4')) {
    let crashState
    try {
      crashState = await reachState('S4', manifest)
      await panicCase(crashState, results)
      captures.push({ state: 'S4-panic', config: crashState.run.configPath, capture: readCapture(crashState.run) })
    } catch (error) {
      if (error?.walkFailure) throw error
      record(results, 'I13', 'S4', false, { fixture_error: String(error) })
    } finally { if (crashState) await leaveState(crashState) }
  }
} catch (error) {
  fatalStep ??= { error: String(error), folder: error?.failureFolder ?? null }
  console.error(`walk stopped: ${error}`)
} finally {
  await manifest.close()
  const summary = {
    mode: focused || w8bOnly ? process.env.WALK_CASES : 'full',
    // w8b mode walks no state; it is complete when its one case was recorded.
    complete: !fatalStep && (w8bOnly ? Boolean(results.W8b?.checks.length) : !subset && selected.every(name => states[name]?.reached)),
    selected_complete: !fatalStep && (w8bOnly ? Boolean(results.W8b?.checks.length) : selected.every(name => states[name]?.reached)),
    fatal_step: fatalStep,
    render_sweep_only: { S17: '45-minute updater fixture timebox; covered by A1 render sweep and A4 F14' },
    selected_states: selected, reached_states: selected.filter(name => states[name]?.reached).length,
    input_hashes: inputHashes,
    rerun_command: `${focused || w8bOnly ? `$env:WALK_CASES='${process.env.WALK_CASES}'; ` : ''}node gui/tests/v8/walk/run.mjs --job ${paths.job} --port ${paths.port}`,
    results: Object.fromEntries(Object.entries(results).map(([name, r]) => [name, { passed: r.passed, count: r.checks.length,
      passed_count: r.checks.filter(c => c.passed).length, failed_count: r.checks.filter(c => !c.passed).length, checks: r.checks }])),
    states, screenshots,
  }
  writeFileSync(join(paths.evidence, 'walk', `summary${suffix}.json`), JSON.stringify(summary, null, 2))
  writeFileSync(join(paths.evidence, 'walk', `console${suffix}.log`), consoleLines.join('\n'))
  writeFileSync(join(paths.evidence, 'walk', `captures${suffix}.json`), JSON.stringify(captures, null, 2))
  writeFileSync(join(paths.evidence, 'walk', `rerun${suffix}.txt`), `${summary.rerun_command}\n`)
  console.log(JSON.stringify({ complete: summary.complete, reached_states: summary.reached_states,
    cases: Object.fromEntries(Object.entries(summary.results).map(([k, v]) => [k, `${v.passed_count}/${v.count}`])) }))
  if (!summary.complete || Object.values(summary.results).some(r => !r.passed)) process.exitCode = 1
}
