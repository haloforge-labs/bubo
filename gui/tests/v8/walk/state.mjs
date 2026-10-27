import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { startTestGUI, stopTestGUI } from '../launcher.mjs'
import { assertIsolated, call, launchRun, loopbackManifest, makeRun, paths, scenario, stopRun, view } from './session.mjs'

const memory = { S7: 'stopped-saved-estimate', S8: 'memory-gpu-headroom', S9: 'memory-default-setup', S10: 'no-fit', S19: 'memory-smaller-context', S20: 'models-warnings' }
const fakeExe = join(paths.tree, 'bin', 'bubo.exe')

export async function waitView(session, predicate, timeout = 10000) {
  const end = Date.now() + timeout
  let last
  while (Date.now() < end) {
    last = await view(session)
    if (predicate(last)) return last
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  throw new Error(`state was not reached within ${timeout} ms: ${JSON.stringify({ state: last?.state, phase: last?.phase, initialized: last?.initialized, failure: last?.failure })}`)
}

// Each state owns one isolated LOCALAPPDATA, fake CLI directory and GUI job.
// A pending action Promise may remain active while a transient state is read.
export async function reachState(name, manifest, extra = {}) {
  const opts = {}
  if (memory[name]) opts.scenario = memory[name]
  if (name === 'S4') opts.scenario = 'not-prepared'
  if (name === 'S1') opts.env = { BUBO_FAKE_STATUS_DELAYS: '90s,0s' }
  if (name === 'S3') opts.env = { BUBO_FAKE_STATUS_DELAYS: '120s,120s,0s' }
  if (name === 'S1' || name === 'S3') opts.watchStall = true
  if (['S6', 'S11', 'S12', 'S13', 'S14', 'S15', 'S18'].includes(name)) opts.prepared = true
  if (name === 'S5' || name === 'S6') opts.env = { BUBO_FAKE_PROGRESS_PAUSE_MS: '9000' }
  if (name === 'S11') opts.env = { BUBO_FAKE_LOAD_MS: '60000' }
  if (name === 'S15') opts.mode = 'serve-runtime-missing'
  if (name === 'S12') opts.env = { BUBO_FAKE_SERVER_LOG_FIXTURE: fileURLToPath(new URL('../../fakecli/recorded-server-info.log', import.meta.url)) }
  if (name === 'S13' || name === 'S18') {
    opts.env = { BUBO_FAKE_HTTP: 'stuck' }
    opts.settings = { port: 19134 }
    if (name === 'S18') opts.stallSeconds = 1
  }
  let updateServer
  if (name === 'S16') {
    const folder = join(paths.evidence, 'walk', 'updater', `S16-${Date.now()}`)
    mkdirSync(folder, { recursive: true })
    updateServer = await loopbackManifest((req, res) => {
      const file = req.url?.endsWith('/bubo-release.json') ? join(folder, 'bubo-release.json') :
        req.url?.endsWith('.zip') ? join(folder, 'bubo-v0.1.1-windows-x64-gfx1151.zip') : null
      if (!file || !existsSync(file)) { res.writeHead(404); res.end(); return }
      const data = readFileSync(file)
      res.writeHead(200, { 'content-length': data.length, 'content-type': file.endsWith('.zip') ? 'application/zip' : 'application/json' })
      res.end(data)
    })
    execFileSync((process.env.BUBO_GO ?? 'go'), ['run', './v8/updaterfixture', '-folder', folder, '-origin', new URL(updateServer.url).origin],
      { cwd: fileURLToPath(new URL('../../', import.meta.url)), encoding: 'utf8', windowsHide: true })
    manifest = updateServer
    opts.prepared = true
  }
  // The fake answers models list, status, estimates and plans from the E21
  // recordings (candidate exe 4924e51e); a state with its own scenario keeps its
  // recorded status.
  opts.env = { ...opts.env, ...extra.env }
  opts.settings = { ...opts.settings, ...extra.settings }
  const run = makeRun(`walk-${name}`, manifest, opts)
  // Keep the recorded model progress shape while extending its step count, so
  // the real GUI can be audited before the background download completes.
  if (name === 'S6') writeFileSync(join(run.fakeDir, 'model-steps'), '40')
  let session, otherJob, parkedCLI, pending
  try {
    if (name === 'S2') {
      parkedCLI = `${fakeExe}.S2-parked`
      if (existsSync(parkedCLI)) throw new Error('previous S2 parked fake CLI was not restored')
      renameSync(fakeExe, parkedCLI)
    }
    if (name === 'S14') {
      mkdirSync(join(paths.evidence, 'walk', 'pid'), { recursive: true })
      otherJob = await startTestGUI({ exe: fakeExe, cwd: paths.tree, args: ['serve', '--jsonl', '--host', '127.0.0.1'],
        environment: run.env, pidFile: join(paths.evidence, 'walk', 'pid', `${run.id}-external.txt`) })
      assertIsolated(otherJob)
      const instance = join(run.fakeDir, 'instance.json')
      const end = Date.now() + 10000
      while (Date.now() < end && (!existsSync(instance) || JSON.parse(readFileSync(instance, 'utf8')).phase !== 'ready')) {
        assertIsolated(otherJob)
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      if (!existsSync(instance) || JSON.parse(readFileSync(instance, 'utf8')).phase !== 'ready') throw new Error('external fake CLI did not reach Ready')
    }
    session = await launchRun(run, { probeDiagnostics: extra.probeDiagnostics })
    if (name === 'S1') await waitView(session, v => !v.initialized && v.startup_phase === 'starting_cli', 3000)
    else if (name === 'S3') {
      await waitView(session, v => !v.initialized, 3000)
      await new Promise(resolve => setTimeout(resolve, 61000))
    } else if (name === 'S2') await waitView(session, v => v.initialized && v.state === 'Failed', 7000)
    else if (name === 'S4' || memory[name]) await waitView(session, v => v.initialized && (name === 'S4' ? v.state === 'Not prepared' : v.state === 'Stopped'), 7000)
    else if (name === 'S5') {
      await waitView(session, v => v.initialized && v.state === 'Not prepared')
      pending = call(session, 'Prepare')
      await waitView(session, v => v.state === 'Preparing' && v.approval != null)
      const approval = await call(session, 'Approve')
      if (approval.error) throw new Error(`prepare approval failed: ${JSON.stringify(approval.error)}`)
      await waitView(session, v => v.state === 'Preparing' && v.progress != null)
    } else if (name === 'S6') {
      await waitView(session, v => v.initialized && v.models?.catalogue?.length > 0)
      const first = (await view(session)).models.catalogue.find(m => m.download_enabled)
      if (!first) throw new Error('no downloadable default model')
      const plan = await call(session, 'Download', first.id)
      if (plan.error) throw new Error(`model plan failed: ${JSON.stringify(plan.error)}`)
      pending = call(session, 'ConfirmDownload', first.id)
      await waitView(session, v => v.models?.download?.state === 'running')
    } else if (['S11', 'S12', 'S13', 'S15', 'S18'].includes(name)) {
      await waitView(session, v => v.initialized && v.state === 'Stopped')
      pending = call(session, 'Start')
      if (name === 'S11') await waitView(session, v => v.state === 'Starting')
      else if (name === 'S15') await waitView(session, v => v.state === 'Failed', 12000)
      else {
        await waitView(session, v => v.state === 'Ready', 12000)
        if (name === 'S13' || name === 'S18') {
          const response = await fetch('http://127.0.0.1:19134/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ messages: [{ role: 'user', content: 'walk fixture request' }], max_tokens: 5 }) })
          if (!response.ok) throw new Error(`fake request returned ${response.status}`)
          await response.arrayBuffer()
          await waitView(session, v => v.slots?.some(slot => slot.processing), 5000)
          if (name === 'S18') await waitView(session, v => v.stall != null, 6000)
        }
      }
    } else if (name === 'S14') await waitView(session, v => v.initialized && v.state === 'Attached-external', 10000)
    else if (name === 'S16') {
      await waitView(session, v => v.initialized && v.state === 'Stopped')
      const checked = await call(session, 'CheckUpdate')
      if (checked.error) throw new Error(`updater check failed: ${JSON.stringify(checked.error)}`)
      await waitView(session, v => v.update?.state === 'available')
      pending = call(session, 'DownloadUpdate')
      await waitView(session, v => v.update?.state === 'ready', 15000)
    }
    else throw new Error(`${name} updater fixture is not yet prepared`)
    return { name, run, session, otherJob, parkedCLI, pending, updateServer }
  } catch (error) {
    if (session) await stopRun(session)
    if (otherJob) await stopTestGUI(otherJob)
    if (parkedCLI) renameSync(parkedCLI, fakeExe)
    if (updateServer) await updateServer.close()
    throw error
  }
}

export async function leaveState(state) {
  try {
    if (state.session) await stopRun(state.session)
  } finally {
    if (state.otherJob) await stopTestGUI(state.otherJob)
    if (state.parkedCLI) renameSync(state.parkedCLI, fakeExe)
    await state.pending?.catch(() => {})
    if (state.updateServer) await state.updateServer.close()
    // The walk has already saved screenshots, captures, and summaries. Keep
    // the adjacent app settings and logs while dropping WebView2's cache.
    const localRoot = resolve(paths.evidence, 'walk', 'local')
    const profile = resolve(state.run.local, 'Bubo', 'gui', 'webview2')
    const owned = relative(localRoot, profile)
    if (!owned || owned.startsWith('..') || resolve(localRoot, owned) !== profile) throw new Error(`walk profile escapes evidence root: ${profile}`)
    // WebView2's helper processes can hold the profile for a moment after the
    // GUI exits. Retry; a profile that stays locked is a disk note, not a case.
    for (let attempt = 0; existsSync(profile); attempt++) {
      try {
        rmSync(profile, { recursive: true, force: true })
      } catch (error) {
        if (attempt >= 20) {
          mkdirSync(join(paths.evidence, 'walk', 'cleanup'), { recursive: true })
          writeFileSync(join(paths.evidence, 'walk', 'cleanup', `${state.run.id}.txt`), `profile left in place: ${profile}\n${error}\n`)
          break
        }
        await new Promise(resolve => setTimeout(resolve, 500))
      }
    }
  }
}
