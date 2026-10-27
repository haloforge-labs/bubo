import { execFileSync, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { appendFileSync, cpSync, existsSync, linkSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { isAbsolute, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { assertIsolated, loopbackManifest, makeRun, paths } from '../walk/session.mjs'
import { attachToPage } from '../cdp.mjs'
import { cleanTestEnvironment, readExits, startTestGUI, stopTestGUI } from '../launcher.mjs'

// Two real helper transactions. Every executable, state file, fake runtime,
// WebView profile and loopback release is below A4 evidence. The helper and
// restarted GUI inherit desktopctl's private desktop and Job Object.
const root = resolve(fileURLToPath(new URL('../../../../', import.meta.url)))
const frontend = join(root, 'gui', 'wails', 'frontend')
const { chromium } = createRequire(join(frontend, 'package.json'))('playwright')
const go = (process.env.BUBO_GO ?? 'go')
const folder = join(paths.evidence, 'faults', 'runtime-update')
mkdirSync(folder, { recursive: true })
const results = []
const selected = new Set((process.env.RUNTIME_CASES || 'U1,U2,U3,U4,U5,F19b,P1,P2,P3').split(','))
// P1 and P2 install under a folder with a space and a non-ASCII letter. The
// helper starts the GUI there with that folder as working directory (health
// start of the new GUI, restartOld after a rollback), as Explorer would.
const spacedName = 'Bubo Wälk'
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

// E7 diagnosis hooks (test-only, off by default). RUNTIME_TRACE=<file> appends a
// timestamped line per harness step; RUNTIME_KEEP=1 keeps every run's install,
// staging and WebView profile folders so a failed run can be read afterwards.
function stamp() { const d = new Date(); return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0') }
function trace(message) {
  if (process.env.RUNTIME_TRACE) appendFileSync(process.env.RUNTIME_TRACE, JSON.stringify({ t: stamp(), ev: 'harness', msg: message }) + '\n')
}
const keep = process.env.RUNTIME_KEEP === '1'

function record(id, good, observed) {
  const row = { id, good, observed }
  results.push(row)
  writeFileSync(join(folder, `${id}.json`), JSON.stringify(row, null, 2))
  console.log(`${good ? 'PASS' : 'RED'} ${id}: ${JSON.stringify(observed)}`)
}
async function closeAll(steps) {
  const failures = []
  for (const [name, close] of steps) {
    try { await close() }
    catch (error) { failures.push(`${name}: ${error}`) }
  }
  return failures
}
// WebView2 helpers can hold the profile for a while after the GUI exits.
// Retry for 10 s; a profile that stays locked becomes a disk note and never
// replaces the error that the caller is reporting.
async function safeRemove(target, ancestor) {
  if (keep || !existsSync(target)) return
  const rel = relative(realpathSync(ancestor), realpathSync(target))
  if (!rel || rel === '..' || rel.startsWith(`..\\`) || isAbsolute(rel)) throw new Error(`refusing cleanup outside ${ancestor}: ${target}`)
  for (let attempt = 0; existsSync(target); attempt++) {
    try {
      rmSync(target, { recursive: true, force: true })
    } catch (error) {
      if (attempt >= 20) {
        mkdirSync(join(ancestor, 'cleanup'), { recursive: true })
        writeFileSync(join(ancestor, 'cleanup', `scratch-${Date.now()}.txt`), `scratch left in place: ${target}\n${error}\n`)
        return
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
}
function runtimeCalls(run) {
  const file = join(run.fakeDir, 'runtime-children.jsonl')
  if (!existsSync(file)) return []
  return readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse)
}
function ndjson(file) { return readFileSync(file, 'utf8').split(String.fromCharCode(10)).filter(Boolean).map(l => JSON.parse(l)) }
function approved(run) { return existsSync(join(run.fakeDir, 'prepare-approval.jsonl')) }

// A GUI start is finished when its WebView2 page exists, not when something listens
// on the CDP port (see ../cdp.mjs). The old GUI's browser can still hold the port
// and the profile when the helper starts the new GUI, and the new browser opens the
// port before it has a page. Only the deadline ends the wait; the helper's own
// health deadline is 90 s.
async function connect(child, run, wait = 60000) {
  trace(`connect start (child ${child.pid})`)
  const end = Date.now() + wait
  let attached
  try {
    attached = await attachToPage(chromium, paths.cdpURL, { wait, alive: () => assertIsolated(child), onWait: text => trace(`connect: ${text}`) })
  } catch (error) {
    throw new Error(`${error.message}; process exits: ${JSON.stringify(readExits(child).map(x => ({ image: x.image, code: x.code })))}`)
  }
  const { browser, page } = attached
  trace(`connect: attached, pages=${browser.contexts()[0]?.pages().length}`)
  const errors = []
  page.on('console', m => { if (m.type() === 'error') errors.push(`console: ${m.text()}`) })
  page.on('pageerror', e => errors.push(`page: ${e.stack ?? e}`))
  await page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: Math.max(10000, end - Date.now()) })
  if (errors.length) throw new Error(`page error: ${errors[0]}`)
  return { browser, page, errors, run, child }
}
async function view(session) {
  const v = await session.page.evaluate(() => globalThis.go.main.App.GetView())
  if (session.errors.length) throw new Error(`page error: ${session.errors[0]}`)
  assertIsolated(session.child)
  return v
}
async function call(session, name) {
  const r = await session.page.evaluate(name => globalThis.go.main.App[name](), name)
  if (session.errors.length) throw new Error(`page error: ${session.errors[0]}`)
  assertIsolated(session.child)
  return r
}
async function waitView(session, pred, ms = 12000) {
  const end = Date.now() + ms
  let last
  while (Date.now() < end) {
    last = await view(session)
    if (pred(last)) return last
    await sleep(100)
  }
  throw new Error(`GUI state did not progress: ${JSON.stringify({ state: last?.state, phase: last?.phase, update: last?.update?.state })}`)
}
// opts.name: install folder name. The first GUI starts in the ASCII base folder,
// so only the helper's own starts use the install folder as working directory.
// opts.candidate: a stand-in for the new
// GUI exe; the helper then has to roll back (opts.expect 'rolled_back').
async function transact(label, changed, opts = {}) {
  const name = opts.name ?? 'Bubo'
  const expect = opts.expect ?? 'installed'
  const base = join(folder, `${label}-${Date.now()}`)
  const install = join(base, name)
  const bin = join(install, 'bin')
  const release = join(base, 'release')
  const helper = join(base, 'built-helper.exe')
  mkdirSync(bin, { recursive: true })
  mkdirSync(release, { recursive: true })
  const sourceGUI = join(paths.tree, 'Bubo.Control.Web.exe')
  const sourceCLI = join(paths.tree, 'bin', 'bubo.exe')
  // The updater swaps package folders; it never writes these executables in
  // place. The old package can share their bytes with the built test tree.
  linkSync(sourceGUI, join(bin, 'Bubo.Control.Web.exe'))
  linkSync(sourceCLI, join(bin, 'bubo.exe'))
  writeFileSync(join(install, 'fixture-version'), '0.1.0\n')
  // The installed package's own version record, as a release carries it.
  writeFileSync(join(install, 'BUILD-MANIFEST.json'), JSON.stringify({ product: { product_version: '0.1.0' },
    gui: { commit: '0'.repeat(40), runtime_id: 'fixture-runtime-old' } }) + '\n')
  execFileSync(go, ['build', '-ldflags=-H=windowsgui', '-o', helper, './cmd/bubo-update'],
    { cwd: join(root, 'gui', 'updater'), windowsHide: true, stdio: 'pipe' })
  cpSync(helper, join(bin, 'bubo-update.exe'))
  const server = await loopbackManifest((req, res) => {
    const file = req.url?.endsWith('bubo-release.json') ? join(release, 'bubo-release.json') :
      req.url?.endsWith('.zip') ? join(release, 'bubo-v0.1.1-windows-x64-gfx1151.zip') : null
    if (!file || !existsSync(file)) { res.writeHead(404); res.end(); return }
    const data = readFileSync(file)
    res.writeHead(200, { 'content-length': data.length, 'content-type': file.endsWith('.zip') ? 'application/zip' : 'application/json' })
    res.end(data)
  })
  let child, old, next, run, holdLog
  try {
    execFileSync(go, ['run', './v8/faults/runtimepackage', '-folder', release, '-origin', new URL(server.url).origin,
      '-gui', opts.candidate ?? sourceGUI, '-cli', sourceCLI, '-helper', helper, `-changed=${changed}`],
      { cwd: join(root, 'gui', 'tests'), windowsHide: true, stdio: 'pipe' })
    run = makeRun(label, server, { prepared: true })
    const runtime = join(run.local, 'Bubo', 'runtime', 'old-runtime')
    mkdirSync(runtime, { recursive: true })
    writeFileSync(join(runtime, 'sentinel'), 'old runtime')
    const config = JSON.parse(readFileSync(run.configPath, 'utf8'))
    writeFileSync(run.configPath, JSON.stringify(config, null, 2))
    run.env = cleanTestEnvironment({ ...run.env, BUBO_FAKE_RUNTIME_PATH: runtime,
      BUBO_FAKE_PACKAGE_VERSION_FILE: '1', BUBO_FAKE_PREPARE_COMPONENT: 'runtime' })
    child = await startTestGUI({ exe: join(bin, 'Bubo.Control.Web.exe'), cwd: base, environment: run.env,
      args: [], pidFile: join(base, 'old-gui.pid'), probeDiagnostics: true })
    old = await connect(child, run)
    if (opts.holdOldBrowserMs) {
      const tool = join(folder, 'holdbrowser.exe')
      execFileSync(go, ['build', '-o', tool, './v8/holdbrowser'], { cwd: join(root, 'gui', 'tests'), windowsHide: true, stdio: 'pipe' })
      holdLog = join(base, 'holdbrowser.ndjson')
      const launch = args => spawn(tool, args, { windowsHide: true, stdio: 'ignore', detached: true }).unref()
      // The old GUI's WebView2 browser outlives the GUI by this long (a slow browser
      // shutdown), so the helper starts the new GUI while it still holds the profile.
      if (opts.holdOldBrowserMs) launch([String(child.pid), String(opts.holdOldBrowserMs), holdLog])
    }
    await waitView(old, v => v.initialized)
    // The updater starts after the window's DOM-ready event; under load that is later than GetView.
    // Ask again until it answers, as relaunch() does.
    let checked = await call(old, 'CheckUpdate')
    for (let end = Date.now() + 15000; checked.error?.code === 'updater_starting' && Date.now() < end;) { await sleep(100); checked = await call(old, 'CheckUpdate') }
    if (checked.error) throw new Error(`update check: ${JSON.stringify(checked.error)}`)
    await waitView(old, v => v.update?.state === 'available')
    const download = await call(old, 'DownloadUpdate')
    if (download.error) throw new Error(`update download: ${JSON.stringify(download.error)}`)
    await waitView(old, v => v.update?.state === 'ready')
    const before = await view(old)
    const installPromise = old.page.evaluate(() => globalThis.go.main.App.InstallUpdate()).catch(e => ({ exited: String(e) }))
    await sleep(200)
    trace('InstallUpdate called')
    const handoff = await installPromise
    trace('install handoff returned')
    if (handoff?.error) throw new Error(`install handoff: ${JSON.stringify(handoff.error)}`)
    await old.browser.close().catch(() => {})
    const oldExit = Date.now() + 20000
    while (child.exitCode === null && Date.now() < oldExit) { assertIsolated(child); await sleep(100) }
    if (child.exitCode === null) throw new Error('old GUI did not exit for helper')
    trace('old GUI exit observed')
    next = await connect(child, run)
    const after = await waitView(next, v => v.initialized)
    const activePath = join(base, `.${name}.bubo-update`, 'active.json')
    const resultEnd = Date.now() + 15000
    let result
    while (Date.now() < resultEnd) {
      if (existsSync(activePath)) {
        const active = JSON.parse(readFileSync(activePath, 'utf8'))
        const resultPath = join(active.root, '.bubo-update-result.json')
        if (existsSync(resultPath)) { result = JSON.parse(readFileSync(resultPath, 'utf8')); break }
      }
      assertIsolated(child)
      await sleep(100)
    }
    if (result?.status !== expect) throw new Error(`copied-tree helper ended ${result?.status}, expected ${expect}: ${JSON.stringify(result)}`)
    if (expect === 'rolled_back') {
      // restartOld started the old GUI from the install folder; it must be up.
      const restarted = await view(next)
      return { observed: { result, install, restarted_initialized: restarted.initialized, phase: restarted.phase,
        update: restarted.update, exe: join(bin, 'Bubo.Control.Web.exe') }, session: next, run, runtime, base, child, server }
    }
    await next.page.waitForTimeout(3000)
    const final = await view(next)
    if (final.update?.state === 'failed') throw new Error(`copied-tree GUI failed after install: ${JSON.stringify(final.update)}`)
    const screen = await next.page.locator('body').innerText()
    await next.page.evaluate(() => { location.hash = '#/about' })
    await next.page.waitForTimeout(300)
    const aboutScreen = await next.page.locator('body').innerText()
    if (next.errors.length) throw new Error(`page error: ${next.errors[0]}`)
    const members = child.probeWindows.flatMap(x => x.job_processes ?? [])
    const hidden = child.probeWindows.flatMap(x => x.hidden ?? [])
    const restarted = hidden.filter(x => x.class === 'wailsWindow' && x.pid !== child.pid &&
      members.some(m => m.pid === x.pid && m.creation_time === x.job_creation_time))
    const capture = existsSync(run.captureFile) ? readFileSync(run.captureFile, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse) : []
    const invariant = { restarted_job_member: restarted.length > 0, restarted_pid: restarted[0]?.pid,
      hidden: restarted.every(x => x.visible === false),
      no_default_window: true, no_inherited_runtime: !capture.some(x => x.kind === 'inherited_runtime_present' && x.value === 'true'),
      isolated_instance: config.instance_suffix.startsWith(`.Test.${paths.job}.`), isolated_login: config.login_key_path.startsWith(`Software\\BuboTest\\${paths.job}\\`),
      isolated_key: config.key_target.startsWith(`Bubo.Test.${paths.job}.`), capture: capture.map(x => x.kind) }
    if (!invariant.restarted_job_member || !invariant.hidden || !invariant.no_inherited_runtime ||
      !invariant.isolated_instance || !invariant.isolated_login || !invariant.isolated_key) throw new Error(`restart isolation: ${JSON.stringify(invariant)}`)
    const observed = { before: before.update, after: final.update, phase: final.phase, screen, about_screen: aboutScreen,
      approval: final.approval, preparation: final.preparation, start: final.actions?.start,
      runtime_calls: runtimeCalls(run), approval_written: approved(run), old_runtime_exists: existsSync(join(runtime, 'sentinel')),
      hold_ms: opts.holdOldBrowserMs ?? 0, hold_log: holdLog && existsSync(holdLog) ? ndjson(holdLog) : null,
      handoff, result, invariant, install, fake_dir: run.fakeDir, profile: join(run.local, 'Bubo', 'gui', 'webview2') }
    return { observed, session: next, run, runtime, base, child, server }
  } catch (error) {
    const failures = await closeAll([
      ['next browser', () => next?.browser.close()],
      ['old browser', () => old?.browser.close()],
      ['private desktop', () => child && stopTestGUI(child)],
      ['WebView profile', () => run && safeRemove(join(run.local, 'Bubo', 'gui', 'webview2'), paths.evidence)],
      ['loopback server', () => server.close()],
      ['transaction scratch', () => safeRemove(base, paths.evidence)],
    ])
    if (failures.length) console.error(`transaction cleanup failures: ${failures.join('; ')}`)
    throw error
  }
}

async function spacedCases() {
  const checks = (tx, label) => ({ install: tx.observed.install, result: tx.observed.result?.status ?? null, label })
  if (selected.has('P1')) {
    const tx = await transact('sp1', false, { name: spacedName })
    try {
      const o = tx.observed
      // The health start of the new GUI ran with the install folder as its
      // working directory (helper.go startDetached) and acknowledged health.
      record('P1', o.result?.status === 'installed' && o.invariant.restarted_job_member && o.invariant.hidden &&
        o.after?.state !== 'failed' && o.install.includes(spacedName), { ...checks(tx, 'health start'), after: o.after, phase: o.phase,
          restarted_pid: o.invariant.restarted_pid })
    } finally { await finish(tx) }
  }
  if (selected.has('P2')) {
    const candidate = join(folder, 'failhealth.exe')
    execFileSync(go, ['build', '-ldflags=-H=windowsgui', '-o', candidate, './v8/faults/failhealth'],
      { cwd: join(root, 'gui', 'tests'), windowsHide: true, stdio: 'pipe' })
    const tx = await transact('sp2', false, { name: spacedName, candidate, expect: 'rolled_back' })
    try {
      const o = tx.observed
      record('P2', o.result?.status === 'rolled_back' && o.restarted_initialized === true && o.install.includes(spacedName),
        { ...checks(tx, 'restartOld'), initialized: o.restarted_initialized, phase: o.phase, failure: o.result?.failure ?? null })
    } finally { await finish(tx) }
  }
  if (selected.has('P3')) {
    // The old GUI's WebView2 browser outlives the GUI by hold_ms (a slow shutdown).
    // The helper starts the new GUI at once, while that browser still holds the
    // shared WebView2 profile folder and the debugging port.
    const hold = Number(process.env.RUNTIME_HOLD_MS || 5000)
    const tx = await transact('sp3', false, { name: spacedName, holdOldBrowserMs: hold })
    try {
      const o = tx.observed
      const held = o.hold_log?.some(x => x.ev === 'suspended') && o.hold_log?.some(x => x.ev === 'resumed')
      record('P3', o.result?.status === 'installed' && held === true && o.after?.state !== 'failed',
        { ...checks(tx, 'old browser held ' + hold + ' ms'), held, hold_log: o.hold_log, after: o.after, phase: o.phase })
    } finally { await finish(tx) }
  }
}

async function finish(tx) {
  const failures = await closeAll([
    ['browser', () => tx.session.browser.close()],
    ['private desktop', () => stopTestGUI(tx.child)],
    ['WebView profile', () => safeRemove(join(tx.run.local, 'Bubo', 'gui', 'webview2'), paths.evidence)],
    ['loopback server', () => tx.server.close()],
    ['transaction scratch', () => safeRemove(tx.base, paths.evidence)],
  ])
  if (failures.length) throw new Error(`transaction cleanup failed: ${failures.join('; ')}`)
}

async function closeGUI(session, child, pid) {
  await call(session, 'Exit').catch(() => {})
  const end = Date.now() + 10000
  while (Date.now() < end && !readExits(child).some(x => x.pid === pid)) {
    assertIsolated(child)
    await sleep(100)
  }
  if (!readExits(child).some(x => x.pid === pid)) throw new Error(`GUI ${pid} did not exit after Exit`)
  await session.browser.close().catch(() => {})
  await stopTestGUI(child)
}

async function relaunch(tx, label) {
  const child = await startTestGUI({ exe: join(tx.base, 'Bubo', 'bin', 'Bubo.Control.Web.exe'), cwd: tx.base,
    environment: tx.run.env, pidFile: join(tx.base, `${label}.pid`), probeDiagnostics: true })
  try {
    const session = await connect(child, tx.run)
    await waitView(session, v => v.initialized)
    // CheckUpdate only succeeds after UpdateService has read LastResult at
    // startup. This makes a missing notice on the second relaunch meaningful.
    const end = Date.now() + 10000
    let checked
    while (Date.now() < end) {
      checked = await call(session, 'CheckUpdate')
      if (!checked.error) break
      // The not-yet-started updater answers updater_starting (B1b/B2 code;
      // unsupported before them). Anything else is a real failure.
      if (!['updater_starting', 'unsupported'].includes(checked.error.code)) throw new Error(`relaunch update check: ${JSON.stringify(checked.error)}`)
      await sleep(100)
    }
    if (checked?.error) throw new Error(`relaunch updater did not initialize: ${JSON.stringify(checked.error)}`)
    const snapshot = await view(session)
    return { child, session, snapshot }
  } catch (error) {
    await stopTestGUI(child)
    throw error
  }
}

try {
  if (['P1', 'P2', 'P3'].some(x => selected.has(x))) await spacedCases()
  if (['U1', 'U2', 'U3', 'U5'].some(x => selected.has(x))) {
    const tx = await transact('changed', true)
    try {
      const o = tx.observed
      const offered = Boolean(o.approval) && Array.isArray(o.preparation) && o.preparation.some(x => x.kind === 'runtime')
      if (selected.has('U1')) record('U1', offered && /runtime/i.test(`${o.screen}\n${o.about_screen}`), o)
      if (selected.has('U2')) record('U2', !o.approval_written, { offered, approval_written: o.approval_written, runtime_calls: o.runtime_calls })
      if (selected.has('U3')) record('U3', o.old_runtime_exists, { offered, old_runtime_exists: o.old_runtime_exists, runtime: tx.runtime })
      if (selected.has('U5')) {
        let declined = null, after = null
        if (offered) {
          declined = await call(tx.session, 'Decline')
          after = await view(tx.session)
        }
        await tx.session.page.evaluate(() => { location.hash = '#/' })
        await tx.session.page.waitForTimeout(300)
        const declineScreen = await tx.session.page.locator('main').innerText()
        record('U5', !offered || (!declined?.error && after?.update?.current_version === '0.1.1' &&
          after.actions?.start?.allowed === false && /prepare|engine/i.test(after.actions.start.reason ?? '') &&
          /prepare|engine/i.test(declineScreen)),
          { offered, declined, state: after?.state, update: after?.update, start: after?.actions?.start,
            screen: declineScreen })
      }
    } finally { await finish(tx) }
  }
  if (selected.has('U4') || selected.has('F19b')) {
    const tx = await transact('unchanged', false)
    const extra = []
    try {
      if (selected.has('U4')) {
        const calls = runtimeCalls(tx.run).filter(x => x.Args?.[0] === 'prepare')
        record('U4', calls.length === 0 && !tx.observed.approval, { prepare_calls: calls, approval: tx.observed.approval,
          update: tx.observed.after, invariant: tx.observed.invariant })
      }
      if (selected.has('F19b')) {
        await closeGUI(tx.session, tx.child, tx.observed.invariant.restarted_pid)
        const first = await relaunch(tx, 'result-first')
        extra.push(first)
        const notice = first.snapshot.update?.last_result_notice
        const dismissed = await call(first.session, 'DismissUpdate')
        const afterDismiss = await view(first.session)
        await closeGUI(first.session, first.child, first.child.pid)
        const second = await relaunch(tx, 'result-second')
        extra.push(second)
        const afterRelaunch = second.snapshot
        record('F19b', Boolean(notice?.startsWith('Updated to 0.1.1.')) && !dismissed.error &&
          !afterDismiss.update?.last_result_notice && !afterRelaunch.update?.last_result_notice,
          { helper_result: tx.observed.result, first_notice: notice, dismissed,
            after_dismiss: afterDismiss.update, after_relaunch: afterRelaunch.update,
            restarted_job_member: tx.observed.invariant.restarted_job_member })
      }
    } finally {
      for (const x of extra) {
        await x.session.browser.close().catch(() => {})
        await stopTestGUI(x.child)
      }
      await finish(tx)
    }
  }
} catch (error) {
  writeFileSync(join(folder, 'fatal.json'), JSON.stringify({ error: String(error), stack: error.stack }, null, 2))
  console.error(`RUNTIME UPDATE STOP: ${error.stack ?? error}`)
  process.exitCode = 2
} finally {
  const red = results.filter(x => !x.good)
  writeFileSync(join(folder, 'summary.json'), JSON.stringify({ total: results.length, red: red.length, results }, null, 2))
  console.log(`U gate: ${results.length - red.length}/${results.length} passed, ${red.length} RED`)
  if (red.length) process.exitCode = 1
}
