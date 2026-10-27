import { createHash, randomUUID } from 'node:crypto'
import { createRequire } from 'node:module'
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { createServer } from 'node:https'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { cleanTestEnvironment, defaultDesktopViolations, readExits, readProbeWindows, startTestGUI, stopTestGUI } from './launcher.mjs'
import { v8Config } from './config.mjs'
import { leaveState, reachState } from './walk/state.mjs'

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const frontend = join(root, 'gui', 'wails', 'frontend')
const { chromium } = createRequire(join(frontend, 'package.json'))('playwright')
const { job: jobID, port, evidence, jobFolder: job, shots, tree, cdpURL } = v8Config()
const helper = join(evidence, 'build', 'winhelper.exe')
const exe = join(tree, 'Bubo.Control.Web.exe')
const brokenCDPExe = join(tree, 'Bubo.Control.Web.no-cdp.exe')
const prodExe = join(evidence, 'build', 'Bubo.Control.Web.production.exe')
const fakeExe = join(tree, 'bin', 'bubo.exe')
const go = (process.env.BUBO_GO ?? 'go')
const openssl = 'C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe'
const scenario = join(root, 'gui', 'tests', 'scenarios', 'recordings', 'not-prepared')
const fixture = join(root, 'tests', 'cli', 'contract')
const results = {}
const consoleLines = []
const snapshots = []
const captures = []
const readOnlyChecks = []
const processes = new Set()
const observedJobMembers = new Map()
const hiddenOutsideJob = new Map()
const runID = randomUUID()

function hash(file) { return createHash('sha256').update(readFileSync(file)).digest('hex') }
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)) }
function record(name, passed, detail, breakFailed) {
  results[name] = { passed: Boolean(passed), break_failed: breakFailed === undefined ? null : Boolean(breakFailed), detail }
  console.log(`${name}: ${passed ? 'PASS' : 'FAIL'}${breakFailed === undefined ? '' : `, break ${breakFailed ? 'failed as expected' : 'did not fail'}`}`)
}
async function helperCommand(config, localAppData, args) {
  const output = join(evidence, 'helper', `${randomUUID()}.txt`)
  mkdirSync(join(evidence, 'helper'), { recursive: true })
  const env = cleanTestEnvironment({ BUBO_TEST_CONFIG: config, LOCALAPPDATA: localAppData, BUBO_TEST_HELPER_OUTPUT: output })
  const child = await startTestGUI({ exe: helper, cwd: tree, args, environment: env, pidFile: join(evidence, 'helper', `${randomUUID()}.pid`), probeDiagnostics: true })
  processes.add(child)
  try {
    const end = Date.now() + 10000
    while (child.exitCode === null && Date.now() < end) {
      assertNoWindow(child)
      await sleep(50)
    }
    if (child.exitCode !== 0) throw new Error(`hidden helper ${args.join(' ')} exit ${child.exitCode}`)
    assertNoWindow(child)
    return existsSync(output) ? readFileSync(output, 'utf8').trim() : ''
  } finally { await stopTestGUI(child); processes.delete(child) }
}
async function observe(config, localAppData) {
  return JSON.parse(await helperCommand(config, localAppData, ['observe']))
}
function assertNoWindow(child) {
  for (const s of readProbeWindows(child)) {
    const members = new Map((s.job_processes ?? []).map(member => [member.pid, member.creation_time]))
    for (const member of s.job_processes ?? []) observedJobMembers.set(`${member.pid}/${member.creation_time}`, member)
    for (const w of s.hidden ?? []) {
      if (members.get(w.pid) !== w.job_creation_time) hiddenOutsideJob.set(`${child.creationTime}/${w.handle}/${w.pid}`, w)
    }
  }
  const diff = defaultDesktopViolations(child)
  snapshots.push({ pid: child.pid, desktop: child.desktop, window_changes: readProbeWindows(child).length, default_desktop_violations: diff })
  if (diff.length) { child.kill(); throw new Error(`default-desktop test window: ${JSON.stringify(diff)}`) }
  if (hiddenOutsideJob.size) { child.kill(); throw new Error(`window on hidden desktop outside current job membership: ${JSON.stringify([...hiddenOutsideJob.values()])}`) }
  return true
}
function hiddenMainVisible(child) {
  return readProbeWindows(child).some(s => (s.hidden ?? []).some(w =>
    w.pid === child.pid && w.job_creation_time === child.creationTime && w.class === 'wailsWindow' && w.visible))
}
function hiddenNativePopup(child) {
  return readProbeWindows(child).some(s => (s.hidden ?? []).some(w =>
    w.job_creation_time && s.job_processes?.some(member => member.pid === w.pid && member.creation_time === w.job_creation_time) &&
    w.visible && ['#32770', 'CabinetWClass'].includes(w.class)))
}
function configFor(label, manifestURL, caFile, changes = {}) {
  label = `${runID}-${label}`
  const localAppData = join(evidence, 'local', label)
  const fakeDir = join(evidence, 'fake', label)
  const startup = join(evidence, 'startup', label)
  const captureFile = join(evidence, 'capture', `${label}.jsonl`)
  const dialogFolder = join(evidence, 'dialogs', label)
  const dialogFile = join(dialogFolder, 'selected.gguf')
  for (const dir of [localAppData, fakeDir, startup, join(evidence, 'capture'), dialogFolder, join(evidence, 'config')]) mkdirSync(dir, { recursive: true })
  if (!existsSync(dialogFile)) writeFileSync(dialogFile, '')
  const config = {
    evidence_root: evidence,
    instance_suffix: `.Test.${jobID}.${label}`,
    login_key_path: `Software\\BuboTest\\${jobID}\\${label}`,
    startup_folder: startup,
    key_target: `Bubo.Test.${jobID}.${label}`,
    capture_file: captureFile,
    dialog_file: dialogFile,
    dialog_folder: dialogFolder,
    manifest_url: manifestURL,
    root_ca_file: caFile,
    ...changes,
  }
  const configPath = join(evidence, 'config', `${label}.json`)
  writeFileSync(configPath, JSON.stringify(config, null, 2))
  return { label, config, configPath, localAppData, fakeDir, captureFile }
}
function envFor(run, extra = {}) {
  return cleanTestEnvironment({
    LOCALAPPDATA: run.localAppData,
    BUBO_TEST_CONFIG: run.configPath,
    BUBO_FAKE_DIR: run.fakeDir,
    BUBO_FAKE_SCENARIO: scenario,
    BUBO_FAKE_MODE: 'ui',
    BUBO_FAKE_FIXTURES: fixture,
    BUBO_FAKE_RECORD_ALL_ENV: '1',
    BUBO_FAKE_RECORD_RUNTIME: '1',
    ...extra,
  })
}
async function connect(deadlineMs = 12000, url = cdpURL) {
  const end = Date.now() + deadlineMs
  let last
  while (Date.now() < end) {
    try {
      const browser = await chromium.connectOverCDP(url, { timeout: 1000 })
      const page = browser.contexts()[0]?.pages()[0]
      if (page) return { browser, page }
      await browser.close()
    } catch (error) { last = error }
    await sleep(150)
  }
  throw new Error(`CDP unavailable: ${last}`)
}
async function launch(run, options = {}) {
  const args = options.restart ? ['--bubo-update-health', randomUUID(), '--bubo-update-health-deadline', new Date(Date.now() + 60000).toISOString()] : []
  const child = await startTestGUI({ exe: options.exe ?? exe, cwd: tree, args, environment: envFor(run, options.extra), pidFile: join(evidence, `pid-${run.label}.txt`), onOutput: (stream, data) => consoleLines.push(`${stream}: ${data}`), probeDiagnostics: true })
  processes.add(child)
  assertNoWindow(child)
  let browser, page
  if (options.cdp !== false) {
    ({ browser, page } = await connect())
    await page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: 5000 })
  }
  assertNoWindow(child)
  return { child, browser, page, run }
}
async function stop(session) {
  try { await session.browser?.close() } catch {}
  await stopTestGUI(session.child)
  processes.delete(session.child)
  await sleep(300)
}
async function call(session, name, ...args) {
  const result = await session.page.evaluate(async ({ name, args }) => globalThis.go.main.App[name](...args), { name, args })
  assertNoWindow(session.child)
  return result
}
function capture(run) {
  if (!existsSync(run.captureFile)) return []
  const lines = readFileSync(run.captureFile, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse)
  captures.push({ run: run.configPath, lines })
  return lines
}
function envNames(run) {
  const file = join(run.fakeDir, 'env.json')
  if (!existsSync(file)) return []
  return Object.keys(JSON.parse(readFileSync(file, 'utf8')).bubo_variables ?? {})
}
function fakeArgs(run) {
  const file = join(run.fakeDir, 'runtime-children.jsonl')
  if (!existsSync(file)) return []
  return readFileSync(file, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line).Args)
}
function jobListeners(child) {
  assertNoWindow(child)
  const members = readProbeWindows(child).at(-1)?.job_processes ?? []
  if (!members.length) throw new Error('I6 has no current Job Object membership snapshot')
  const pids = new Set(members.map(member => String(member.pid)))
  return execFileSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
    .split(/\r?\n/).map(line => line.trim().split(/\s+/))
    .filter(parts => parts[0] === 'TCP' && parts[3] === 'LISTENING' && pids.has(parts[4]))
    .map(parts => ({ address: parts[1], pid: Number(parts[4]) }))
}
function loopbackListener(address) {
  const host = address.replace(/:\d+$/, '').replace(/^\[|\]$/g, '')
  return host === '::1' || host === '127.0.0.1' || host.startsWith('127.')
}
async function launchHolder(run) {
  const output = join(evidence, 'helper', `${randomUUID()}-holder.txt`)
  const child = await startTestGUI({ exe: helper, cwd: tree, args: ['holder'], environment: envFor(run, { BUBO_TEST_HELPER_OUTPUT: output }), pidFile: join(evidence, 'helper', `${randomUUID()}.pid`), probeDiagnostics: true })
  processes.add(child)
  const end = Date.now() + 5000
  while ((!existsSync(output) || !readFileSync(output, 'utf8').includes('READY')) && Date.now() < end) { assertNoWindow(child); await sleep(50) }
  if (!existsSync(output) || !readFileSync(output, 'utf8').includes('READY')) throw new Error(`decoy instance holder did not start: exit=${child.exitCode}`)
  return { child, get output() { return readFileSync(output, 'utf8') } }
}
async function waitCDPDown() {
  const end = Date.now() + 5000
  while (Date.now() < end) {
    try {
      const browser = await chromium.connectOverCDP(cdpURL, { timeout: 500 })
      await browser.close()
    } catch { return }
    await sleep(150)
  }
  throw new Error('old CDP endpoint stayed up after exact test PID stopped')
}
async function startServer(cert, key) {
  const requests = []
  const server = createServer({ cert: readFileSync(cert), key: readFileSync(key) }, (req, res) => {
    requests.push({ url: req.url, method: req.method })
    res.writeHead(404, { 'content-type': 'text/plain' })
    res.end('no update in probe')
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  if (port >= 8081 && port <= 8086) { server.close(); return startServer(cert, key) }
  return { server, requests, url: `https://127.0.0.1:${port}/haloforge-labs/bubo/releases/latest/download/bubo-release.json` }
}
function makeCert() {
  const cert = join(evidence, 'build', 'loopback.crt')
  const key = join(evidence, 'build', 'loopback.key')
  if (!existsSync(cert) || !existsSync(key)) throw new Error('loopback certificate from hidden GUI smoke is missing')
  return { cert, key }
}

async function main() {
  if (!existsSync(brokenCDPExe)) throw new Error('Build I10 negative control with node gui/tests/v8/negative-cdp.mjs first')
  mkdirSync(evidence, { recursive: true })
  mkdirSync(shots, { recursive: true })
  const { cert, key } = makeCert()
  const mainServer = await startServer(cert, key)
  const decoyServer = await startServer(cert, key)
  const safe = configFor('safe', mainServer.url, cert)
  const decoy = configFor('decoy', mainServer.url, cert)
  const ownerBefore = await observe(safe.configPath, safe.localAppData)
  const decoyStartup = decoy.config.startup_folder
  const sentinelLink = join(decoyStartup, 'Bubo.Control.lnk')
  writeFileSync(sentinelLink, 'A0 decoy sentinel')
  await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'seed'])
  await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'seed'])
  let session, holder
  try {
    holder = await launchHolder(decoy)
    session = await launch(safe)
    const view = await call(session, 'GetView')
    record('drive', Boolean(view), 'CDP GetView returned a View')
    record('I1', Boolean(view) && !holder.output.includes('ACTIVATED'), 'safe instance started while decoy lease stayed unsignaled')
    const shot = join(shots, 'hidden-first-view.png')
    await session.page.screenshot({ path: shot })
    const before = await observe(safe.configPath, safe.localAppData)
    const copied = await call(session, 'CopyText', 'A0 clipboard capture')
    const after = await observe(safe.configPath, safe.localAppData)
    readOnlyChecks.push({ case: 'I4', before: { clipboard_sequence: before.clipboard_sequence, owner_run_present: before.owner_run_present, owner_key_present: before.owner_key_present }, after: { clipboard_sequence: after.clipboard_sequence, owner_run_present: after.owner_run_present, owner_key_present: after.owner_key_present } })
    record('I4', !copied.error && capture(safe).some(e => e.kind === 'copy' && e.value === 'A0 clipboard capture') && before.clipboard_sequence === after.clipboard_sequence, { before: before.clipboard_sequence, after: after.clipboard_sequence })
    const opened = await call(session, 'OpenExternal', 'https://haloforgelabs.com/docs/')
    const folder = await call(session, 'BrowseModelFolder')
    await call(session, 'BrowseModelFile')
    if (folder.folder) await call(session, 'OpenModelFolder', folder.folder)
    await call(session, 'SetVisible', false)
    await call(session, 'SetVisible', true)
    const native = capture(safe)
    const nativeKinds = new Set(native.map(e => e.kind))
    record('I5', !opened.error && ['tray_start', 'open_external', 'dialog_folder', 'dialog_file', 'open_folder', 'hide', 'show'].every(kind => nativeKinds.has(kind)) && !hiddenNativePopup(session.child) && !defaultDesktopViolations(session.child).length, { capture: native })
    const names = envNames(safe)
    record('I7', names.length > 0 && !names.includes('BUBO_RUNTIME_DIR') && capture(safe).some(e => e.kind === 'inherited_runtime_present' && e.value === 'false'), { names })
    const webviewInJob = [...observedJobMembers.values()].filter(member => member.image.toLowerCase() === 'msedgewebview2.exe')
    record('I8', webviewInJob.length > 0 && !hiddenMainVisible(session.child) && !defaultDesktopViolations(session.child).length,
      { webview_job_members: webviewInJob, hidden_wails_window: !hiddenMainVisible(session.child), default_desktop_test_windows: defaultDesktopViolations(session.child).length })
    const reset = await call(session, 'ResetKey')
    record('I3', !reset.error && await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'check']) === 'true', { reset })
    const loginOn = await call(session, 'PatchPreferences', { set: { start_gui_at_login: true }, clear: [] })
    const ownOn = await helperCommand(safe.configPath, safe.localAppData, ['registry', 'present']) === 'true'
    const loginOff = await call(session, 'PatchPreferences', { set: { start_gui_at_login: false }, clear: [] })
    const ownOff = await helperCommand(safe.configPath, safe.localAppData, ['registry', 'present']) === 'false'
    record('I2', !loginOn.error && !loginOff.error && ownOn && ownOff && await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'check']) === 'true' && readFileSync(sentinelLink, 'utf8') === 'A0 decoy sentinel', { ownOn, ownOff, decoy: 'unchanged' })
    await call(session, 'CheckUpdate')
    record('I11', mainServer.requests.length > 0 && decoyServer.requests.length === 0, { main_count: mainServer.requests.length, decoy_count: decoyServer.requests.length })
    await stop(session); session = undefined
    await waitCDPDown()

    const lanManifest = await startServer(cert, key)
    const lanPort = 19437 // probe-owned loopback fixture, not the CLI default
    let lanState
    try {
      lanState = await reachState('S12', { ...lanManifest, ca: cert }, { env: { BUBO_FAKE_HTTP: 'ready' }, settings: { port: lanPort }, probeDiagnostics: true })
      const lan = await call(lanState.session, 'PatchPreferences', { set: { lan_access: true }, clear: [] })
      const restarted = await call(lanState.session, 'Restart')
      const deadline = Date.now() + 12000
      let instance
      while (Date.now() < deadline) {
        const file = join(lanState.run.fakeDir, 'instance.json')
        if (existsSync(file)) instance = JSON.parse(readFileSync(file, 'utf8'))
        if (instance?.phase === 'ready' && instance.endpoint?.startsWith('http://0.0.0.0:')) break
        await sleep(100)
      }
      const listeners = jobListeners(lanState.session.child)
      const hostPassed = fakeArgs(lanState.run).some(args => args.some((arg, index) => arg === '--host' && args[index + 1] === '0.0.0.0'))
      const fakeListener = listeners.some(item => item.pid === instance?.pid && item.address === `127.0.0.1:${lanPort}`)
      record('I6', !lan.error && !restarted.error && instance?.phase === 'ready' && hostPassed && fakeListener && listeners.every(item => loopbackListener(item.address)),
        { lan_error: lan.error?.code, restart_error: restarted.error?.code, fake_phase: instance?.phase, host_passed: hostPassed, fake_listener: fakeListener, listeners })
    } finally { if (lanState) await leaveState(lanState); await new Promise(resolve => lanManifest.server.close(resolve)) }
    await waitCDPDown()

    const brokenInstance = configFor('broken-instance', mainServer.url, cert, { instance_suffix: decoy.config.instance_suffix })
    let broken = await launch(brokenInstance, { cdp: false })
    await sleep(300)
    results.I1.break_failed = holder.output.includes('ACTIVATED') && broken.child.exitCode !== null
    await stop(broken)
    await stopTestGUI(holder.child); processes.delete(holder.child); holder = undefined

    const brokenLogin = configFor('broken-login', mainServer.url, cert, { login_key_path: decoy.config.login_key_path, startup_folder: decoyStartup })
    broken = await launch(brokenLogin)
    results.I2.break_failed = await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'check']) !== 'true' || !existsSync(sentinelLink) || readFileSync(sentinelLink, 'utf8') !== 'A0 decoy sentinel'
    await stop(broken); await waitCDPDown()
    await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'seed'])
    writeFileSync(sentinelLink, 'A0 decoy sentinel')

    const brokenKey = configFor('broken-key', mainServer.url, cert, { key_target: decoy.config.key_target })
    broken = await launch(brokenKey)
    await call(broken, 'ResetKey')
    results.I3.break_failed = await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'check']) !== 'true'
    await stop(broken); await waitCDPDown()
    await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'seed'])

    const inherited = configFor('inherited-env', mainServer.url, cert)
    const decoyRuntime = join(evidence, 'decoy-runtime')
    mkdirSync(decoyRuntime, { recursive: true })
    broken = await launch(inherited, { extra: { BUBO_RUNTIME_DIR: decoyRuntime } })
    results.I7.break_failed = capture(inherited).some(e => e.kind === 'inherited_runtime_present' && e.value === 'true')
    results.I7.detail.break_fake_cli_still_scrubbed = !envNames(inherited).includes('BUBO_RUNTIME_DIR')
    await stop(broken); await waitCDPDown()

    holder = await launchHolder(decoy)

    // The health flags are the only arguments the updater's helper passes.
    const restarted = await launch(safe, { restart: true })
    const restartView = await call(restarted, 'GetView')
    record('I10', Boolean(restartView), 'CDP reattached after health-flag restart')
    const restartNames = envNames(safe)
    record('I9', Boolean(restartView) && !holder.output.includes('ACTIVATED') && await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'check']) === 'true' && await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'check']) === 'true' && !restartNames.includes('BUBO_RUNTIME_DIR') && !hiddenMainVisible(restarted.child) && !defaultDesktopViolations(restarted.child).length, 'I1-I3, I7, I8 after restart')
    await stop(restarted)
    await waitCDPDown()
    const brokenRestart = configFor('broken-restart', mainServer.url, cert, { instance_suffix: decoy.config.instance_suffix })
    broken = await launch(brokenRestart, { restart: true, cdp: false })
    const brokenDeadline = Date.now() + 5000
    while (broken.child.exitCode === null && Date.now() < brokenDeadline) { assertNoWindow(broken.child); await sleep(50) }
    results.I9.break_failed = holder.output.includes('ACTIVATED') && broken.child.exitCode === 0
    await stop(broken)
    await stopTestGUI(holder.child); processes.delete(holder.child); holder = undefined

    const missingCDP = configFor('missing-cdp', mainServer.url, cert)
    broken = await launch(missingCDP, { restart: true, cdp: false, exe: brokenCDPExe })
    const cdpDeadline = Date.now() + 8000
    while (!capture(missingCDP).some(e => e.kind === 'tray_start') && Date.now() < cdpDeadline) {
      assertNoWindow(broken.child)
      if (broken.child.exitCode !== null) throw new Error(`I10 negative GUI exited 0x${broken.child.exitCode.toString(16)}`)
      await sleep(50)
    }
    const negativeStarted = capture(missingCDP).some(e => e.kind === 'tray_start') && broken.child.exitCode === null
    let missingPortAttached = false
    try { const missing = await connect(1800); missingPortAttached = true; await missing.browser.close() } catch {}
    results.I10.break_failed = negativeStarted && !missingPortAttached && !defaultDesktopViolations(broken.child).length
    results.I10.detail = { restarted_get_view: Boolean(restartView), negative_gui_started: negativeStarted, negative_exe: brokenCDPExe, fixed_port_attached: missingPortAttached }
    await stop(broken)

    const wrongManifest = configFor('wrong-manifest', decoyServer.url, cert)
    const mainCount = mainServer.requests.length
    const decoyCount = decoyServer.requests.length
    broken = await launch(wrongManifest)
    const check = await call(broken, 'CheckUpdate')
    results.I11.break_failed = mainServer.requests.length === mainCount && decoyServer.requests.length > decoyCount
    results.I11.detail.break = { main_before: mainCount, main_after: mainServer.requests.length,
      decoy_before: decoyCount, decoy_after: decoyServer.requests.length, code: check.error?.code }
    await stop(broken); await waitCDPDown()

    const prodSymbols = execFileSync(go, ['tool', 'nm', prodExe], { encoding: 'utf8', windowsHide: true, maxBuffer: 64 * 1024 * 1024 })
    const testSymbols = execFileSync(go, ['tool', 'nm', exe], { encoding: 'utf8', windowsHide: true, maxBuffer: 64 * 1024 * 1024 })
    const prodFiles = execFileSync(go, ['list', '-tags=production,wv2runtime.error', '-f', '{{join .GoFiles ","}}', '.'], { cwd: join(root, 'gui', 'wails'), encoding: 'utf8', windowsHide: true })
    const prodWails = JSON.parse(execFileSync(go, ['list', '-m', '-json', 'github.com/wailsapp/wails/v2'], { cwd: join(root, 'gui', 'wails'), encoding: 'utf8', windowsHide: true }))
    const markers = ['supervisor.TestConfiguration', 'main.captureNative']
    const symbolCheck = symbols => markers.every(marker => !symbols.includes(marker))
    record('I12', symbolCheck(prodSymbols) && !prodFiles.includes('bubo_test') && !prodWails.Replace && prodWails.Version === 'v2.16.0', { markers, production: prodExe, production_go_files: prodFiles.trim(), wails_module: prodWails.Dir }, !symbolCheck(testSymbols))
  } finally {
    if (session) {
      await stop(session)
      session = undefined
    }
    for (const child of [...processes]) {
      await stopTestGUI(child)
      processes.delete(child)
    }
    await helperCommand(decoy.configPath, decoy.localAppData, ['credential', 'clean'])
    await helperCommand(decoy.configPath, decoy.localAppData, ['registry', 'clean'])
    const ownerAfter = await observe(safe.configPath, safe.localAppData)
    readOnlyChecks.push({ case: 'I2/I3', before: { owner_run_present: ownerBefore.owner_run_present, owner_key_present: ownerBefore.owner_key_present }, after: { owner_run_present: ownerAfter.owner_run_present, owner_key_present: ownerAfter.owner_key_present } })
    if (results.I2) results.I2.passed &&= ownerBefore.owner_run_present === ownerAfter.owner_run_present
    if (results.I3) results.I3.passed &&= ownerBefore.owner_key_present === ownerAfter.owner_key_present
    mainServer.server.close()
    decoyServer.server.close()
    const inputHashes = { exe: hash(exe), fake_cli: hash(fakeExe), scenario: hash(join(scenario, 'scenario.json')) }
    const params = `--job ${jobID} --port ${port}`
    const rerun = `node ${join(root, 'gui', 'tests', 'v8', 'build.mjs')} ${params} --skip-frontend\nnode ${join(root, 'gui', 'tests', 'v8', 'negative-cdp.mjs')} ${params}\nnode ${join(root, 'gui', 'tests', 'v8', 'probe.mjs')} ${params}`
    writeFileSync(join(job, 'PROBE-SUMMARY.json'), JSON.stringify({
      results,
      input_hashes: inputHashes,
      rerun_command: rerun,
    }, null, 2))
    writeFileSync(join(evidence, 'console.log'), consoleLines.join('\n'))
    writeFileSync(join(evidence, 'window-diff.json'), JSON.stringify(snapshots, null, 2))
    writeFileSync(join(evidence, 'capture-summary.json'), JSON.stringify(captures, null, 2))
    writeFileSync(join(job, 'PROBE-CONSOLE.log'), consoleLines.join('\n'))
    writeFileSync(join(job, 'PROBE-WINDOW-DIFF.json'), JSON.stringify(snapshots, null, 2))
    writeFileSync(join(job, 'PROBE-CAPTURE.json'), JSON.stringify(captures, null, 2))
    writeFileSync(join(job, 'PROBE-READONLY.json'), JSON.stringify(readOnlyChecks, null, 2))
    writeFileSync(join(job, 'PROBE-INPUT-HASHES.json'), JSON.stringify(inputHashes, null, 2))
    writeFileSync(join(job, 'PROBE-RERUN.txt'), rerun + '\n')
    writeFileSync(join(job, 'PROBE-JOB-MEMBERSHIP.json'), JSON.stringify({ members: [...observedJobMembers.values()], hidden_outside_job: [...hiddenOutsideJob.values()] }, null, 2))
  }
  const positiveOnly = new Set(['I4', 'I5', 'I6', 'I8'])
  const failed = Array.from({ length: 12 }, (_, i) => `I${i + 1}`).filter(name => !results[name]?.passed || (!positiveOnly.has(name) && results[name]?.break_failed !== true))
  if (failed.length) throw new Error(`probe cases incomplete or failed: ${failed.join(', ')}`)
}

main().catch(error => { console.error(error); process.exitCode = 1 })
