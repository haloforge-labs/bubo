import { randomUUID } from 'node:crypto'
import { createRequire } from 'node:module'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:https'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { attachToPage } from '../cdp.mjs'
import { cleanTestEnvironment, readSnapshots, startTestGUI, stopTestGUI } from '../launcher.mjs'
import { v8Config } from '../config.mjs'

export const paths = v8Config()
const runnerErrors = []
for (const kind of ['uncaughtException', 'unhandledRejection']) {
  process.on(kind, error => {
    const message = `${kind}: ${error?.stack ?? error}`
    runnerErrors.push(message)
    console.error(message)
    process.exitCode = 1
  })
}
const root = resolve(fileURLToPath(new URL('../../../../', import.meta.url)))
const frontend = join(root, 'gui', 'wails', 'frontend')
const { chromium } = createRequire(join(frontend, 'package.json'))('playwright')
const cert = join(paths.build, 'loopback.crt')
const key = join(paths.build, 'loopback.key')

export function scenario(name) { return join(root, 'gui', 'tests', 'scenarios', 'recordings', name) }
export function hashInputFiles() { return [join(paths.tree, 'Bubo.Control.Web.exe'), join(paths.tree, 'bin', 'bubo.exe')] }
export function readCapture(run) {
  if (!existsSync(run.captureFile)) return []
  return readFileSync(run.captureFile, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse)
}

export async function loopbackManifest(handler = (_req, res) => { res.writeHead(404); res.end('no update fixture') }) {
  if (!existsSync(cert) || !existsSync(key)) throw new Error('run hidden-gui-probe first to create the A3 loopback certificate')
  const server = createServer({ cert: readFileSync(cert), key: readFileSync(key) }, handler)
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  if (port >= 8081 && port <= 8086) {
    await new Promise(resolve => server.close(resolve))
    return loopbackManifest(handler)
  }
  return { server, url: `https://127.0.0.1:${port}/haloforge-labs/bubo/releases/latest/download/bubo-release.json`, ca: cert,
    close: () => {
      const closed = new Promise(resolve => server.close(resolve))
      server.closeAllConnections()
      return closed
    } }
}

export function makeRun(label, manifest, options = {}) {
  const id = `${label}-${randomUUID()}`
  const local = join(paths.evidence, 'walk', 'local', id)
  const fakeDir = join(paths.evidence, 'walk', 'fake', id)
  const startup = join(paths.evidence, 'walk', 'startup', id)
  const dialogs = join(paths.evidence, 'walk', 'dialogs', id)
  const captureFile = join(paths.evidence, 'walk', 'capture', `${id}.jsonl`)
  const configPath = join(paths.evidence, 'walk', 'config', `${id}.json`)
  for (const dir of [local, fakeDir, startup, dialogs, join(paths.evidence, 'walk', 'capture'), join(paths.evidence, 'walk', 'config')]) mkdirSync(dir, { recursive: true })
  if (options.prepared) writeFileSync(join(fakeDir, 'prepared'), 'true')
  if (options.watchStall) writeFileSync(join(fakeDir, 'watch-stall'), '')
  const settingsDir = join(local, 'Bubo', 'gui')
  mkdirSync(settingsDir, { recursive: true })
  writeFileSync(join(settingsDir, 'settings.json'), JSON.stringify({ version: 1, stall_seconds: options.stallSeconds ?? 120,
    check_for_updates: options.autoUpdate ?? false, ...options.settings }))
  const dialogFile = join(dialogs, 'selected.gguf')
  writeFileSync(dialogFile, '')
  const config = {
    evidence_root: paths.evidence,
    instance_suffix: `.Test.${paths.job}.${id}`,
    login_key_path: `Software\\BuboTest\\${paths.job}\\${id}`,
    startup_folder: startup,
    key_target: `Bubo.Test.${paths.job}.${id}`,
    capture_file: captureFile,
    dialog_file: dialogFile,
    dialog_folder: dialogs,
    manifest_url: manifest.url,
    root_ca_file: manifest.ca,
  }
  writeFileSync(configPath, JSON.stringify(config, null, 2))
  const env = cleanTestEnvironment({
    LOCALAPPDATA: local,
    BUBO_TEST_CONFIG: configPath,
    BUBO_FAKE_DIR: fakeDir,
    BUBO_FAKE_FIXTURES: join(root, 'tests', 'cli', 'contract'),
    BUBO_FAKE_RECORD_RUNTIME: '1',
    BUBO_FAKE_MODE: options.mode ?? 'ui',
    ...(options.scenario ? { BUBO_FAKE_SCENARIO: scenario(options.scenario) } : {}),
    ...options.env,
  })
  return { id, local, fakeDir, startup, captureFile, configPath, env }
}

export function assertIsolated(child) {
  const violations = readSnapshots(child)
  if (violations.length) { child.kill(); throw new Error(`desktop isolation violation: ${JSON.stringify(violations)}`) }
  return 0
}

export async function checkWalkError(session) {
  const boundary = session.page.isClosed() ? 0 : await session.page.locator('#root strong').filter({ hasText: 'Something went wrong!' }).count()
  if (boundary) session.fatalErrors.push({ kind: 'react_boundary', message: 'Router or root error boundary rendered' })
  if (runnerErrors.length) session.fatalErrors.push({ kind: 'runner', message: runnerErrors.shift() })
  if (!session.fatalErrors.length) return
  const first = session.fatalErrors[0]
  if (!session.failureFolder) {
    const folder = join(paths.evidence, 'walk', 'failures', session.run.id)
    mkdirSync(folder, { recursive: true })
    session.failureFolder = folder
    const step = { state: session.step.state, route: session.step.route ?? await session.page.evaluate(() => location.hash).catch(() => 'unavailable'),
      control: session.step.control, error: first, all_errors: session.fatalErrors }
    let screenshotError
    try { await session.page.screenshot({ path: join(folder, 'screenshot.png') }) } catch (error) { screenshotError = String(error) }
    if (screenshotError) step.screenshot_error = screenshotError
    writeFileSync(join(folder, 'step.json'), JSON.stringify(step, null, 2))
    writeFileSync(join(folder, 'console.log'), session.consoleLines.join('\n') + '\n')
  }
  const error = new Error(`walk page error at ${session.step.state} ${session.step.route} ${session.step.control}: ${first.kind}: ${first.message}; ${session.failureFolder}`)
  error.walkFailure = true
  error.failureFolder = session.failureFolder
  throw error
}

export async function launchRun(run, options = {}) {
  const exe = join(paths.tree, 'Bubo.Control.Web.exe')
  const consoleLines = []
  const listeners = execFileSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
    .split(/\r?\n/).filter(line => new RegExp(`^\\s*TCP\\s+\\S+:${paths.port}\\s+\\S+\\s+LISTENING\\s+`, 'i').test(line))
  if (listeners.length) throw new Error(`CDP port ${paths.port} already listening: ${listeners.join(' | ')}`)
  mkdirSync(join(paths.evidence, 'walk', 'pid'), { recursive: true })
  const child = await startTestGUI({ exe, cwd: paths.tree, args: options.args ?? [], environment: run.env,
    pidFile: join(paths.evidence, 'walk', 'pid', `${run.id}.txt`),
    onOutput: (stream, data) => consoleLines.push(`${stream}: ${data}`), probeDiagnostics: options.probeDiagnostics ?? false })
  let browser
  try {
    assertIsolated(child)
    // Ready means a WebView2 page exists, not that the port listens (cdp.mjs).
    const attached = await attachToPage(chromium, paths.cdpURL, { wait: 15000,
      alive: () => { if (child.exitCode !== null) throw new Error(`test GUI exited ${child.exitCode} before CDP`) } })
    browser = attached.browser
    const page = attached.page
    const session = { run, child, browser, page, consoleLines, fatalErrors: [],
      step: { state: run.id.match(/walk-(S\d+)/)?.[1] ?? run.id, route: '#/', control: 'startup' } }
    page.on('console', m => {
      const location = m.location().url
      const line = `console ${m.type()}: ${m.text()}${location ? ` (${location})` : ''}`
      consoleLines.push(line)
      if (m.type() === 'error') session.fatalErrors.push({ kind: 'console_error', message: m.text(), location })
    })
    page.on('pageerror', e => {
      consoleLines.push(`pageerror: ${e?.stack ?? e}`)
      session.fatalErrors.push({ kind: 'pageerror', message: String(e), stack: e?.stack })
    })
    // Harness setup, not a product measure (S1/S3 and startup_slow cover a
    // slow start): allow the same 15 s as the CDP connect, and keep the time.
    const bindingsFrom = Date.now()
    await page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: 15000 })
    session.bindingsMs = Date.now() - bindingsFrom
    consoleLines.push(`harness: bindings after ${session.bindingsMs} ms`)
    assertIsolated(child)
    await checkWalkError(session)
    return session
  } catch (error) {
    try { await browser?.close() } catch {}
    await stopTestGUI(child)
    throw error
  }
}

export async function stopRun(session) {
  let pageError
  try { await checkWalkError(session) } catch (error) { pageError = error }
  try { await session.browser?.close() } catch {}
  await stopTestGUI(session.child)
  assertIsolated(session.child)
  if (pageError) throw pageError
}
export async function call(session, method, ...args) {
  await checkWalkError(session)
  const result = await session.page.evaluate(async ({ method, args }) => globalThis.go.main.App[method](...args), { method, args })
  assertIsolated(session.child)
  await checkWalkError(session)
  return result
}
export async function view(session) { return call(session, 'GetView') }
export async function route(session, hash) {
  session.step.route = hash
  await checkWalkError(session)
  await session.page.evaluate(hash => { location.hash = hash }, hash)
  await session.page.waitForTimeout(100)
  assertIsolated(session.child)
  await checkWalkError(session)
}
