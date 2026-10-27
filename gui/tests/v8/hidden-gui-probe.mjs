import { randomUUID } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:https'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { attachToPage } from './cdp.mjs'
import { cleanTestEnvironment, defaultDesktopViolations, readExits, readProbeWindows, startTestGUI, stopTestGUI } from './launcher.mjs'
import { v8Config } from './config.mjs'

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const frontend = join(root, 'gui', 'wails', 'frontend')
const { chromium } = createRequire(join(frontend, 'package.json'))('playwright')
const { job: jobID, port, evidence, jobFolder: job, tree, cdpURL } = v8Config()
const noCDP = process.argv.includes('--no-cdp')
const exe = join(tree, noCDP ? 'Bubo.Control.Web.no-cdp.exe' : 'Bubo.Control.Web.exe')
const runID = randomUUID()
const localAppData = join(evidence, 'local', `gui-smoke-r3-${runID}`)
const captureFile = join(evidence, 'capture', `gui-smoke-r3-${runID}.jsonl`)
const configFile = join(evidence, 'config', `gui-smoke-r3-${runID}.json`)
const resultFile = join(job, noCDP ? 'I10-NEGATIVE.json' : 'GUI-SMOKE-R3.json')
const cert = join(evidence, 'build', 'loopback.crt')
const key = join(evidence, 'build', 'loopback.key')
const openssl = 'C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe'

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)) }
function readCapture() {
  if (!existsSync(captureFile)) return []
  return readFileSync(captureFile, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse)
}
function testLogs(folder) {
  const result = []
  function walk(dir, depth) {
    if (depth > 6 || !existsSync(dir)) return
    for (const item of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, item.name)
      if (item.isDirectory()) walk(path, depth + 1)
      else if (/\.log$|\.dmp$|debug\.log$/i.test(item.name)) {
        let tail = ''
        if (!/\.dmp$/i.test(item.name)) {
          try { tail = readFileSync(path, 'utf8').slice(-65536) } catch {}
        }
        result.push({ path, tail })
      }
    }
  }
  walk(folder, 0)
  return result
}
async function loopbackServer() {
  if (!existsSync(cert) || !existsSync(key)) {
    execFileSync(openssl, ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2', '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1', '-keyout', key, '-out', cert], { windowsHide: true, stdio: 'ignore' })
  }
  const server = createServer({ cert: readFileSync(cert), key: readFileSync(key) }, (_req, res) => { res.writeHead(404); res.end('no update') })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  if (server.address().port >= 8081 && server.address().port <= 8086) {
    await new Promise(resolve => server.close(resolve))
    return loopbackServer()
  }
  return server
}

async function main() {
  const netstat = execFileSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
  const portUsers = netstat.split(/\r?\n/).filter(line => new RegExp(`^\\s*TCP\\s+\\S+:${port}\\s+\\S+\\s+LISTENING\\s+`, 'i').test(line))
  if (portUsers.length) throw new Error(`CDP port ${port} already listening: ${portUsers.join(' | ')}`)
  for (const folder of [localAppData, join(evidence, 'capture'), join(evidence, 'config'), join(evidence, 'startup', runID), join(evidence, 'dialogs', runID), join(evidence, 'fake', runID)]) mkdirSync(folder, { recursive: true })
  const dialogFile = join(evidence, 'dialogs', runID, 'selected.gguf')
  if (!existsSync(dialogFile)) writeFileSync(dialogFile, '')
  const server = await loopbackServer()
  const config = {
    evidence_root: evidence,
    instance_suffix: `.Test.${jobID}.smoke.${randomUUID()}`,
    login_key_path: `Software\\BuboTest\\${jobID}\\${runID}`,
    startup_folder: join(evidence, 'startup', runID),
    key_target: `Bubo.Test.${jobID}.${runID}`,
    capture_file: captureFile,
    dialog_file: dialogFile,
    dialog_folder: join(evidence, 'dialogs', runID),
    manifest_url: `https://127.0.0.1:${server.address().port}/manifest.json`,
    root_ca_file: cert,
  }
  writeFileSync(configFile, JSON.stringify(config, null, 2))
  const env = cleanTestEnvironment({
    LOCALAPPDATA: localAppData,
    BUBO_TEST_CONFIG: configFile,
    BUBO_FAKE_DIR: join(evidence, 'fake', runID),
    BUBO_FAKE_SCENARIO: join(root, 'gui', 'tests', 'scenarios', 'recordings', 'not-prepared'),
    BUBO_FAKE_MODE: 'ui',
    BUBO_FAKE_FIXTURES: join(root, 'tests', 'cli', 'contract'),
    BUBO_FAKE_RECORD_ALL_ENV: '1',
    BUBO_FAKE_RECORD_RUNTIME: '1',
  })
  let child, browser
  const result = { cdp_port: port, initial_cdp_listeners: 0, gui_launched_on_hidden_desktop: false, cdp_connected: false, get_view: false, default_desktop_violations: [], capture: [], exits: [], webview_logs: [], baseline_file: null, violation_file: null }
  let failure
  try {
    child = await startTestGUI({ exe, cwd: tree, environment: env, pidFile: join(evidence, 'pid-gui-smoke-r3.txt'), probeDiagnostics: true })
    result.gui_launched_on_hidden_desktop = true
    result.desktop = child.desktop
    result.pid = child.pid
    result.baseline_file = child.baselineFile
    result.violation_file = child.violationFile
    const deadline = Date.now() + 15000
    if (noCDP) {
      while (Date.now() < deadline && !readCapture().some(event => event.kind === 'tray_start')) {
        const violations = defaultDesktopViolations(child)
        if (violations.length) throw new Error(`test process created a default-desktop window: ${JSON.stringify(violations)}`)
        if (child.exitCode !== null) throw new Error(`no-CDP GUI exited with 0x${child.exitCode.toString(16).padStart(8, '0')}`)
        await sleep(100)
      }
      if (!readCapture().some(event => event.kind === 'tray_start')) throw new Error('no-CDP GUI did not reach tray_start')
      let all, members, webviews
      do {
        all = readProbeWindows(child)
        members = new Map(all.flatMap(s => s.job_processes ?? []).map(member => [`${member.pid}/${member.creation_time}`, member]))
        webviews = [...members.values()].filter(member => member.image.toLowerCase() === 'msedgewebview2.exe')
        if (webviews.length || child.exitCode !== null) break
        await sleep(100)
      } while (Date.now() < deadline)
      const outside = all.flatMap(s => (s.hidden ?? []).filter(w =>
        !(s.job_processes ?? []).some(member => member.pid === w.pid && member.creation_time === w.job_creation_time)))
      result.job_members = [...members.values()]
      result.webview_job_member_count = webviews.length
      result.hidden_outside_job = outside
      if (!webviews.length || outside.length) throw new Error(`WebView2 job membership failed: webviews=${webviews.length}, hidden outside job=${outside.length}`)
      try {
        browser = await chromium.connectOverCDP(cdpURL, { timeout: 1000 })
        result.cdp_connected = true
      } catch {}
      if (result.cdp_connected) throw new Error(`no-CDP build unexpectedly exposed port ${port}`)
      result.default_desktop_violations = defaultDesktopViolations(child)
      if (result.default_desktop_violations.length) throw new Error('no-CDP GUI created a default-desktop window')
      console.log(`PASS I10 no-CDP negative: ${webviews.length} WebView2 job members, 0 hidden breakaways, 0 default-desktop test windows, port ${port} absent`)
    } else {
      const attached = await attachToPage(chromium, cdpURL, { wait: Math.max(1000, deadline - Date.now()), alive: () => {
        const violations = defaultDesktopViolations(child)
        if (violations.length) {
          result.default_desktop_violations = violations
          throw new Error(`test process created a default-desktop window: ${JSON.stringify(violations)}`)
        }
        if (child.exitCode !== null) throw new Error(`test GUI exited with 0x${child.exitCode.toString(16).padStart(8, '0')}`)
      } })
      browser = attached.browser
      result.cdp_connected = true
      const page = attached.page
      await page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: 5000 })
      const view = await page.evaluate(async () => globalThis.go.main.App.GetView())
      result.get_view = Boolean(view)
      if (!result.get_view) throw new Error('GetView returned no View')
      result.default_desktop_violations = defaultDesktopViolations(child)
      if (result.default_desktop_violations.length) throw new Error('test process created a default-desktop window after GetView')
      console.log(`PASS hidden GUI: PID ${child.pid}, desktop ${child.desktop}, CDP GetView returned a View`)
    }
  } catch (error) {
    failure = error
    result.failure = String(error)
  } finally {
    try { await browser?.close() } catch {}
    if (child) {
      try { await stopTestGUI(child) } catch (error) { result.stop_error = String(error) }
      result.exits = readExits(child)
      result.window_changes = readProbeWindows(child).length
      result.default_desktop_violations = defaultDesktopViolations(child)
    }
    result.capture = readCapture()
    result.webview_logs = testLogs(localAppData)
    writeFileSync(resultFile, JSON.stringify(result, null, 2))
    await new Promise(resolve => server.close(resolve))
  }
  if (failure) throw failure
  if (result.stop_error) throw new Error(result.stop_error)
  console.log(resultFile)
}

main().catch(error => { console.error(error); process.exitCode = 1 })
