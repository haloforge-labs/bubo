import { randomUUID } from 'node:crypto'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { v8Config } from './config.mjs'

const { evidence } = v8Config()
const desktopctl = join(evidence, 'build', 'desktopctl.exe')
const launches = join(evidence, 'launches')

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)) }

// The updater restart inherits this same clean environment. Only explicit
// BUBO_* names are admitted. WebView2 must use the test build's own settings.
export function cleanTestEnvironment(values = {}) {
  return Object.fromEntries(Object.entries({ ...process.env, ...values }).filter(([name]) =>
    (!/^BUBO_/i.test(name) && !/^WEBVIEW2_/i.test(name)) || Object.hasOwn(values, name)))
}

export function readSnapshots(child) {
  if (existsSync(child.stateFile + '.error')) throw new Error(`desktopctl failed for job root ${child.pid}: ${readFileSync(child.stateFile + '.error', 'utf8')}`)
  if (!existsSync(child.violationFile)) return []
  const content = readFileSync(child.violationFile, 'utf8')
  const complete = content.slice(0, content.lastIndexOf('\n'))
  return complete.split(/\r?\n/).filter(Boolean).map(JSON.parse)
}

// Probe diagnostics stay in memory. The walk never requests this stream.
export function readProbeWindows(child) { return child.probeWindows ?? [] }

export function readExits(child) {
  if (!existsSync(child.exitFile)) return []
  try {
    const exits = JSON.parse(readFileSync(child.exitFile, 'utf8'))
    return Array.isArray(exits) ? exits : []
  } catch { return [] }
}

export function defaultDesktopViolations(child) {
  return readSnapshots(child).filter(item => item.kind === 'default_job_window').map(item => item.window)
}

export async function startTestGUI({ exe, cwd, args = [], environment, pidFile, onOutput = () => {}, probeDiagnostics = false, stderrFile = '' }) {
  mkdirSync(launches, { recursive: true })
  const stem = join(launches, randomUUID())
  const configFile = `${stem}.config.json`
  const stateFile = `${stem}.state.json`
  const baselineFile = `${stem}.baseline.json`
  const violationFile = `${stem}.violations.jsonl`
  const exitFile = `${stem}.exits.json`
  // The GUI resolves its package from its exe. A space or non-ASCII character
  // in its working directory makes WebView2's child launch exit before CDP.
  // Keep the desktop controller and GUI on the fixed harness working directory.
  const launchCwd = cwd
  writeFileSync(configFile, JSON.stringify({ exe, cwd: launchCwd, args, state_file: stateFile, baseline_file: baselineFile,
    violation_file: violationFile, exit_file: exitFile, probe_diagnostics: probeDiagnostics, stderr_file: stderrFile }))
  const controller = spawn(desktopctl, [configFile], {
    cwd: dirname(desktopctl), env: cleanTestEnvironment(environment), windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
  })
  let diagnostic = ''
  const probeWindows = []
  let stdoutLine = ''
  controller.stdout.on('data', data => {
    stdoutLine += data.toString()
    let end
    while ((end = stdoutLine.indexOf('\n')) !== -1) {
      const line = stdoutLine.slice(0, end).trimEnd()
      stdoutLine = stdoutLine.slice(end + 1)
      if (line.startsWith('PROBE_WINDOWS ')) probeWindows.push(JSON.parse(line.slice('PROBE_WINDOWS '.length)))
      else onOutput('stdout', line + '\n')
    }
  })
  controller.stderr.on('data', data => { diagnostic += data.toString(); onOutput('stderr', data) })
  for (let i = 0; i < 100; i++) {
    if (existsSync(stateFile)) {
      let state
      try {
        state = JSON.parse(readFileSync(stateFile, 'utf8'))
      } catch (error) {
        // desktopctl writes this file in place. Its first visible bytes can
        // precede the closing brace when the editing host is busy.
        if (!(error instanceof SyntaxError)) throw error
        await sleep(50)
        continue
      }
      writeFileSync(pidFile, `${state.pid}\n`)
      let stopSent = false
      return {
        pid: state.pid, creationTime: state.creation_time, desktop: state.desktop, controller, baselineFile, violationFile, exitFile, stateFile, probeWindows,
        get exitCode() { return readExits(this).find(row => row.pid === state.pid && row.creation_time === state.creation_time)?.code ?? null },
        closeWindow() { if (controller.exitCode === null) controller.stdin.write('close\n') },
        kill() { if (!stopSent && controller.exitCode === null) { controller.stdin.write('stop\n'); stopSent = true } },
      }
    }
    if (controller.exitCode !== null) throw new Error(`desktopctl exited before GUI state (${controller.exitCode}): ${diagnostic}`)
    await sleep(50)
  }
  if (controller.exitCode === null) controller.stdin.write('stop\n')
  throw new Error(`desktopctl did not publish GUI state: ${diagnostic}`)
}

export async function stopTestGUI(child) {
  child.kill()
  for (let i = 0; i < 400; i++) {
    if (existsSync(child.stateFile + '.done')) return
    if (existsSync(child.stateFile + '.error')) throw new Error(`desktopctl failed for job root ${child.pid}: ${readFileSync(child.stateFile + '.error', 'utf8')}`)
    await sleep(50)
  }
  throw new Error(`desktopctl did not empty job rooted at ${child.pid}; desktop ${child.desktop} remains held`)
}
