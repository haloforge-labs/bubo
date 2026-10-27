// Hidden-desktop regression for a GUI executable in a folder with spaces and
// non-ASCII characters. The exe sits in bin\ of a package folder and starts with
// its own folder as working directory, as Explorer starts it: once from bin\
// and once from the package root. The launcher must not replace that working
// directory. Invoke with --job <ID> --port <port> after build.mjs.
//
// Red before E6: supervisor init() compiled the contract schema under a
// relative id that the compiler resolved against the working directory, so the
// exe panicked before main and exited with code 2.
import { createRequire } from 'node:module'
import { existsSync, linkSync, mkdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { attachToPage } from './cdp.mjs'
import { defaultDesktopViolations, startTestGUI, stopTestGUI } from './launcher.mjs'
import { makeRun } from './walk/session.mjs'
import { v8Config } from './config.mjs'

const paths = v8Config()
const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const { chromium } = createRequire(join(root, 'gui', 'wails', 'frontend', 'package.json'))('playwright')
const folder = join(paths.evidence, 'Bubo Wälk Test')
const bin = join(folder, 'bin')
mkdirSync(bin, { recursive: true })
function linkCurrent(source, target) {
  if (existsSync(target)) {
    const a = statSync(source)
    const b = statSync(target)
    if (a.dev === b.dev && a.ino === b.ino) return
    unlinkSync(target)
  }
  linkSync(source, target)
}
const exe = join(bin, 'Bubo.Control.Web.exe')
linkCurrent(join(paths.tree, 'Bubo.Control.Web.exe'), exe)
linkCurrent(join(paths.tree, 'BUILD-MANIFEST.json'), join(folder, 'BUILD-MANIFEST.json'))
linkCurrent(join(paths.tree, 'bin', 'bubo.exe'), join(bin, 'bubo.exe'))

async function launchFrom(label, cwd) {
  const run = makeRun(`unicode-path-${label}`, { url: 'https://127.0.0.1:1/manifest.json', ca: join(paths.build, 'loopback.crt') },
    { scenario: 'not-prepared', mode: 'ui' })
  const stderrFile = join(paths.evidence, `path-launch-${label}.stderr.txt`)
  let child
  let browser
  try {
    child = await startTestGUI({ exe, cwd, environment: run.env, pidFile: join(paths.evidence, 'path-launch.pid'), stderrFile })
    const attached = await attachToPage(chromium, paths.cdpURL, { wait: 15000, alive: () => {
      if (child.exitCode !== null) throw new Error(`GUI exited ${child.exitCode} before CDP`)
      if (defaultDesktopViolations(child).length) throw new Error('GUI created a default-desktop window')
    } })
    browser = attached.browser
    const page = attached.page
    await page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: 5000 })
    const view = await page.evaluate(() => globalThis.go.main.App.GetView())
    if (!view?.phase) throw new Error('GetView did not return a phase')
    if (defaultDesktopViolations(child).length) throw new Error('GUI created a default-desktop window')
    return { label, exe, cwd, pid: child.pid, desktop: child.desktop, cdp_port: paths.port, phase: view.phase, default_desktop_violations: 0 }
  } catch (error) {
    const stderr = existsSync(stderrFile) ? readFileSync(stderrFile, 'utf8') : ''
    throw new Error(`${label} (exe ${exe}, cwd ${cwd}): ${error?.message ?? error}\nGUI stderr: ${stderr}`, { cause: error })
  } finally {
    try { await browser?.close() }
    finally { if (child) await stopTestGUI(child) }
  }
}

const results = []
try {
  for (const [label, cwd] of [['bin', bin], ['root', folder]]) {
    const result = await launchFrom(label, cwd)
    results.push(result)
    console.log(`PASS ${folder} from cwd ${label}: CDP ${paths.port}, GetView phase ${result.phase}, 0 desktop violations`)
  }
  writeFileSync(join(paths.evidence, 'path-launch.json'), JSON.stringify(results, null, 2))
} catch (error) {
  writeFileSync(join(paths.evidence, 'path-launch-error.txt'), `${error?.stack ?? error}\n`)
  throw error
}
