import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { chmodSync, copyFileSync, existsSync, linkSync, mkdirSync, readFileSync, readdirSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { v8Config } from './config.mjs'

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const { port, evidence, tree, build: scratch } = v8Config()
const patchedWails = join(evidence, 'wails-patched')
const go = (process.env.BUBO_GO ?? 'go')
const wails = join(execFileSync(go, ['env', 'GOPATH'], { encoding: 'utf8', windowsHide: true }).trim(), 'bin', 'wails.exe')
const frontend = join(root, 'gui', 'wails', 'frontend')
const gui = join(root, 'gui', 'wails')
const tests = join(root, 'gui', 'tests')
const fakeCLI = join(tests, 'fakecli')
const moduleRoot = join(execFileSync(go, ['env', 'GOMODCACHE'], { encoding: 'utf8', windowsHide: true }).trim(), 'github.com', 'wailsapp', 'wails', 'v2@v2.16.0')
const patch = join(root, 'gui', 'tests', 'v8', 'wails-capture.patch')
const originalFrontendSHA256 = '9598c7f21779e2332db788a9e09f3b4856d563ab5fc2c93bf7035743b9d7befa'
const patchSHA256 = 'b7950ff6ac5454517c09d7268a12ef63b50e3a4e134cc7d6d9442f1bc08d9483'
const patchedHashes = new Map([
  ['internal/frontend/desktop/windows/frontend.go', '1a4f49a0932c38accd152b140f94efb4a4a3ceaaf735e4bb96d9c5e689f5ddba'],
  ['internal/frontend/desktop/windows/dialog.go', '4217d5197c82691f1bb87190a5e254dc0efbe9a94196414d92c484d51c95f3ad'],
  ['internal/frontend/desktop/windows/winc/w32/user32.go', 'e66b7068bacfca910ab685e4900f3927aa3160439c81a58d277bceda163c70fc'],
  ['internal/frontend/desktop/windows/winc/w32/testcapture.go', 'ff59ab3fc3b44682cdfba88e03eeecb1ae1f492ac13b8a85ed822e5058ccaa21'],
])
const addedFile = 'internal/frontend/desktop/windows/winc/w32/testcapture.go'
const frontendFile = 'internal/frontend/desktop/windows/frontend.go'
const baseCDPLine = 'chromium.AdditionalBrowserArgs = append(chromium.AdditionalBrowserArgs, "--remote-debugging-port=9333")'
const jobCDPLine = baseCDPLine.replace('9333', String(port))
const pathKey = Object.keys(process.env).find(name => name.toLowerCase() === 'path') ?? 'Path'
const toolEnv = { ...process.env, [pathKey]: (process.env[pathKey] ?? ''), CGO_ENABLED: '0', GOFLAGS: '-mod=readonly', PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD: '1' }

function run(file, args, cwd) {
  execFileSync(file, args, { cwd, stdio: 'inherit', windowsHide: true, env: toolEnv })
}
function sha256(value) { return createHash('sha256').update(value).digest('hex') }
function hashFile(file) { return sha256(readFileSync(file)) }
function filesUnder(base) {
  const result = []
  function walk(folder) {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      const path = join(folder, entry.name)
      if (entry.isDirectory()) walk(path)
      else if (entry.isFile()) result.push(relative(base, path).replaceAll('\\', '/'))
      else throw new Error(`unexpected Wails module entry: ${path}`)
    }
  }
  walk(base)
  return result.sort()
}

mkdirSync(join(tree, 'bin'), { recursive: true })
mkdirSync(scratch, { recursive: true })
const prodMod = readFileSync(join(gui, 'go.mod'))
const prodSum = readFileSync(join(gui, 'go.sum'))
const sumText = prodSum.toString('utf8')
if (!sumText.includes('github.com/wailsapp/wails/v2 v2.16.0 ') || !sumText.includes('github.com/wailsapp/go-webview2 v1.0.22 ')) {
  throw new Error('Wails or go-webview2 version changed; review the test patch')
}
if (hashFile(join(moduleRoot, 'internal', 'frontend', 'desktop', 'windows', 'frontend.go')) !== originalFrontendSHA256) {
  throw new Error('Wails frontend.go module-cache hash changed')
}
const patchText = readFileSync(patch, 'utf8').replace(/\r\n/g, '\n')
if (sha256(patchText) !== patchSHA256) throw new Error('committed Wails patch hash changed')
const scratchPatch = join(scratch, 'wails-capture-lf.patch')
writeFileSync(scratchPatch, patchText)

let linkedFiles = 0
let linkedBytes = 0
let copiedBytes = 0
if (!existsSync(patchedWails)) {
  // The module cache is the shared, pinned copy. The verified patch changes
  // only these three existing files plus one new file. Copy those before git
  // apply; every other file is a read-only hard link to the shared copy.
  for (const name of filesUnder(moduleRoot)) {
    const source = join(moduleRoot, name)
    const target = join(patchedWails, name)
    mkdirSync(dirname(target), { recursive: true })
    if (patchedHashes.has(name)) {
      copyFileSync(source, target)
    } else {
      linkSync(source, target)
    }
  }
  for (const name of patchedHashes.keys()) {
    if (name !== addedFile) chmodSync(join(patchedWails, name), 0o666)
  }
  run('git', ['-c', 'core.autocrlf=false', 'apply', '--check', scratchPatch], patchedWails)
  run('git', ['-c', 'core.autocrlf=false', 'apply', scratchPatch], patchedWails)
  if (port !== 9333) {
    const file = join(patchedWails, frontendFile)
    const original = readFileSync(file, 'utf8')
    if (hashFile(file) !== patchedHashes.get(frontendFile) || original.split(baseCDPLine).length !== 2) {
      throw new Error('pinned patched Wails frontend changed before CDP port substitution')
    }
    writeFileSync(file, original.replace(baseCDPLine, jobCDPLine))
  }
}

const sourceFiles = filesUnder(moduleRoot)
const copyFiles = filesUnder(patchedWails)
const expectedFiles = [...sourceFiles, addedFile].sort()
if (JSON.stringify(copyFiles) !== JSON.stringify(expectedFiles)) {
  throw new Error('scratch Wails copy file list differs from pinned module plus the patch')
}
let unpatchedVerified = 0
for (const name of copyFiles) {
  const file = join(patchedWails, name)
  let actual = hashFile(file)
  if (name === frontendFile && port !== 9333) {
    const content = readFileSync(file, 'utf8')
    if (content.split(jobCDPLine).length !== 2) throw new Error('CDP port is not present exactly once in scratch Wails')
    actual = sha256(content.replace(jobCDPLine, baseCDPLine))
  }
  const expected = patchedHashes.get(name) ?? hashFile(join(moduleRoot, name))
  if (actual !== expected) throw new Error(`scratch Wails SHA-256 mismatch: ${name}`)
  if (!patchedHashes.has(name)) unpatchedVerified++
}
// Existing job folders predate hard links. After verifying every byte, convert
// their unchanged files in place. Never replace a file the patch can modify.
for (const name of copyFiles) {
  if (patchedHashes.has(name)) continue
  const source = join(moduleRoot, name)
  const target = join(patchedWails, name)
  const shared = statSync(source)
  const current = statSync(target)
  if (shared.dev !== current.dev || shared.ino !== current.ino) {
    unlinkSync(target)
    try { linkSync(source, target) }
    catch (error) { copyFileSync(source, target); throw error }
  }
  linkedFiles++
  linkedBytes += shared.size
}
copiedBytes = [...patchedHashes.keys()].reduce((sum, name) => sum + statSync(join(patchedWails, name)).size, 0)

const testMod = join(scratch, 'wails-test.mod')
const modText = prodMod.toString('utf8')
  .replace('replace bubo/gui/supervisor => ../supervisor', `replace bubo/gui/supervisor => ${join(root, 'gui', 'supervisor').replaceAll('\\', '/')}`)
  .replace('replace bubo/gui/updater => ../updater', `replace bubo/gui/updater => ${join(root, 'gui', 'updater').replaceAll('\\', '/')}`)
writeFileSync(testMod, modText + `\nreplace github.com/wailsapp/wails/v2 => ${patchedWails.replaceAll('\\', '/')}\n`)
writeFileSync(join(scratch, 'wails-test.sum'), prodSum)

if (!process.argv.includes('--skip-frontend')) {
  run(wails, ['generate', 'module'], gui)
  run('C:\\Program Files\\nodejs\\node.exe', ['C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js', 'run', 'build'], frontend)
}
run(go, ['build', '-o', join(tree, 'bin', 'bubo.exe')], fakeCLI)
// The package's own version record, as a release package carries it. 0.1.0
// matches the fake CLI's reported version; update fixtures offer 0.1.1.
writeFileSync(join(tree, 'BUILD-MANIFEST.json'), JSON.stringify({ product: { product_version: '0.1.0' },
  gui: { commit: '0'.repeat(40), runtime_id: 'test-tree' } }) + '\n')
run(go, ['build', '-tags=bubo_test', '-o', join(scratch, 'winhelper.exe'), './v8'], tests)
run(go, ['build', '-ldflags=-H=windowsgui', '-o', join(scratch, 'desktopctl.exe'), './v8/desktopctl'], tests)
run(go, ['build', '-ldflags=-H=windowsgui', '-o', join(scratch, 'decoy.exe'), './v8/decoy'], tests)
// Wails' -webview2 error selection is the wv2runtime.error Go build tag.
// These mirror gui/wails/build.ps1; the test exe adds only bubo_test and the
// alternate modfile's hash-checked Wails replacement.
run(go, ['build', `-modfile=${testMod}`, '-trimpath', '-tags=production,wv2runtime.error,bubo_test', '-ldflags=-H=windowsgui', '-o', join(tree, 'Bubo.Control.Web.exe')], gui)
run(go, ['build', '-trimpath', '-tags=production,wv2runtime.error', '-ldflags=-H=windowsgui', '-o', join(scratch, 'Bubo.Control.Web.production.exe')], gui)
if (!prodMod.equals(readFileSync(join(gui, 'go.mod'))) || !prodSum.equals(readFileSync(join(gui, 'go.sum')))) {
  throw new Error('production go.mod/go.sum changed during test build')
}
console.log(JSON.stringify({ tree, patchedWails, unpatchedVerified, patchedVerified: patchedHashes.size,
  linkedFiles, linkedBytes, copiedBytes, oldCopyBytes: linkedBytes + copiedBytes, patchSHA256, cdpPort: port }))
