// Build the I10 negative control from the verified scratch Wails copy. The
// source edit exists only during this build and is restored in finally.
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { v8Config } from './config.mjs'

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)))
const { port, evidence } = v8Config()
const source = join(evidence, 'wails-patched', 'internal', 'frontend', 'desktop', 'windows', 'frontend.go')
const moduleCache = join(execFileSync(process.env.BUBO_GO ?? 'go', ['env', 'GOMODCACHE'], { encoding: 'utf8', windowsHide: true }).trim(), 'github.com', 'wailsapp', 'wails', 'v2@v2.16.0')
const upstream = join(moduleCache, 'internal', 'frontend', 'desktop', 'windows', 'frontend.go')
const go = (process.env.BUBO_GO ?? 'go')
const originalSHA256 = '1a4f49a0932c38accd152b140f94efb4a4a3ceaaf735e4bb96d9c5e689f5ddba'
const upstreamSHA256 = '9598c7f21779e2332db788a9e09f3b4856d563ab5fc2c93bf7035743b9d7befa'
const cdpLine = `\tchromium.AdditionalBrowserArgs = append(chromium.AdditionalBrowserArgs, "--remote-debugging-port=${port}")`
const modfile = join(evidence, 'build', 'wails-test.mod')
const output = join(evidence, 'tree', 'Bubo.Control.Web.no-cdp.exe')
const hash = value => createHash('sha256').update(value).digest('hex')

if (hash(readFileSync(upstream)) !== upstreamSHA256) throw new Error('pinned Wails module cache changed')
const original = readFileSync(source)
const text = original.toString('utf8')
const baseline = text.replace(cdpLine, cdpLine.replace(String(port), '9333'))
if (hash(baseline) !== originalSHA256) throw new Error('verified scratch Wails frontend changed')
if (text.split(cdpLine).length !== 2) throw new Error('CDP argument is not present exactly once')
const broken = text.replace(cdpLine, '// I10 negative control: CDP port is absent')
const pathKey = Object.keys(process.env).find(name => name.toLowerCase() === 'path') ?? 'Path'
const env = { ...process.env, [pathKey]: (process.env[pathKey] ?? ''), CGO_ENABLED: '0', GOFLAGS: '-mod=readonly' }

try {
  writeFileSync(source, broken)
  execFileSync(go, ['build', `-modfile=${modfile}`, '-trimpath', '-tags=production,wv2runtime.error,bubo_test', '-ldflags=-H=windowsgui', '-o', output], { cwd: join(root, 'gui', 'wails'), env, stdio: 'inherit', windowsHide: true })
} finally {
  writeFileSync(source, original)
  if (hash(readFileSync(source)) !== hash(original)) throw new Error('failed to restore scratch Wails frontend')
}
console.log(JSON.stringify({ output, exe_sha256: hash(readFileSync(output)), restored_wails_sha256: hash(original) }))
