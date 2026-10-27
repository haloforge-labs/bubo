import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { cleanTestEnvironment, readSnapshots, startTestGUI, stopTestGUI } from '../launcher.mjs'
import { v8Config } from '../config.mjs'

// Run the updater's tagged test binary on the same private desktop and inside
// the same Job Object as every GUI child it starts. The runner never starts a
// packaged or owner-installed Bubo executable.
const { evidence } = v8Config()
const name = process.env.A4_GO_CASE
const moduleName = process.env.A4_GO_MODULE ?? 'updater'
if (!['updater', 'supervisor'].includes(moduleName) ||
  !/^(TestV8F(?:3|11|12|14|16|17|19|25)|TestUncooperativeCandidateRetainsPreviousThenRecoverRollsBack)/.test(name ?? '')) {
  throw new Error('set A4_GO_CASE to an approved updater gate test')
}
const exe = join(evidence, `${moduleName}-gate.exe`)
const dir = join(evidence, 'updater', 'tmp')
const log = join(evidence, 'updater', `${name}.txt`)
mkdirSync(dir, { recursive: true })
const script = join(evidence, 'updater', `${name}.cmd`)
writeFileSync(script, `@echo off\r\n"${exe}" -test.run ${name} -test.v > "${log}" 2>&1\r\nexit /b %errorlevel%\r\n`)
const child = await startTestGUI({ exe: 'C:\\Windows\\System32\\cmd.exe', cwd: join(process.cwd(), 'gui', moduleName),
  args: ['/d', '/c', script],
  environment: cleanTestEnvironment({ TEMP: dir, TMP: dir }), pidFile: join(evidence, 'updater', `${name}.pid`),
  onOutput: () => {} })
try {
  const end = Date.now() + (name.includes('F17') || name.includes('F3') ? 110000 : name.includes('F19') ? 90000 : 50000)
  while (Date.now() < end && child.exitCode === null) {
    const violations = readSnapshots(child)
    if (violations.length) throw new Error(`private desktop violation: ${JSON.stringify(violations)}`)
    await new Promise(resolve => setTimeout(resolve, 200))
  }
  if (child.exitCode === null) throw new Error(`${name} exceeded test deadline`)
  const output = readFileSync(log, 'utf8')
  console.log(output)
  console.log(`${name}: exit ${child.exitCode}; 0 desktop violations`)
  if (child.exitCode !== 0) process.exitCode = 1
} finally {
  await stopTestGUI(child)
}
