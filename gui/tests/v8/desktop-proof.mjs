import { randomUUID } from 'node:crypto'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { v8Config } from './config.mjs'

const { evidence, jobFolder } = v8Config()
const folder = join(evidence, 'desktop-proof')
const root = join(folder, randomUUID())
const desktopctl = join(evidence, 'build', 'desktopctl.exe')
const decoy = join(evidence, 'build', 'decoy.exe')
const stateFile = root + '.state.json'
const baselineFile = root + '.baseline.json'
const violationFile = root + '.violations.jsonl'
const configFile = root + '.config.json'
const resultFile = join(jobFolder, 'DESKTOP-PROOF.json')

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const lines = file => existsSync(file) ? readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse) : []

async function main() {
  mkdirSync(folder, { recursive: true })
  writeFileSync(configFile, JSON.stringify({
    exe: decoy, cwd: evidence, args: ['parent', root + '.primary'],
    state_file: stateFile, baseline_file: baselineFile, violation_file: violationFile,
    outside_job_decoy_exe: decoy, outside_job_decoy_root: root + '.sibling',
  }))
  const child = spawn(desktopctl, [configFile], {
    cwd: evidence, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, LOCALAPPDATA: join(evidence, 'local', 'desktop-proof') },
  })
  let stderr = ''
  child.stderr.on('data', data => { stderr += data.toString() })
  child.stdin.on('error', error => { if (error.code !== 'EPIPE') throw error })
  try {
    const deadline = Date.now() + 10000
    while (!existsSync(stateFile + '.error') && Date.now() < deadline) await sleep(50)
    const baseline = JSON.parse(readFileSync(baselineFile, 'utf8'))
    const violations = lines(violationFile)
    const siblingReady = existsSync(root + '.sibling.parent.json')
    const result = {
      passed: siblingReady && baseline.default.every(w => Number.isInteger(w.handle) && Number.isInteger(w.pid) && typeof w.class === 'string') &&
        violations.length === 1 && violations[0].kind === 'hidden_outside_job' &&
        readFileSync(stateFile + '.error', 'utf8').includes('window outside job on hidden desktop'),
      sibling_ready: siblingReady, baseline_bytes: readFileSync(baselineFile).length,
      violation_bytes: readFileSync(violationFile).length, violations, stderr,
    }
    writeFileSync(resultFile, JSON.stringify(result, null, 2))
    if (!result.passed) throw new Error(`hidden-desktop outside-job decoy was not caught; see ${resultFile}`)
    console.log(`PASS hidden-desktop outside-job decoy: ${violations.length} violation, 0 default-desktop decoys`)
    console.log(resultFile)
  } finally {
    if (child.exitCode === null && !existsSync(stateFile + '.error')) {
      child.stdin.write('stop\n')
    }
    for (let i = 0; i < 100 && child.exitCode === null; i++) await sleep(50)
    if (child.exitCode === null) throw new Error(`desktopctl did not exit after stop: ${stderr}`)
  }
}

main().catch(error => { console.error(error); process.exitCode = 1 })
