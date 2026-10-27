import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { checkWalkError, loopbackManifest, paths, route } from './session.mjs'
import { leaveState, reachState } from './state.mjs'

const root = resolve(fileURLToPath(new URL('../../../../', import.meta.url)))
const name = process.env.WALK_BREAK
const specs = {
  W2: ['gui/wails/frontend/src/components/copy-text.tsx', 'onClick={() => void task.run(() => actions.CopyText(text))}', 'onClick={() => { void actions }}', 'S4'],
  W3: ['gui/supervisor/app_view_windows.go', 'return a.invoke("diagnose", "")', 'return a.invoke("walk_unknown", "")', 'S4'],
  W4: ['gui/supervisor/actions_windows.go', 'return ActionAvailability{ReasonCode: code, Reason: reason}', 'return ActionAvailability{Reason: reason}', 'S4'],
  W7: ['gui/wails/frontend/src/features/about/index.tsx', 'label="Copy version"', 'label="Copy ROCm version"', 'S4'],
  W9: ['gui/wails/frontend/src/features/server/components/server-setup.tsx',
    '{!view.failure?.switch &&\n          !running &&\n          preset &&',
    '<button type="button">Use 262,144 tokens</button>\n        {!view.failure?.switch &&\n          !running &&\n          preset &&', 'S10'],
  W10: ['gui/supervisor/backend_windows.go', 'c := config\n\t\t\tc.RuntimeDir = settings.RuntimePath',
    'c := config\n\t\t\tif action == "restart" { settings.Port = config.Port }\n\t\t\tc.RuntimeDir = settings.RuntimePath', 'S12'],
  W11: ['gui/tests/v8/walk/cases.mjs', 'const controls = []', 'const controls = []; routes.length = 0 // W11 scratch empty route', 'S4'],
  W12: ['gui/wails/app_windows.go', 'err != nil || !strings.EqualFold(u.Scheme, "https") || u.Host == ""',
    'err != nil || (false && strings.EqualFold(u.Scheme, "https"))', 'S4'],
  W14: ['gui/wails/frontend/src/routes/about.tsx', 'component: Page',
    "component: () => { void Page; throw new Error('W14 scratch route throw') }", 'S4'],
  W17: ['gui/tests/fakecli/main.go', 'if source := os.Getenv("BUBO_FAKE_SERVER_LOG_FIXTURE"); source != "" {',
    'if source := os.Getenv("BUBO_FAKE_SERVER_LOG_FIXTURE"); source != "" && false {', 'S12'],
}
const spec = specs[name]
if (!spec) throw new Error('set WALK_BREAK to W2, W3, W4, W7, W9, W10, W11, W12, W14 or W17')
const [source, from, to, stateName] = spec
const target = join(root, source)
const original = readFileSync(target, 'utf8')
const normalized = original.replaceAll('\r\n', '\n')
if (normalized.split(from).length !== 2) throw new Error(`${name} scratch anchor changed`)
const build = () => execFileSync('node', [join(root, 'gui', 'tests', 'v8', 'build.mjs'), '--job', paths.job, '--port', String(paths.port)],
  { cwd: root, stdio: 'inherit', windowsHide: true })
let manifest, state
try {
  writeFileSync(target, normalized.replace(from, to))
  if (name !== 'W11') build()
  manifest = await loopbackManifest()
  state = await reachState(stateName, manifest)
  await state.session.page.waitForTimeout(2100)
  const { controlCases, linkCase, liveLogCase, passiveCases, settingsRestartCase } = await import(`./cases.mjs?break=${name}`)
  const results = {}
  let detail
  if (name === 'W2' || name === 'W3' || name === 'W11') {
    const only = name === 'W2' ? item => item.route === 'About' && item.text === 'Copy version' :
      name === 'W3' ? item => item.route === 'Logs' && item.text === 'Run diagnostics' : null
    await controlCases(state, results, only)
    detail = results[name === 'W2' ? 'W2:About:Copy version' : name]
  } else if (name === 'W10') {
    await settingsRestartCase(state, results)
    detail = results.W10
  } else if (name === 'W12') {
    await linkCase(state, results)
    detail = results.W12
  } else if (name === 'W14') {
    // The scratch throw is this case's own input. Navigate without route()'s
    // error check, drop only the errors that carry the throw's own message,
    // then check again so any other page or console error still fails.
    const session = state.session
    session.step.route = '#/about'
    await checkWalkError(session)
    await session.page.evaluate(() => { location.hash = '#/about' })
    await session.page.waitForTimeout(500)
    const screen = await session.page.locator('body').innerText()
    for (let i = session.fatalErrors.length - 1; i >= 0; i--)
      if (String(session.fatalErrors[i].message ?? '').includes('W14 scratch route throw')) session.fatalErrors.splice(i, 1)
    await checkWalkError(session)
    detail = { passed: screen.includes('This page hit a problem. Go to the Server page, or restart Bubo.') &&
      !screen.includes('Something went wrong!'), checks: [{ detail: { screen } }] }
  } else if (name === 'W17') {
    const audit = await passiveCases(state, results)
    liveLogCase(state, audit.frames, results)
    detail = results.W17
  } else {
    await passiveCases(state, results)
    detail = results[name]
  }
  const evidence = { break: name, source, state: stateName, passed: detail?.passed === true, detail }
  writeFileSync(join(paths.evidence, 'walk', `break-${name}.json`), JSON.stringify(evidence, null, 2))
  console.log(JSON.stringify({ break: name, passed: evidence.passed, count: detail?.checks?.length ?? 0 }))
  if (evidence.passed) throw new Error(`${name} scratch break did not make the gate RED`)
} finally {
  if (state) await leaveState(state)
  if (manifest) await manifest.close()
  writeFileSync(target, original)
  if (name !== 'W11') build()
  if (readFileSync(target, 'utf8') !== original) throw new Error(`${name} scratch file was not restored`)
}
