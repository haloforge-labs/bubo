// W19-W25: hidden-desktop tray, login, and LAN journeys. Run after build.mjs.
import { createHash } from 'node:crypto'
import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { cleanTestEnvironment, startTestGUI, stopTestGUI } from './launcher.mjs'
import { call, checkWalkError, launchRun, loopbackManifest, paths, readCapture, route, stopRun, view } from './walk/session.mjs'
import { leaveState, reachState, waitView } from './walk/state.mjs'

const selected = process.env.FLOW_CASE ?? ''
const names = ['W19', 'W20', 'W21', 'W22', 'W23', 'W24', 'W25']
if (selected && !names.includes(selected)) throw new Error(`unknown case ${selected}`)
const results = {}
const captures = []
let fatalStep = null
const manifest = await loopbackManifest()
mkdirSync(join(paths.evidence, 'flows'), { recursive: true })
mkdirSync(paths.shots, { recursive: true })

function record(name, passed, detail) {
  const row = results[name] ??= { checks: [] }
  row.checks.push({ passed: Boolean(passed), detail })
  console.log(`${name}: ${passed ? 'PASS' : 'RED'} ${JSON.stringify(detail)}`)
}
function hash(value) { return createHash('sha256').update(value).digest('hex') }
function serveArgs(run) {
  const file = join(run.fakeDir, 'runtime-children.jsonl')
  if (!existsSync(file)) return []
  return readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse)
    .map(row => row.Args).filter(args => args[0] === 'serve' && !args.includes('--list-switches'))
}
async function click(session, locator) {
  await checkWalkError(session)
  await locator.click({ timeout: 5000 })
  await checkWalkError(session)
}
async function paint(session, hashRoute) {
  await route(session, hashRoute)
  await session.page.waitForTimeout(300)
  await checkWalkError(session)
}
async function walk(name, stateName, body, tag = stateName) {
  let state
  try {
    state = await reachState(stateName, manifest)
    await state.session.page.waitForTimeout(2100)
    await body(state)
    await checkWalkError(state.session)
    const screenshot = join(paths.shots, `${name}-${tag}.png`)
    if (!state.session.page.isClosed()) await state.session.page.screenshot({ path: screenshot })
    captures.push({ case: name, state: stateName, tag, config: state.run.configPath,
      capture_file: state.run.captureFile, screenshot })
  } catch (error) {
    if (error?.walkFailure) throw error
    record(name, false, { fixture_error: String(error) })
  } finally {
    if (state) {
      try {
        if (name === 'W24') await helper(state.run, 'registry', 'clean')
        if (name === 'W25') await helper(state.run, 'credential', 'clean')
      } finally { await leaveState(state) }
    }
  }
}
async function lanOn(state) {
  await paint(state.session, '#/connect')
  await click(state.session, state.session.page.getByRole('switch', { name: 'Local network access' }))
  return waitView(state.session, v => v.key?.lan_access === true)
}
function latestMenu(run) {
  const entry = readCapture(run).filter(item => item.kind === 'tray_menu').at(-1)
  if (!entry) throw new Error('tray capture has no menu')
  return JSON.parse(entry.value)
}
async function helper(run, ...args) {
  const output = join(paths.evidence, 'flows', `helper-${randomUUID()}.txt`)
  const exe = join(paths.build, 'winhelper.exe')
  const environment = cleanTestEnvironment({ BUBO_TEST_CONFIG: run.configPath, LOCALAPPDATA: run.local, BUBO_TEST_HELPER_OUTPUT: output })
  const child = await startTestGUI({ exe, cwd: paths.tree, args, environment,
    pidFile: join(paths.evidence, 'flows', `helper-${randomUUID()}.pid`) })
  try {
    const deadline = Date.now() + 10000
    while (child.exitCode === null && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 50))
    if (child.exitCode !== 0) throw new Error(`hidden helper ${args.join(' ')} exited ${child.exitCode}`)
    return existsSync(output) ? readFileSync(output, 'utf8').trim() : ''
  } finally { await stopTestGUI(child) }
}

try {
  if (!selected || selected === 'W19') await walk('W19', 'S12', async state => {
    const initial = serveArgs(state.run).length
    const changed = await lanOn(state)
    const page = state.session.page
    const button = page.getByRole('button', { name: /restart/i })
    const link = page.getByRole('link', { name: /restart/i })
    let restarted = false
    if (await button.count() && await button.first().isEnabled()) {
      await click(state.session, button.first())
      const end = Date.now() + 12000
      while (Date.now() < end && serveArgs(state.run).length <= initial) await page.waitForTimeout(100)
      restarted = serveArgs(state.run).length > initial
    }
    record('W19', changed.key.lan_access && await button.count() > 0 && await link.count() === 0 && restarted,
      { lan_on: changed.key.lan_access, restart_buttons: await button.count(), restart_links: await link.count(), dispatched_restart: restarted })
  })

  if (!selected || selected === 'W20') await walk('W20', 'S12', async state => {
    await paint(state.session, '#/')
    const link = state.session.page.getByRole('link', { name: 'Settings', exact: true }).last()
    const found = await link.count()
    if (found) await click(state.session, link)
    const active = await state.session.page.locator('[role="tab"][data-state="active"]').allTextContents()
    record('W20', found > 0 && (await state.session.page.evaluate(() => location.hash)).startsWith('#/settings') && active.some(text => text.trim() === 'Inference'),
      { link_found: found > 0, route: await state.session.page.evaluate(() => location.hash), active_tabs: active })
  })

  if (!selected || selected === 'W21') await walk('W21', 'S7', async state => {
    await paint(state.session, '#/models')
    const oldCount = await state.session.page.getByText('Use on Server', { exact: true }).count()
    const newCount = await state.session.page.getByText('Select for Server', { exact: true }).count()
    record('W21', oldCount === 0 && newCount > 0, { old_count: oldCount, new_count: newCount })
  })

  if (!selected || selected === 'W22') await walk('W22', 'S7', async state => {
    const setting = '--temperature'
    const replacement = '0.5'
    const set = await call(state.session, 'SetSwitchOverrides', { set: [{ name: setting, value: replacement }], clear: [] })
    if (set.error) throw new Error(`setting setup failed: ${set.error.code}`)
    await waitView(state.session, v => v.switch_overrides?.some(row => row.name === setting && row.saved != null))
    await paint(state.session, '#/settings')
    await click(state.session, state.session.page.getByRole('tab', { name: 'Inference' }))
    const row = state.session.page.locator('[data-setting="--temperature"]')
    const helpBefore = await row.innerText()
    await click(state.session, row.getByRole('button', { name: 'Reset to default' }))
    await click(state.session, state.session.page.getByRole('button', { name: 'Save changes' }))
    const after = await waitView(state.session, v => !v.switch_overrides?.some(item => item.name === setting && item.saved != null))
    const registry = after.switches.find(item => item.name === setting)
    const helpAfter = await row.innerText()
    const help = registry?.note || registry?.purpose
    record('W22', !after.switch_overrides?.some(item => item.name === setting && item.saved != null) && Boolean(help) && helpBefore.includes(help) && helpAfter.includes(help),
      { reset_to_cli_default: true, help_present_before: Boolean(help && helpBefore.includes(help)), help_present_after: Boolean(help && helpAfter.includes(help)), default_text: registry?.default_text })
  })

  if (!selected || selected === 'W23') {
    for (const [action, stateName] of [['show', 'S7'], ['start', 'S12'], ['stop', 'S12'], ['restart', 'S12'], ['quit', 'S4']]) {
      await walk('W23', stateName, async state => {
        if (action === 'start') {
          const stopped = await call(state.session, 'Stop')
          if (stopped.error) throw new Error(`start setup could not stop server: ${stopped.error.code}`)
          await waitView(state.session, v => v.state === 'Stopped')
          await state.session.page.waitForTimeout(300)
        }
        const item = latestMenu(state.run).Items.find(row => row.Action === action && row.Enabled)
        const known = new Set(['show', 'start', 'stop', 'restart', 'quit'])
        const menu = latestMenu(state.run)
        const mapped = menu.Items.filter(row => row.Enabled && !row.Separator).every(row => known.has(row.Action))
        if (!item) { record('W23', false, { action, menu_item: 'missing or disabled', mapped }); return }
        const before = serveArgs(state.run).length
        try { await call(state.session, 'TestTrayAction', action) } catch (error) {
          if (action !== 'quit') throw error // Quit may close CDP before the binding replies.
        }
        let effect = false
        if (action === 'show') effect = readCapture(state.run).some(row => row.kind === 'show')
        else if (action === 'start') effect = (await waitView(state.session, v => v.state === 'Ready')).state === 'Ready'
        else if (action === 'stop') effect = (await waitView(state.session, v => v.state === 'Stopped')).state === 'Stopped'
        else if (action === 'restart') {
          const end = Date.now() + 12000
          while (Date.now() < end && serveArgs(state.run).length <= before) await state.session.page.waitForTimeout(100)
          effect = serveArgs(state.run).length > before
        } else {
          const end = Date.now() + 10000
          while (Date.now() < end && state.session.child.exitCode === null) await new Promise(resolve => setTimeout(resolve, 100))
          effect = state.session.child.exitCode !== null
        }
        record('W23', mapped && effect, { action, menu_item: item.Label, mapped, effect })
      }, action)
    }
  }

  if (!selected || selected === 'W24') await walk('W24', 'S4', async state => {
    await paint(state.session, '#/settings')
    await click(state.session, state.session.page.getByRole('tab', { name: 'App' }))
    const toggle = state.session.page.getByRole('switch', { name: 'Start this app at login' })
    await click(state.session, toggle)
    await click(state.session, state.session.page.getByRole('button', { name: 'Save app settings' }))
    const on = await waitView(state.session, v => v.saved.start_gui_at_login === true)
    const value = await helper(state.run, 'registry', 'value')
    const expected = `"${join(paths.tree, 'Bubo.Control.Web.exe')}"`
    await stopRun(state.session)
    state.session = await launchRun(state.run)
    const restored = await waitView(state.session, v => v.initialized && v.saved.start_gui_at_login === true)
    await paint(state.session, '#/settings')
    await click(state.session, state.session.page.getByRole('tab', { name: 'App' }))
    const checked = await state.session.page.getByRole('switch', { name: 'Start this app at login' }).getAttribute('data-state')
    await click(state.session, state.session.page.getByRole('switch', { name: 'Start this app at login' }))
    await click(state.session, state.session.page.getByRole('button', { name: 'Save app settings' }))
    const off = await waitView(state.session, v => v.saved.start_gui_at_login === false)
    const absent = await helper(state.run, 'registry', 'present') === 'false'
    record('W24', on.saved.start_gui_at_login && value === expected && restored.saved.start_gui_at_login && checked === 'checked' && !off.saved.start_gui_at_login && absent,
      { test_exe_path: value === expected, restored_on: restored.saved.start_gui_at_login, toggle_restored: checked, removed_when_off: absent })
  })

  if (!selected || selected === 'W25') await walk('W25', 'S12', async state => {
    const initial = await lanOn(state)
    const page = state.session.page
    const old = await call(state.session, 'RevealKey')
    if (old.error || !old.value) throw new Error(`test credential unavailable: ${old.error?.code}`)
    const show = page.getByRole('button', { name: 'Show', exact: true })
    const present = await show.count()
    if (present) await click(state.session, show)
    const revealDeadline = Date.now() + 5000
    let shown = false
    while (Date.now() < revealDeadline && !shown) {
      shown = (await page.getByTestId('key-value').innerText()).includes(old.value)
      if (!shown) await page.waitForTimeout(100)
    }
    await click(state.session, page.getByRole('button', { name: 'Reset key…' }))
    await click(state.session, page.getByRole('button', { name: 'Reset key', exact: true }))
    const renewed = await call(state.session, 'RevealKey')
    if (renewed.error || !renewed.value) throw new Error(`new test credential unavailable: ${renewed.error?.code}`)
    await click(state.session, page.getByRole('button', { name: 'Copy address and key', exact: true }))
    const copyDeadline = Date.now() + 5000
    let copied = ''
    while (Date.now() < copyDeadline && !copied.includes(renewed.value)) {
      copied = readCapture(state.run).filter(row => row.kind === 'copy').at(-1)?.value ?? ''
      if (!copied.includes(renewed.value)) await page.waitForTimeout(100)
    }
    await click(state.session, page.getByRole('switch', { name: 'Local network access' }))
    const off = await waitView(state.session, v => v.key?.lan_access === false)
    const copyGone = await page.getByRole('button', { name: 'Copy address and key', exact: true }).count() === 0
    record('W25', initial.key.lan_access && initial.key.present && present > 0 && shown && old.value !== renewed.value && copied.includes(renewed.value) && !copied.includes(old.value) && !off.key.lan_access && copyGone,
      { lan_on: initial.key.lan_access, key_present: initial.key.present, show_control: present > 0, reveal_shown: shown,
        document_hidden: await page.evaluate(() => document.hidden), key_changed: old.value !== renewed.value,
        copied_new_key: copied.includes(renewed.value), copied_old_key: copied.includes(old.value), copy_removed_when_off: copyGone,
        original_key_sha256: hash(old.value), new_key_sha256: hash(renewed.value) })
  })
} catch (error) {
  fatalStep = { error: String(error), folder: error?.failureFolder ?? null }
  console.error(`flow walk stopped: ${error}`)
} finally {
  await manifest.close()
  const artifact = join(paths.evidence, 'flows', `${selected ? selected : 'walk'}.json`)
  const summary = { selected: selected || names, fatal_step: fatalStep, results: Object.fromEntries(Object.entries(results).map(([name, row]) =>
    [name, { count: row.checks.length, passed: row.checks.filter(check => check.passed).length, failed: row.checks.filter(check => !check.passed).length, checks: row.checks }])),
    captures, rerun: `${selected ? `$env:FLOW_CASE='${selected}'; ` : ''}node gui/tests/v8/flows.mjs --job ${paths.job} --port ${paths.port}` }
  writeFileSync(artifact, JSON.stringify(summary, null, 2))
  console.log(artifact)
  if (fatalStep || Object.values(summary.results).some(row => row.failed)) process.exitCode = 1
}
