import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { reachState, leaveState, waitView } from '../walk/state.mjs'
import { call, launchRun, loopbackManifest, makeRun, paths, readCapture, route, scenario, stopRun, view } from '../walk/session.mjs'

// G2 cases use the real bubo_test executable on the private desktop. The
// state helper owns the Job Object and restores the temporarily parked CLI.
const folder = join(paths.evidence, 'faults')
mkdirSync(folder, { recursive: true })
const selected = new Set((process.env.FAULT_CASES || 'F1a,F1b,F1c,F2,F4,F5,F6,F7a,F7b,F8,F9a,F9b,F10,F13a,F13b,F18,F23,F24,F25a,F25b').split(','))
const results = []
const manifest = await loopbackManifest()

function check(id, cause, next, observed, good) {
  const result = { id, cause, next, good, observed }
  results.push(result)
  writeFileSync(join(folder, `${id}.json`), JSON.stringify(result, null, 2))
  console.log(`${good ? 'PASS' : 'RED'} ${id}: ${cause}; next: ${next}`)
}

// "View logs" must lead somewhere: the View carries log lines (View.logs is
// a list of lines), or the CLI's own log file exists.
function logsAvailable(v) {
  const lines = v?.logs?.length ?? 0
  const path = v?.log_path ?? null
  return { lines, path, available: lines > 0 || (path !== null && existsSync(path)) }
}

async function capture(name, cases) {
  let state
  try {
    state = await reachState(name, manifest)
    const { session, run } = state
    if (name !== 'S1' && name !== 'S3') await session.page.waitForTimeout(2100)
    const snapshot = await view(session)
    const screen = await session.page.locator('body').innerText()
    const buttons = await session.page.locator('button').evaluateAll(nodes =>
      nodes.map(node => ({ text: node.textContent?.trim(), disabled: node.disabled, title: node.title, aria: node.getAttribute('aria-label') })))
    const record = { state: name, view: snapshot, screen, buttons, capture: readCapture(run), console: session.consoleLines,
      config: run.configPath, input_sha256: createHash('sha256').update(readFileSync(join(paths.tree, 'Bubo.Control.Web.exe'))).digest('hex') }
    writeFileSync(join(folder, `${name}.json`), JSON.stringify(record, null, 2))
    await cases(record, session)
  } finally {
    if (state) await leaveState(state)
  }
}

async function custom(id, options, action) {
  const run = makeRun(id, manifest, options)
  let session
  try {
    session = await launchRun(run)
    await waitView(session, v => v.initialized, 10000)
    await action(session, run)
    writeFileSync(join(folder, `${id}-capture.json`), JSON.stringify({ config: run.configPath,
      capture: readCapture(run), console: session.consoleLines }, null, 2))
  } finally {
    if (session) await stopRun(session)
  }
}

async function modelPlan(session) {
  const v = await view(session)
  const model = v.models?.catalogue?.find(m => m.download_enabled)
  if (!model) throw new Error('fault fixture has no downloadable catalogue model')
  const plan = await call(session, 'Download', model.id)
  return { model, plan }
}

async function actionLabels(session) {
  return session.page.locator('main button, main a').evaluateAll(nodes => nodes.map(n => n.textContent?.trim()).filter(Boolean))
}

async function modelFailure(code, sentence) {
  await custom(`F10-${code}`, { prepared: true, env: { BUBO_FAKE_MODEL_FAILURE: code } }, async session => {
    const { model, plan } = await modelPlan(session)
    if (plan.error) throw new Error(`model plan failed: ${JSON.stringify(plan.error)}`)
    const pending = call(session, 'ConfirmDownload', model.id)
    const v = await waitView(session, x => x.models?.download?.state === 'failed', 10000)
    await pending
    await route(session, '#/models')
    await session.page.waitForTimeout(2100)
    const screen = await session.page.locator('main').innerText()
    const buttons = await actionLabels(session)
    check(`F10-${code}`, `${code} stopped the model download`, 'Try again',
      { failure: v.models.download.error, screen, buttons, _source: 'fixture' },
      v.models.download.error?.code === code && screen.includes(sentence) && buttons.some(x => x.trim() === 'Try again'))
  })
}

async function cancelCase(id, cancelMode) {
  await custom(id, { prepared: true, env: { BUBO_FAKE_MODEL_CANCEL_MODE: cancelMode,
    BUBO_FAKE_PROGRESS_PAUSE_MS: '1000' } }, async (session, run) => {
    writeFileSync(join(run.fakeDir, 'model-steps'), '40')
    const { model, plan } = await modelPlan(session)
    if (plan.error) throw new Error(`model plan failed: ${JSON.stringify(plan.error)}`)
    const pending = call(session, 'ConfirmDownload', model.id)
    await waitView(session, v => v.models?.download?.state === 'running', 8000)
    const cancelled = await call(session, 'CancelDownload')
    if (cancelled.error) throw new Error(`CancelDownload failed: ${JSON.stringify(cancelled.error)}`)
    const interim = await waitView(session, v => v.models?.download?.state === 'cancelling' || v.models?.download?.state === 'cancelled', 4000)
    const late = id === 'F18' ? await waitView(session, v => v.models?.download?.bytes_done > interim.models.download.bytes_done, 4000) : null
    const final = await waitView(session, v => ['cancelled', 'failed'].includes(v.models?.download?.state), 7000)
    await pending.catch(() => {})
    check(id, id === 'F18' ? 'A late progress record arrives after Cancel' : 'The CLI exits nonzero after Cancel',
      'Cancelled', { interim: interim.models.download, late: late?.models?.download, final: final.models.download },
      interim.models.download.state === 'cancelling' &&
        (!late || late.models.download.state === 'cancelling' && late.models.download.message.startsWith('Cancelling.')) &&
        final.models.download.state === 'cancelled' && !final.models.download.error)
  })
}

async function missingCLIUpdateCases() {
  const folder = join(paths.evidence, 'faults', `update-fixture-${Date.now()}`)
  mkdirSync(folder, { recursive: true })
  const server = await loopbackManifest((req, res) => {
    const file = req.url?.endsWith('/bubo-release.json') ? join(folder, 'bubo-release.json') :
      req.url?.endsWith('.zip') ? join(folder, 'bubo-v0.1.1-windows-x64-gfx1151.zip') : null
    if (!file || !existsSync(file)) { res.writeHead(404); res.end(); return }
    const data = readFileSync(file)
    res.writeHead(200, { 'content-length': data.length, 'content-type': file.endsWith('.zip') ? 'application/zip' : 'application/json' })
    res.end(data)
  })
  const cli = join(paths.tree, 'bin', 'bubo.exe')
  const parked = `${cli}.A4-parked`
  let session
  try {
    execFileSync((process.env.BUBO_GO ?? 'go'),
      ['run', './v8/updaterfixture', '-folder', folder, '-origin', new URL(server.url).origin],
      { cwd: join(process.cwd(), 'gui', 'tests'), encoding: 'utf8', windowsHide: true })
    if (existsSync(parked)) throw new Error('previous A4 parked CLI was not restored')
    renameSync(cli, parked)
    const run = makeRun('F1c', server, { prepared: true })
    session = await launchRun(run)
    await waitView(session, v => v.initialized && v.state === 'Failed', 10000)
    await session.page.waitForTimeout(2100)
    const initial = await view(session)
    const checked = await call(session, 'CheckUpdate')
    const afterCheck = await view(session)
    const screen = await session.page.locator('body').innerText()
    if (selected.has('F1c')) check('F1c', 'The command-line tool is missing, but an update is available', 'Check for updates',
      { initial: initial.phase, checked, update: afterCheck.update, screen, manifest: join(folder, 'bubo-release.json') },
      !checked.error && afterCheck.update?.state === 'available')
    if (selected.has('F13a')) check('F13a', 'The command-line tool is missing while an update is available', 'Reinstall Bubo',
      { phase: afterCheck.phase, screen },
      afterCheck.phase === 'cli_missing' && screen.includes('Bubo needs to be reinstalled.') && screen.includes('Open download page'))
    if (selected.has('F13b')) {
      const download = await call(session, 'DownloadUpdate')
      const staged = await view(session)
      check('F13b', 'The command-line tool is missing while an update package is staged', 'Install update',
        { checked, download, update: staged.update, install: staged.actions?.install },
        !download.error && staged.update?.state === 'ready' && staged.actions?.install?.allowed === true)
    }
    writeFileSync(join(folder, 'capture.json'), JSON.stringify({ config: run.configPath,
      capture: readCapture(run), console: session.consoleLines }, null, 2))
  } finally {
    try {
      if (session) await stopRun(session)
    } finally {
      if (existsSync(parked)) renameSync(parked, cli)
      await server.close()
    }
  }
}

async function updateHistoryCase(id, contents, expectedCode) {
  const folder = join(paths.evidence, 'faults', `${id}-fixture-${Date.now()}`)
  mkdirSync(folder, { recursive: true })
  const server = await loopbackManifest((req, res) => {
    const file = req.url?.endsWith('/bubo-release.json') ? join(folder, 'bubo-release.json') :
      req.url?.endsWith('.zip') ? join(folder, 'bubo-v0.1.1-windows-x64-gfx1151.zip') : null
    if (!file || !existsSync(file)) { res.writeHead(404); res.end(); return }
    const data = readFileSync(file)
    res.writeHead(200, { 'content-length': data.length, 'content-type': file.endsWith('.zip') ? 'application/zip' : 'application/json' })
    res.end(data)
  })
  const root = join(paths.evidence, '.tree.bubo-update')
  const seen = join(root, 'seen.json')
  mkdirSync(root, { recursive: true })
  const old = existsSync(seen) ? readFileSync(seen) : null
  let session
  try {
    execFileSync((process.env.BUBO_GO ?? 'go'),
      ['run', './v8/updaterfixture', '-folder', folder, '-origin', new URL(server.url).origin],
      { cwd: join(process.cwd(), 'gui', 'tests'), encoding: 'utf8', windowsHide: true })
    writeFileSync(seen, contents)
    const run = makeRun(id, server, { prepared: true })
    session = await launchRun(run)
    await waitView(session, v => v.initialized, 10000)
    const failed = await call(session, 'CheckUpdate')
    const first = await view(session)
    if (failed.error?.code !== expectedCode) throw new Error(`${id} fault injection returned ${JSON.stringify(failed)}`)
    await route(session, '#/about')
    await session.page.waitForTimeout(2100)
    const before = await session.page.locator('main').innerText()
    const labels = await session.page.locator('[data-update] button').allTextContents()
    let next, recovered = false
    for (const raw of labels) {
      const label = raw.trim()
      if (!label) continue
      const button = session.page.getByRole('button', { name: label, exact: true }).first()
      if (!await button.isEnabled()) continue
      await button.click()
      await session.page.waitForTimeout(500)
      next = await call(session, 'CheckUpdate')
      const after = await view(session)
      if (!next.error && ['available', 'current'].includes(after.update?.state)) { recovered = true; break }
      await route(session, '#/about')
    }
    check(id, id === 'F25a' ? 'A withdrawn release is older than saved update history' : 'Saved update history is corrupt',
      'Clear saved update history, then check again',
      { first: first.update, before, labels, next, recovered, seen, manifest: join(folder, 'bubo-release.json') }, recovered)
    writeFileSync(join(folder, 'capture.json'), JSON.stringify({ config: run.configPath,
      capture: readCapture(run), console: session.consoleLines }, null, 2))
  } finally {
    try { if (session) await stopRun(session) } finally {
      if (old) writeFileSync(seen, old)
      else if (existsSync(seen)) unlinkSync(seen)
      await server.close()
    }
  }
}

try {
  if (selected.has('F1a') || selected.has('F1b')) await capture('S2', async r => {
    if (selected.has('F1a')) check('F1a', 'The command-line tool is missing', 'Reinstall Bubo',
      { phase: r.view.phase, status_label: r.view.status_label, failure: r.view.failure },
      r.view.phase === 'cli_missing' && r.view.status_label === 'Bubo needs to be reinstalled')
    if (selected.has('F1b')) check('F1b', 'The command-line tool is missing', 'Reinstall Bubo',
      { screen: r.screen, buttons: r.buttons },
      r.screen.includes("Bubo's command-line tool is missing. Reinstall Bubo."))
  })
  if (selected.has('F1c') || selected.has('F13a') || selected.has('F13b')) await missingCLIUpdateCases()
  if (selected.has('F2')) await capture('S3', async (r, session) => {
    await route(session, '#/')
    const screen = await session.page.locator('main').innerText()
    const actions = await actionLabels(session)
    check('F2', 'The command-line tool has not answered its first status request after 60 seconds', 'View logs',
      { startup_phase: r.view.startup_phase, screen, actions, logs: logsAvailable(r.view) },
      screen.includes('Bubo is taking longer than usual to start. It keeps trying.') &&
        actions.includes('View logs') && logsAvailable(r.view).available)
  })
  if (selected.has('F5')) await capture('S4', async (r, session) => {
    // The first capture can precede the first repaint; wait for the page.
    await session.page.getByText('Download the engine files').first().waitFor({ timeout: 10000 }).catch(() => {})
    const screen = await session.page.locator('body').innerText()
    check('F5', 'Engine files and a model are not ready', 'Download the engine files',
      { phase: r.view.phase, status_label: r.view.status_label, screen },
      r.view.phase === 'needs_engine' && r.view.status_label === 'Setup needed: download the engine files' &&
        screen.includes('Download the engine files'))
  })
  if (selected.has('F4')) await custom('F4', { prepared: true, mode: 'before-ready' }, async session => {
    const pending = call(session, 'Start')
    const v = await waitView(session, x => x.state === 'Failed', 12000)
    await pending.catch(() => {})
    await route(session, '#/')
    await session.page.waitForTimeout(2100)
    const screen = await session.page.locator('main').innerText()
    const buttons = await actionLabels(session)
    const logs = logsAvailable(await view(session))
    check('F4', 'The command-line tool exited while loading', 'View logs',
      { failure: v.failure, screen, buttons, logs },
      screen.includes('The server stopped unexpectedly.') && buttons.some(x => x.trim() === 'View logs') &&
        logs.available)
  })
  if (selected.has('F6')) await custom('F6', { prepared: true, env: { BUBO_FAKE_PORT_OCCUPIED: '1' } }, async session => {
    const pending = call(session, 'Start')
    const v = await waitView(session, x => x.state === 'Failed', 12000)
    await pending.catch(() => {})
    await route(session, '#/')
    await session.page.waitForTimeout(2100)
    const screen = await session.page.locator('main').innerText()
    const buttons = await actionLabels(session)
    check('F6', 'Another app is using the server address', 'Review connection settings',
      { failure: v.failure, screen, buttons, _source: 'fixture' },
      v.failure?.code === 'port_occupied' && screen.includes('Another app is using the server address') &&
        buttons.some(x => x.trim() === 'Review connection settings'))
  })
  if (selected.has('F7a') || selected.has('F7b')) await custom('F7',
    { prepared: true, env: { BUBO_FAKE_SCENARIO: scenario('no-fit') } }, async session => {
    const v = await waitView(session, x => x.initialized && x.state === 'Stopped', 10000)
    writeFileSync(join(folder, 'F7-view.json'), JSON.stringify(v, null, 2))
    await route(session, '#/')
    // The page can still show "Starting Bubo…" before its first View paint.
    await session.page.getByText("This model doesn't fit in memory at any context size.").first().waitFor({ timeout: 10000 }).catch(() => {})
    const screen = await session.page.locator('body').innerText()
    const buttons = await session.page.locator('button').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent?.trim() })))
    const estimate = v.models?.catalogue?.find(x => x.id === 'huihui-qwen38-flash-next-ud-q4-k-xl-v1')?.estimate?.memory_estimate
    const noContextFits = estimate?.fits === false && estimate.recommended_context?.by_sessions?.length > 0 &&
      estimate.recommended_context.by_sessions.every(x => x.fits === false)
    if (selected.has('F7a')) check('F7a', 'The selected model cannot fit in memory at any context size', 'Open Models',
      { noContextFits, fit: estimate?.fits, reason: estimate?.fit_reason, start: v.actions?.start },
      noContextFits &&
        v.actions?.start?.reason_code === 'memory_does_not_fit_any')
    if (selected.has('F7b')) check('F7b', 'The selected model cannot fit in memory at any context size', 'Open Models',
      { noContextFits, screen, buttons },
      noContextFits && screen.includes("This model doesn't fit in memory at any context size.") && buttons.some(b => b.text === 'Open Models'))
  })
  if (selected.has('F23')) await capture('S4', async (r, session) => {
    await route(session, '#/connect')
    const buttons = await session.page.locator('button').evaluateAll(nodes =>
      nodes.map(node => ({ text: node.textContent?.trim(), disabled: node.disabled, title: node.title, aria: node.getAttribute('aria-label') })))
    const copy = buttons.filter(b => /copy/i.test(`${b.text} ${b.aria}`) && /address/i.test(`${b.text} ${b.aria}`))
    check('F23', 'There is no client address while the server is stopped', 'Start the server',
      { copy, screen: await session.page.locator('main').innerText() },
      copy.length > 0 && copy.every(b => b.disabled && b.title?.includes('Start the server')))
  })
  if (selected.has('F8')) await custom('F8', {}, async (session, run) => {
    writeFileSync(join(run.fakeDir, 'volume-short'), '')
    const pending = call(session, 'Prepare')
    const v = await waitView(session, x => x.approval != null, 8000)
    const approved = await call(session, 'Approve')
    // Approval is accepted at once; the space failure arrives as the result
    // of the pending Prepare.
    const outcome = await Promise.race([pending, new Promise(resolve => setTimeout(() => resolve(null), 10000))])
    const code = outcome?.error?.code ?? (await view(session)).failure?.code ?? null
    await route(session, '#/')
    await session.page.getByText('There is not enough space on the destination drive').first().waitFor({ timeout: 5000 }).catch(() => {})
    const screen = await session.page.locator('main').innerText()
    const buttons = await actionLabels(session)
    check('F8', 'The destination drive lacks space before downloading', 'Try again',
      { approval: v.approval, preparation: v.preparation, approved, outcome, code, screen, buttons },
      v.preparation?.some(p => p.volumes?.some(x => x.free_bytes < x.required_bytes)) &&
        code === 'insufficient_space' &&
        screen.includes('There is not enough space on the destination drive') &&
        buttons.some(x => x.trim() === 'Try again'))
    await call(session, 'Cancel')
    await pending.catch(() => {})
  })
  if (selected.has('F9a')) await custom('F9a', { env: { BUBO_FAKE_ENGINE_STALL: '1' } }, async session => {
    const pending = call(session, 'Prepare').catch(error => ({ error: String(error) }))
    await waitView(session, v => v.approval != null, 8000)
    await call(session, 'Approve')
    await waitView(session, v => v.progress != null, 8000)
    await session.page.waitForTimeout(35000)
    const v = await view(session)
    await route(session, '#/')
    const screen = await session.page.locator('main').innerText()
    const actions = await actionLabels(session)
    check('F9a', 'The engine download stopped reporting progress', 'Try again',
      { phase: v.phase, failure: v.failure, screen, actions },
      v.failure?.code === 'download_stalled' && screen.includes('The download stopped making progress.') && actions.includes('Try again'))
    void pending
  })
  if (selected.has('F9b')) await custom('F9b', { prepared: true, env: { BUBO_FAKE_MODEL_STALL: '1' } }, async session => {
    const { model, plan } = await modelPlan(session)
    if (plan.error) throw new Error(`model plan failed: ${JSON.stringify(plan.error)}`)
    const pending = call(session, 'ConfirmDownload', model.id).catch(error => ({ error: String(error) }))
    await waitView(session, v => v.models?.download?.state === 'running', 8000)
    await session.page.waitForTimeout(35000)
    const v = await view(session)
    await route(session, '#/models')
    const screen = await session.page.locator('main').innerText()
    const actions = await actionLabels(session)
    check('F9b', 'The model download stopped reporting progress', 'Try again',
      { download: v.models?.download, screen, actions },
      v.models?.download?.error?.code === 'download_stalled' &&
        screen.includes('The download stopped making progress.') && actions.includes('Try again'))
    void pending
  })
  if (selected.has('F10')) for (const [code, sentence] of [
    ['transport_failed', 'Bubo could not download the files'],
    ['integrity_failed', 'The downloaded files did not pass verification'],
    ['storage_failed', 'Bubo could not save the files'],
  ]) await modelFailure(code, sentence)
  if (selected.has('F18')) await cancelCase('F18', 'late-progress')
  if (selected.has('F24')) await cancelCase('F24', 'exit-one')
  if (selected.has('F25a')) await updateHistoryCase('F25a', JSON.stringify({ version: '0.1.2', released: '2026-10-17T00:00:00Z', via: null }), 'stale_manifest')
  if (selected.has('F25b')) await updateHistoryCase('F25b', '{invalid json', 'filesystem')
} catch (error) {
  const fatal = { error: String(error), stack: error?.stack, failure_folder: error?.failureFolder }
  writeFileSync(join(folder, 'fatal.json'), JSON.stringify(fatal, null, 2))
  console.error(`FAULT WALK STOP: ${fatal.error}`)
  process.exitCode = 2
} finally {
  await manifest.close()
  const failed = results.filter(r => !r.good)
  const summary = { job: paths.job, port: paths.port, total: results.length, passed: results.length - failed.length,
    red: failed.length, results }
  writeFileSync(join(folder, 'summary.json'), JSON.stringify(summary, null, 2))
  console.log(`G2: ${summary.passed}/${summary.total} passed, ${summary.red} RED`)
  if (failed.length) process.exitCode = 1
}
