import { createRequire } from 'node:module'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { assertIsolated, call, checkWalkError, launchRun, makeRun, readCapture, route, stopRun, view } from './session.mjs'
import { oracle } from './oracle.mjs'

const root = resolve(fileURLToPath(new URL('../../../../', import.meta.url)))
const frontend = join(root, 'gui', 'wails', 'frontend')
const typescript = createRequire(join(frontend, 'package.json'))('typescript')
const source = readFileSync(join(frontend, 'src', 'lib', 'gui-errors.ts'), 'utf8')
const compiled = typescript.transpileModule(source, { compilerOptions: { module: typescript.ModuleKind.ESNext, target: typescript.ScriptTarget.ES2022 } }).outputText
const { guiError } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

export function record(results, name, state, passed, detail) {
  const item = results[name] ??= { passed: true, checks: [] }
  item.passed &&= Boolean(passed)
  item.checks.push({ state, passed: Boolean(passed), detail })
}

const routes = [
  ['/', '#/'], ['Models', '#/models'], ['Connect', '#/connect'],
  ['Requests', '#/requests'], ['Logs', '#/logs'], ['Settings', '#/settings'], ['About', '#/about'],
]

const actionNames = new Set(['GetView', 'TestPanic'])
function controlPriority(text) {
  if (/^(?:Copy|Download$|Check for updates)/i.test(text)) return 0
  if (/^Download the engine files/i.test(text)) return 2
  if (/^(?:Browse|Select for Server)/i.test(text)) return 3
  if (/^(?:Exit|Restart server|Stop server|Start server|Install and restart|Recover)/i.test(text)) return 4
  return 1
}

const dialogSelector = '[role="dialog"], [role="alertdialog"]'

async function closeDialogs(session) {
  const isOpen = () => session.page.locator(dialogSelector).count().then(n => n > 0)
  if (!(await isOpen())) return { open: false }
  await session.page.keyboard.press('Escape')
  await session.page.waitForTimeout(300)
  if (!(await isOpen())) return { open: true, closed: true, how: 'Escape' }
  const cancel = session.page.locator(dialogSelector).getByRole('button', { name: /^(Cancel|Close|Not now|No|Keep running)$/ }).first()
  if (await cancel.count()) {
    const label = (await cancel.innerText()).trim()
    await cancel.click({ timeout: 3000 }).catch(() => {})
    await session.page.waitForTimeout(300)
    if (!(await isOpen())) return { open: true, closed: true, how: label }
  }
  return { open: true, closed: false, text: (await session.page.locator(dialogSelector).first().innerText().catch(() => '')).slice(0, 300) }
}

export async function controlCases(state, results, only = null, includeNav = !only, deferMissing = !only) {
  const { name, run, session } = state
  // The recorder lives in the page, so a page reload removes it. It is
  // reinstalled before every click, and a reload is reported as a finding:
  // no control should reload the app window.
  const instrument = () => session.page.evaluate(() => {
    if (globalThis.__walkActions) return false
    globalThis.__walkActions = []
    for (const [method, original] of Object.entries(globalThis.go.main.App)) {
      if (typeof original !== 'function' || method === 'TestPanic') continue
      globalThis.go.main.App[method] = async (...args) => {
        const value = await original(...args)
        // GetView returns a multi-megabyte View every poll. Retaining each
        // result made long control walks quadratic in memory and CDP traffic.
        globalThis.__walkActions.push({ method, args,
          value: method === 'GetView' ? undefined : value, at: performance.now() })
        return value
      }
    }
    return true
  })
  await instrument()
  let clicked = 0
  const deferred = []
  const controls = []
  for (const [label, hash] of routes) {
    session.step.control = `inventory ${label}`
    await route(session, hash)
    const found = await session.page.evaluate(() => {
      const seen = new Map()
      const indexes = new Map()
      return [...document.querySelectorAll('main button, main a[href]')].map(b => {
        const kind = b.tagName === 'A' ? 'link' : 'button'
        const index = indexes.get(kind) ?? 0
        indexes.set(kind, index + 1)
        const text = (b.innerText || b.getAttribute('aria-label') || b.title || '').trim()
        const key = `${kind}:${text}`
        const occurrence = seen.get(key) ?? 0
        seen.set(key, occurrence + 1)
        return { kind, index, text, occurrence, href: b.getAttribute('href'),
          disabled: b.disabled || b.getAttribute('aria-disabled') === 'true' || b.getAttribute('aria-selected') === 'true' }
      }).filter(b => !b.disabled)
    })
    for (const control of found) controls.push({ route: label, hash, ...control })
  }
  if (only) controls.splice(0, controls.length, ...controls.filter(item => only(item)))
  controls.sort((a, b) => controlPriority(a.text) - controlPriority(b.text))
  for (const item of controls) {
    if (session.child.exitCode !== null) break
    try {
      session.step.control = `${item.kind} ${item.text || `index ${item.index}`}`
      await route(session, item.hash)
      if (item.text) await session.page.waitForFunction(({ text, kind }) => [...document.querySelectorAll(kind === 'link' ? 'main a[href]' : 'main button')]
        .some(b => !b.disabled && (b.innerText || b.getAttribute('aria-label') || b.title || '').trim() === text), item,
      { timeout: 3000 })
      await instrument()
      const beforeView = await view(session)
      const before = await session.page.evaluate(() => ({ text: document.querySelector('main')?.innerText,
        markup: document.querySelector('main')?.outerHTML, hash: location.hash, actionCount: globalThis.__walkActions.length }))
      const captureCount = readCapture(run).length
      const handle = await session.page.evaluateHandle(({ kind, index, text, occurrence }) => {
        const all = [...document.querySelectorAll(kind === 'link' ? 'main a[href]' : 'main button')]
        const named = all.filter(b => (b.innerText || b.getAttribute('aria-label') || b.title || '').trim() === text)
        // A Copy button changes its own label to "Copied", shortening `named`.
        // Its DOM position stays stable, so prefer that position before a
        // same-label occurrence fallback.
        const atIndex = all[index]
        const b = (atIndex && (atIndex.innerText || atIndex.getAttribute('aria-label') || atIndex.title || '').trim() === text)
          ? atIndex : text ? named[occurrence] : atIndex
        if (!b || b.disabled || (b.innerText || b.getAttribute('aria-label') || b.title || '').trim() !== text) return null
        return b
      }, item)
      const button = handle.asElement()
      const pressed = Boolean(button)
      if (!pressed && deferMissing) {
        await handle.dispose()
        deferred.push(item)
        continue
      }
      const clickAt = await session.page.evaluate(() => performance.now())
      if (button && item.kind === 'link' && item.href === '#main-content') {
        await button.focus()
        await session.page.keyboard.press('Enter')
      } else if (button) await button.click({ timeout: 3000 })
      await handle.dispose()
      await session.page.waitForTimeout(500)
      await checkWalkError(session)
      const after = await session.page.evaluate(start => ({ text: document.querySelector('main')?.innerText,
        markup: document.querySelector('main')?.outerHTML, hash: location.hash, href: location.href,
        actions: globalThis.__walkActions?.slice(start) ?? null }), before.actionCount)
      if (after.actions === null) {
        record(results, `W2-reload:${item.route}:${item.text || `button-${item.index}`}`, name, false,
          { route: item.hash, kind: item.kind, index: item.index, href_after: after.href,
            note: 'the click reloaded the app page; its binding results were lost' })
        await instrument()
      }
      const actions = after.actions ?? []
      const captured = readCapture(run).length > captureCount
      const refreshCall = item.text === 'Refresh addresses' && actions.some(a => a.method === 'GetView' && a.at - clickAt < 250)
      const screenChanged = before.markup !== after.markup || before.hash !== after.hash
      const afterView = !screenChanged && session.child.exitCode === null ? await view(session) : null
      const viewChanged = afterView !== null && JSON.stringify(beforeView) !== JSON.stringify(afterView)
      const changed = screenChanged || viewChanged || session.child.exitCode !== null ||
        (item.kind === 'link' && item.href?.startsWith('https://') && captured)
      clicked += Number(pressed)
      record(results, `W2:${item.route}:${item.text || `button-${item.index}`}`, name, pressed && changed,
        { route: item.hash, kind: item.kind, index: item.index, pressed, changed, screenChanged, viewChanged, captured, refreshCall,
          action_methods: actions.map(a => a.method) })
      for (const action of actions) {
        if (actionNames.has(action.method)) continue
        const code = action.value?.error?.code
        // Two codes are caused by the walk's own setup, each checked against
        // it: only S16 serves an update feed, and a recorded folder may not
        // exist on this machine. Any other error code is RED.
        const folder = action.args?.[0]
        const environmental = (action.method === 'CheckUpdate' && code === 'network' && name !== 'S16') ||
          (action.method === 'OpenModelFolder' && code === 'model_folder_unavailable' && typeof folder === 'string' && !existsSync(folder))
        record(results, 'W3', name, !code || environmental || (action.method === 'PatchPreferences' && code === 'lan_unavailable'),
          { control: item.text, method: action.method, code: code ?? null, environmental })
      }
      // A dialog this click opened would cover every later control. The
      // user's way out is Escape or the dialog's own Cancel; a dialog that
      // neither closes is a finding.
      const dialog = await closeDialogs(session)
      if (dialog.open && !dialog.closed) record(results, `W2-dialog-stuck:${item.route}:${item.text || `button-${item.index}`}`, name, false, dialog)
    } catch (error) {
      if (session.fatalErrors.length) throw error
      if (deferMissing && String(error).includes('TimeoutError')) deferred.push(item)
      else record(results, `W2:${item.route}:${item.text || `button-${item.index}`}`, name, false,
        { route: item.hash, index: item.index, error: String(error) })
    }
  }
  if (includeNav && routes.length) {
    await route(session, '#/')
    // An earlier button can publish its dialog after the walk's 500 ms click
    // observation. Close it by the same user path before testing navigation.
    const beforeNav = await closeDialogs(session)
    if (beforeNav.open && !beforeNav.closed) record(results, 'W2-dialog-stuck:Nav', name, false, beforeNav)
    const nav = await session.page.evaluate(() => [...document.querySelectorAll('nav[aria-label="Main navigation"] a')]
      .map(a => ({ text: a.textContent?.trim() ?? '', href: a.getAttribute('href') ?? '' }))
      .filter(a => a.text && a.href.includes('#/') && !a.href.endsWith('#/')))
    for (const item of nav) {
      if (session.child.exitCode !== null) break
      try {
        session.step.control = `navigation ${item.text}`
        await route(session, '#/')
        const dialog = await closeDialogs(session)
        if (dialog.open && !dialog.closed) throw new Error(`dialog still blocks navigation: ${JSON.stringify(dialog)}`)
        const link = session.page.locator('nav[aria-label="Main navigation"] a').filter({ hasText: item.text }).first()
        await link.click({ timeout: 3000 })
        await session.page.waitForTimeout(100)
        await checkWalkError(session)
        const changed = await session.page.evaluate(href => location.hash === new URL(href, location.href).hash, item.href)
        clicked++
        record(results, `W2:Nav:${item.text}`, name, changed, { href: item.href, clicked: true, route_changed: changed })
      } catch (error) {
        if (session.fatalErrors.length) throw error
        record(results, `W2:Nav:${item.text}`, name, false, { href: item.href, error: String(error) })
      }
    }
  }
  record(results, 'W11', name, clicked > 0, { enabled_controls_found: controls.length, clicked })
  return deferred
}

function cliStrings(v) {
  const b1b = [v.remedy, v.failure?.remedy, v.plan_error?.remedy, v.models?.reason, v.update?.error?.remedy]
  const b4 = [
    ...((v.models?.catalogue ?? []).map(m => m.reason)),
    ...((v.models?.scan_warnings ?? []).map(w => w.reason)),
    ...((v.models?.scanned ?? []).map(m => m.reason)),
    v.npu_prefill_runtime?.reason,
  ]
  return { b1b: [...new Set(b1b.filter(s => typeof s === 'string' && s.length >= 12))],
    b4: [...new Set(b4.filter(s => typeof s === 'string' && s.length >= 12))] }
}

async function frame(session, hash) {
  session.step.control = `render ${hash}`
  await route(session, hash)
  return session.page.evaluate(() => {
    const main = document.querySelector('main')
    const clean = main?.cloneNode(true)
    clean?.querySelectorAll('details').forEach(node => node.remove())
    const labels = [...(clean?.querySelectorAll('[title], [aria-label]') ?? [])]
      .flatMap(node => [node.getAttribute('title'), node.getAttribute('aria-label')]).filter(Boolean)
    return {
      text: main?.innerText ?? '', outsideDetails: clean?.textContent ?? '', labels,
      buttons: [...(main?.querySelectorAll('button') ?? [])].map(b => ({ text: (b.innerText || b.getAttribute('aria-label') || '').trim(), disabled: b.disabled,
        title: b.title, aria: b.getAttribute('aria-label') })),
      activeNav: [...document.querySelectorAll('nav[aria-label="Main navigation"] a[aria-current="page"]')].map(a => a.textContent?.trim()),
      header: document.querySelector('header [role="status"]')?.textContent?.trim() ?? '',
      hash: location.hash,
    }
  })
}

export async function passiveCases(state, results) {
  const { name, run, session } = state
  const expected = oracle[name]
  const v = await view(session)
  const frames = new Map()
  for (const [label, hash] of routes) frames.set(label, await frame(session, hash))
  const home = frames.get('/')
  const about = frames.get('About')

  for (const [action, want] of Object.entries(expected.actions)) {
    const got = v.actions?.[action]
    record(results, 'W4', name, got?.allowed === want.allowed && (want.allowed || Boolean(got?.reason_code)) &&
      (!want.reason_code || got?.reason_code === want.reason_code), { action, want, got: got ?? null })
    if (!want.allowed && want.reason_code) {
      const target = ['check', 'install', 'recover'].includes(action) ? about : home
      const control = { prepare: /Download the engine files/, download_default: /Download (recommended|default) model/,
        start: /Start server/, stop: /Stop server/, restart: /Restart server/, check: /Check for updates/,
        install: /Install/, recover: /Recover/ }[action]
      if (target.buttons.some(b => control.test(b.text))) {
        const sentence = guiError({ code: want.reason_code, remedy: 'RAW_REMEDY_SENTINEL' }).message
        record(results, 'W4', name, target.outsideDetails.includes(sentence), { action, sentence, screen: target.outsideDetails.slice(0, 600) })
      }
    }
  }
  const restart = home.buttons.find(b => /Restart server/.test(b.text))
  if (!expected.actions.restart) record(results, 'W4', name, !restart || restart.disabled, { restart: restart ?? null })

  const menu = readCapture(run).filter(e => e.kind === 'tray_menu').map(e => JSON.parse(e.value)).at(-1)
  record(results, 'W5a', name, v.phase === expected.phase, { field: 'phase', expected: expected.phase, actual: v.phase })
  record(results, 'W5a', name, v.status_label === expected.label, { field: 'status_label', expected: expected.label, actual: v.status_label ?? null })
  const header = frames.get('Connect').header
  record(results, 'W5a', name, header === expected.label, { field: 'header', expected: expected.label, actual: header })
  record(results, 'W5a', name, menu?.Tip?.startsWith(`Bubo · ${expected.label}`) === true,
    { field: 'tray', expected: expected.label, actual: menu?.Tip ?? null })
  if (v.initialized && v.models?.catalogue?.length) {
    record(results, 'W5a', name, v.current_model?.display_name === expected.model,
      { field: 'current_model', expected: expected.model, actual: v.current_model ?? null })
  }
  if (expected.fit) {
    record(results, 'W5a', name, v.fit?.sentence === expected.fit, { field: 'fit', expected: expected.fit, actual: v.fit?.sentence ?? null })
    record(results, 'W5a', name, home.text.includes(expected.fit), { field: 'fit screen', expected: expected.fit, screen: home.text.slice(0, 900) })
  }
  for (const [label, hash] of routes) {
    const current = frames.get(label)
    record(results, 'W5b', name, current.activeNav.length === 1 && current.activeNav[0] === (label === '/' ? 'Server' : label),
      { route: hash, active: current.activeNav })
  }

  const raw = cliStrings(v)
  for (const [part, strings] of [['W6a', raw.b1b], ['W6b', raw.b4]]) {
    for (const text of strings) {
      const leaked = [...frames].filter(([, f]) => f.outsideDetails.includes(text)).map(([label]) => label)
      record(results, part, name, leaked.length === 0, { raw: text, routes: leaked })
    }
  }
  const jargon = /\b(?:ROCm|HIP|Vulkan|llama|ggml|GGUF|KV|quant|quantization|manifest|slot|schema|JSONL|stderr|stdout|PID|supervisor|registry|unsupported|not implemented|not attached|null|undefined|NaN)\b|\[object Object\]/gi
  for (const [label, f] of frames) {
    let words = `${f.outsideDetails} ${f.labels.join(' ')}`
    for (const rawText of [...raw.b1b, ...raw.b4]) words = words.replaceAll(rawText, '')
    words = words.replaceAll('Browse GGUF file', 'Browse model file')
    const matches = [...words.matchAll(jargon)].map(m => m[0])
    record(results, 'W7', name, matches.length === 0, { route: label, matches })
  }

  const offers = [...frames].flatMap(([label, f]) => f.buttons.filter(b => /Use [\d,]+ tokens/i.test(b.text)).map(b => ({ label, text: b.text })))
  const currentContext = v.saved?.context || v.saved_estimate?.memory_estimate?.context_tokens || v.memory_estimate?.context_tokens
  for (const offer of offers) {
    const value = Number(offer.text.match(/Use ([\d,]+) tokens/i)?.[1].replaceAll(',', ''))
    record(results, 'W9', name, Boolean(currentContext) && value < currentContext, { offer, current_context: currentContext ?? null })
  }
  if (name === 'S1' || name === 'S3') {
    const settings = frames.get('Settings').text
    record(results, 'W16', name, settings.includes('Loading settings…') && !/read-only in this build/i.test(settings),
      { loading: settings.includes('Loading settings…'), read_only: /read-only in this build/i.test(settings) })
  }
  return { view: v, frames: Object.fromEntries(frames), raw, offers }
}

export async function linkCase(state, results) {
  const { name, run, session } = state
  const attempts = ['file:///C:/Windows', 'ms-settings:', 'http://example.com']
  for (const target of attempts) {
    const before = readCapture(run).filter(e => e.kind === 'open_external').length
    const answer = await call(session, 'OpenExternal', target)
    const after = readCapture(run).filter(e => e.kind === 'open_external').length
    record(results, 'W12', name, answer.error?.code === 'internal_error' && before === after,
      { target, code: answer.error?.code ?? null, captures_before: before, captures_after: after })
  }
  const target = 'https://haloforgelabs.com/docs/'
  const answer = await call(session, 'OpenExternal', target)
  const captured = readCapture(run).some(e => e.kind === 'open_external' && e.value === target)
  record(results, 'W12', name, !answer.error && captured, { target, code: answer.error?.code ?? null, captured })
}

function logFiles(folder, depth = 0) {
  if (depth > 5 || !existsSync(folder)) return []
  return readdirSync(folder, { withFileTypes: true }).flatMap(entry => {
    const path = join(folder, entry.name)
    if (entry.isDirectory()) return entry.name === 'webview2' ? [] : logFiles(path, depth + 1)
    return entry.isFile() && /\.log$/i.test(entry.name) ? [path] : []
  })
}

export async function logCase(state, results, updater) {
  const { name, run, session } = state
  const part = updater ? 'W8b' : 'W8a'
  if (updater && !(await view(session)).update?.error) {
    const checked = await call(session, 'CheckUpdate')
    if (!checked.error) throw new Error('W8b fixture did not produce an updater error')
  }
  await route(session, updater ? '#/about' : '#/')
  const v = await view(session)
  const code = updater ? v.update?.error?.code : v.failure?.code
  const expectedAction = code ? guiError({ code, remedy: 'RAW_REMEDY_SENTINEL' }).action : ''
  const links = session.page.getByRole('link', { name: 'View logs', exact: true })
  const count = await links.count()
  if (!count) {
    record(results, part, name, expectedAction !== 'View logs', { code, expected_action: expectedAction, view_logs_links: 0 })
    return
  }
  const first = links.first()
  await first.click()
  await session.page.waitForFunction(() => location.hash === '#/logs', undefined, { timeout: 3000 })
  const screen = await session.page.locator('main').innerText()
  const candidates = logFiles(run.local).map(path => ({ path, text: readFileSync(path, 'utf8') }))
  const relevant = candidates.filter(file => code && file.text.includes(code))
  const shown = relevant.some(file => file.text.split(/\r?\n/).some(line => line.length >= 12 && screen.includes(line)))
  record(results, part, name, Boolean(code) && relevant.length > 0 && shown,
    { code: code ?? null, route: await session.page.evaluate(() => location.hash), files: candidates.map(file => file.path),
      relevant: relevant.map(file => file.path), shown })
}

export function liveLogCase(state, frames, results) {
  const source = readFileSync(join(root, 'gui', 'tests', 'fakecli', 'recorded-server-info.log'), 'utf8')
  const first = source.split(/\r?\n/)[0]
  const file = join(state.run.local, 'Bubo', 'gui', 'logs', 'stderr.0.log')
  const fileOK = existsSync(file) && readFileSync(file, 'utf8').includes(first)
  const shown = frames.Logs?.text.includes(first) === true
  record(results, 'W17', state.name, fileOK && shown, { first_line: first, file, file_exists: existsSync(file), shown })
}

export async function settingsRestartCase(state, results) {
  const { name, run, session } = state
  const argvFile = join(run.fakeDir, 'argv.json')
  const before = existsSync(argvFile) ? JSON.parse(readFileSync(argvFile, 'utf8')) : null
  await route(session, '#/settings')
  const saved = await call(session, 'PatchPreferences', { set: { port: 19135 }, clear: [] })
  const stored = (await view(session)).saved?.port === 19135
  const restart = await call(session, 'Restart')
  const until = Date.now() + 12000
  let after
  while (Date.now() < until) {
    after = existsSync(argvFile) ? JSON.parse(readFileSync(argvFile, 'utf8')) : null
    if (JSON.stringify(after) !== JSON.stringify(before) && after?.includes?.('19135')) break
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  record(results, 'W10', name, !saved.error && stored && !restart.error && after?.includes?.('19135') === true,
    { save_error: saved.error ?? null, stored, restart_error: restart.error ?? null, before, after })
}

export async function nativeCloseCase(state, results) {
  const { session, run, name } = state
  const cdp = await session.page.context().newCDPSession(session.page)
  await cdp.send('Debugger.enable')
  try {
    await cdp.send('Debugger.pause')
    session.child.closeWindow()
    const until = Date.now() + 4000
    while (Date.now() < until && session.child.exitCode === null) {
      assertIsolated(session.child)
      await new Promise(resolve => setTimeout(resolve, 100))
    }
    record(results, 'W15', name, session.child.exitCode !== null,
      { exit_code: session.child.exitCode, capture: readCapture(run).filter(e => ['quit', 'hide'].includes(e.kind)) })
  } finally {
    if (session.child.exitCode === null) await cdp.send('Debugger.resume').catch(() => {})
    await cdp.detach().catch(() => {})
  }
}

export async function panicCase(state, results) {
  const { session, run, name } = state
  const trigger = session.page.evaluate(() => globalThis.go.main.App.TestPanic()).catch(error => String(error))
  const until = Date.now() + 5000
  let boxes = []
  while (Date.now() < until) {
    assertIsolated(session.child)
    boxes = readCapture(run).filter(e => e.kind === 'message_box')
    if (boxes.length || session.child.exitCode !== null) break
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  record(results, 'I13', name, boxes.length === 1 && boxes[0].text === "Bubo hit a problem it can't recover from and must close.",
    { boxes, exit_code: session.child.exitCode })
  await Promise.race([trigger, new Promise(resolve => setTimeout(resolve, 100))])
}

export async function renderRateCase(manifest, results) {
  const run = makeRun('walk-W13', manifest, { scenario: 'not-prepared',
    env: { BUBO_FAKE_PROGRESS_REPLAY_COUNT: '180', BUBO_FAKE_PROGRESS_PAUSE_MS: '50' } })
  let session
  try {
    session = await launchRun(run)
    await session.page.context().addInitScript({ content: `
      globalThis.__walkCommits = [];
      globalThis.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
        supportsFiber: true, inject() { return 1 },
        onCommitFiberRoot() { globalThis.__walkCommits.push(performance.now()) },
        onCommitFiberUnmount() {}, onPostCommitFiberRoot() {}
      };
    ` })
    await session.page.reload()
    await session.page.waitForFunction(() => Boolean(globalThis.go?.main?.App?.GetView), undefined, { timeout: 5000 })
    const until = Date.now() + 7000
    while (Date.now() < until && !(await view(session)).initialized) await new Promise(resolve => setTimeout(resolve, 100))
    const pending = call(session, 'Prepare')
    while (Date.now() < until && !(await view(session)).approval) await new Promise(resolve => setTimeout(resolve, 100))
    await route(session, '#/')
    await session.page.evaluate(() => { globalThis.__walkCommits.length = 0 })
    const approved = await call(session, 'Approve')
    await pending
    await session.page.waitForTimeout(5500)
    const times = await session.page.evaluate(() => globalThis.__walkCommits)
    const finalView = await view(session)
    const screen = await session.page.locator('main').innerText()
    const maximum = Math.max(0, ...times.map(start => times.filter(t => t >= start && t < start + 1000).length))
    record(results, 'W13', 'S5', !approved.error && maximum <= 10 && times.length > 0,
      { maximum_renders_per_second: maximum, cap: 10, renders: times.length, approval_error: approved.error ?? null,
        view_state: finalView.state, progress: finalView.progress ?? null, screen: screen.slice(0, 400) })
  } catch (error) {
    if (session?.fatalErrors.length) throw error
    record(results, 'W13', 'S5', false, { fixture_error: String(error) })
  }
  finally { if (session) await stopRun(session) }
}
