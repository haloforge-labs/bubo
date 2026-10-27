import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { call, launchRun, loopbackManifest, makeRun, paths, route, stopRun, view } from './session.mjs'

const name = process.env.WALK_STATE ?? 'S4'
const scenario = { S4: 'not-prepared', S7: 'stopped-saved-estimate', S8: 'memory-gpu-headroom', S9: 'memory-default-setup', S10: 'no-fit', S19: 'memory-smaller-context', S20: 'models-warnings' }[name]
const manifest = await loopbackManifest()
const run = makeRun(`inspect-${name}`, manifest, { scenario, prepared: name === 'S12' })
let session
try {
  session = await launchRun(run)
  await session.page.waitForTimeout(1500)
  if (name === 'S12') {
    console.log('Start', JSON.stringify(await call(session, 'Start')))
    await session.page.waitForFunction(async () => (await globalThis.go.main.App.GetView()).state === 'Ready', undefined, { timeout: 10000 })
  }
  const result = {}
  for (const hash of ['#/', '#/models', '#/settings', '#/connect', '#/logs', '#/about']) {
    await route(session, hash)
    result[hash] = {
      buttons: await session.page.locator('button').evaluateAll(nodes => nodes.map(n => ({ text: n.textContent?.trim(), title: n.title, aria: n.getAttribute('aria-label'), disabled: n.disabled, html: n.outerHTML.slice(0, 400) }))),
      links: await session.page.locator('a').evaluateAll(nodes => nodes.map(n => ({ text: n.textContent?.trim(), href: n.getAttribute('href'), aria: n.getAttribute('aria-label') }))),
      main: await session.page.locator('main').allTextContents(),
    }
  }
  result.view = await view(session)
  result.bound_methods = await session.page.evaluate(() => Object.keys(globalThis.go.main.App))
  const file = join(paths.evidence, 'walk', `inspect-${name}.json`)
  writeFileSync(file, JSON.stringify(result, null, 2))
  console.log(file)
} finally {
  if (session) await stopRun(session)
  await manifest.close()
}
