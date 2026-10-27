// node --test gui/tests/v8/cdp.test.mjs
// Gate for the CDP attach: a listener without a page is "not ready yet", not a
// failure. Before attachToPage the harness threw 'CDP has no WebView2 page' at the
// first page-less attach (E6 P1 runs, E7 load6 run 9).
import assert from 'node:assert/strict'
import test from 'node:test'
import { attachToPage } from './cdp.mjs'

const fakeBrowser = (pages, closed) => ({ contexts: () => [{ pages: () => pages }], close: async () => { closed.count++ } })

test('waits through refused connections and a page-less listener, then returns the page', async () => {
  const closed = { count: 0 }
  const page = { name: 'page' }
  const script = ['refused', 'refused', 'no-page', 'no-page', 'no-page', 'page']
  let attempts = 0
  const chromium = { connectOverCDP: async () => {
    const step = script[attempts++]
    if (step === 'refused') throw new Error('browserType.connectOverCDP: connect ECONNREFUSED 127.0.0.1:9349\nCall log: ...')
    return fakeBrowser(step === 'page' ? [page] : [], closed)
  } }
  const waits = []
  const got = await attachToPage(chromium, 'http://127.0.0.1:9349', { wait: 5000, pause: 1, onWait: text => waits.push(text) })
  assert.equal(got.page, page)
  assert.equal(attempts, 6)
  assert.equal(closed.count, 3, 'each page-less attach is closed before the next try')
  assert.equal(waits.filter(w => w.includes('no WebView2 page')).length, 3)
})

test('a listener that never makes a page ends at the deadline and says so', async () => {
  const closed = { count: 0 }
  const chromium = { connectOverCDP: async () => fakeBrowser([], closed) }
  await assert.rejects(attachToPage(chromium, 'http://127.0.0.1:9349', { wait: 60, pause: 5 }),
    /CDP not usable within 60 ms: CDP listener has no WebView2 page yet/)
  assert.ok(closed.count >= 1)
})

test('alive() stops the wait at once', async () => {
  let calls = 0
  const chromium = { connectOverCDP: async () => { throw new Error('refused') } }
  await assert.rejects(attachToPage(chromium, 'u', { wait: 5000, pause: 1, alive: () => { if (++calls === 3) throw new Error('GUI exited 2 before CDP') } }),
    /GUI exited 2 before CDP/)
  assert.equal(calls, 3)
})
