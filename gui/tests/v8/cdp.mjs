// The one place the v8 harness attaches to a GUI's WebView2 over CDP.
//
// A started GUI is not usable when something listens on the debugging port. The
// WebView2 browser process opens the port first and creates the page target later:
// 0.1 s later on an idle machine, several seconds later on a loaded one (E7,
// 2026-10-20: "CDP has no WebView2 page" on the new GUI after an update restart,
// with the browser listening and no page yet). Both "connection refused" and
// "listening, no page yet" therefore mean "not ready", and only the deadline ends
// the wait. Throwing at the first page-less attach killed the job (the helper and
// the new GUI with it) while the product still had 90 s to start.

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

// chromium: Playwright's chromium. alive(): called before every attempt; throw from
// it to stop at once (the GUI exited, or a desktop isolation violation).
// onWait(text): called when an attempt finds the GUI not ready, for traces.
// Returns { browser, page }. The caller closes the browser.
export async function attachToPage(chromium, url, { wait = 60000, alive = () => {}, onWait = () => {}, attachTimeout = 500, pause = 100 } = {}) {
  const end = Date.now() + wait
  let last = 'no attempt'
  while (Date.now() < end) {
    alive()
    let browser
    try { browser = await chromium.connectOverCDP(url, { timeout: attachTimeout }) }
    catch (error) {
      last = `${error?.message ?? error}`.split(/\r?\n/)[0]
      onWait(`not listening: ${last}`)
      await sleep(pause)
      continue
    }
    const page = browser.contexts()[0]?.pages()[0]
    if (page) return { browser, page }
    last = 'CDP listener has no WebView2 page yet'
    onWait(last)
    await browser.close().catch(() => {})
    await sleep(pause)
  }
  throw new Error(`CDP not usable within ${wait} ms: ${last}`)
}
