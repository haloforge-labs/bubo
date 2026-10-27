import { join } from 'node:path'

// A port is baked into each test exe's verified Wails copy. It is never a GUI
// command-line flag, so updater health launches retain the same CDP endpoint.
export function v8Config(args = process.argv.slice(2)) {
  let job = 'A0'
  let port = 9333
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--job') job = args[++i]
    else if (args[i] === '--port') port = Number(args[++i])
    else if (args[i] !== '--skip-frontend' && args[i] !== '--no-cdp') throw new Error(`unknown v8 option: ${args[i]}`)
  }
  if (!/^[A-Z][A-Z0-9]*$/.test(job)) throw new Error(`invalid v8 job id: ${job}`)
  if (!Number.isInteger(port) || port < 1024 || port > 65535 || port >= 8081 && port <= 8086) {
    throw new Error(`invalid v8 CDP port: ${port}`)
  }
  if (job !== 'A0' && port === 9333) throw new Error('non-A0 jobs must choose their own CDP port')
  const base = process.env.BUBO_TEST_EVIDENCE_DIR
  if (!base) throw new Error('Set BUBO_TEST_EVIDENCE_DIR to a scratch folder outside the repo')
  const evidence = join(base, 'v8', job)
  return {
    job, port, evidence, tree: join(evidence, 'tree'), build: join(evidence, 'build'),
    jobFolder: join(evidence, 'reports'),
    shots: join(base, 'shots', `v8-${job}`),
    cdpURL: `http://127.0.0.1:${port}`,
  }
}
