import { readFileSync } from 'node:fs'
import { afterEach, expect, test, vi } from 'vitest'
import {
  actions,
  bindAction,
  connectBindings,
  subscribeViewEvents,
  viewKey,
  viewOptions,
  viewRefetchInterval,
  type View,
} from '@/lib/bindings'
import { createQueryClient, queryClient } from '@/lib/query-client'
import { scenarioByName, scenarios } from '../../tests/scenarios'

const desktop = vi.hoisted(() => ({ call: vi.fn() }))
const declarations = readFileSync(
  new URL('../../wailsjs/go/main/App.d.ts', import.meta.url),
  'utf8',
)
const methods = [...declarations.matchAll(/export function (\w+)\(([^)]*)\)/g)]
const argsForMethod = new Map(
  methods.map(([, name, parameters]) => [
    name,
    parameters
      ? parameters.split(',').map((parameter) => {
          switch (parameter.split(':')[1]) {
            case 'string':
              return 'fixture-path'
            case 'boolean':
              return true
            case 'number':
              return 7
            case 'supervisor.Settings':
              return { ...scenarioByName('ready-idle').view.saved, version: 7 }
            case 'supervisor.PreferencesPatch':
              return { set: { lan_access: false }, clear: [] }
            case 'supervisor.SwitchOverridesPatch':
              return { set: [{ name: '--fixture', value: '0.75' }], clear: [] }
            default:
              throw new Error(`Uncovered generated input: ${parameter}`)
          }
        })
      : [],
  ]),
)
vi.mock('../../wailsjs/go/main/App', async (importOriginal) => {
  const original =
    await importOriginal<
      Record<string, (...args: unknown[]) => Promise<unknown>>
    >()
  return Object.fromEntries(
    Object.keys(original).map((name) => [
      name,
      (...args: unknown[]) => desktop.call(name, ...args),
    ]),
  )
})

afterEach(() => {
  vi.restoreAllMocks()
  desktop.call.mockReset()
  queryClient.clear()
})

test('GetView uses the generated binding by default', async () => {
  const view = scenarioByName('ready-idle').view
  desktop.call.mockResolvedValue(view)
  expect(await queryClient.fetchQuery(viewOptions())).toBe(view)
  expect(desktop.call).toHaveBeenCalledExactlyOnceWith('GetView')
})

test('the injectable transport returns a generated View and restores the desktop bridge', async () => {
  const client = createQueryClient()
  const view = scenarioByName('not-prepared').view
  const disconnect = connectBindings({ getView: async () => view })
  try {
    expect(await client.fetchQuery(viewOptions())).toBe(view)
  } finally {
    disconnect()
    client.clear()
  }
  desktop.call.mockResolvedValue(view)
  expect(await client.fetchQuery(viewOptions())).toBe(view)
  expect(desktop.call).toHaveBeenCalledExactlyOnceWith('GetView')
  client.clear()
})

test('visible and hidden polling use the architecture intervals', () => {
  expect(viewRefetchInterval(false)).toBe(2000)
  expect(viewRefetchInterval(true)).toBe(10000)
  expect(viewOptions(false).refetchInterval).toBe(2000)
  expect(viewOptions(true).refetchInterval).toBe(10000)
  expect(viewOptions().refetchIntervalInBackground).toBe(true)
})

test('every generated method has exactly one boundary wrapper or the GetView query', () => {
  const generated = methods.map((match) => match[1]).sort()
  expect([...Object.keys(actions), 'GetView'].sort()).toEqual(generated)
  console.log(
    `Binding coverage: ${generated.length} generated methods, ${Object.keys(actions).length} actions and 1 query.`,
  )
})

test.each(Object.entries(actions))(
  'action %s preserves typed failure and invalidates View',
  async (name, action) => {
    const result = {
      error: { code: 'fixture-rejected', remedy: 'Fixture remedy.' },
    }
    desktop.call.mockResolvedValue(result)
    const invalidate = vi
      .spyOn(queryClient, 'invalidateQueries')
      .mockResolvedValue()
    // Varying generated signatures share this runtime forwarding contract.
    const invoke = action as (...args: unknown[]) => Promise<unknown>
    const args = argsForMethod.get(name) ?? []
    await expect(invoke(...args)).rejects.toBe(result.error)
    expect(desktop.call).toHaveBeenCalledExactlyOnceWith(name, ...args)
    expect(invalidate).toHaveBeenCalledExactlyOnceWith({ queryKey: viewKey })
  },
)

test('typed actions retain signatures and invalidate on rejection', async () => {
  const client = createQueryClient()
  client.setQueryData(viewKey, scenarioByName('ready-idle').view)
  const error = { code: 'fixture-rejected', remedy: 'Try the fixture again.' }
  const action = bindAction(async (name: string, enabled: boolean) => {
    expect([name, enabled]).toEqual(['fixture', true])
    throw error
  }, client)
  await expect(action('fixture', true)).rejects.toBe(error)
  expect(client.getQueryState(viewKey)?.isInvalidated).toBe(true)
  client.clear()
})

test('an untyped action error payload cannot be reported as success', async () => {
  const client = createQueryClient()
  const action = bindAction(
    async () => ({ error: 'untyped bridge failure' }),
    client,
  )
  await expect(action()).rejects.toMatchObject({ code: 'protocol_mismatch' })
  client.clear()
})

function eventBridge() {
  const callbacks = new Map<string, (payload: unknown) => void>()
  const stops: ReturnType<typeof vi.fn>[] = []
  const subscribe = (name: string, callback: (payload: unknown) => void) => {
    callbacks.set(name, callback)
    const stop = vi.fn(() => {
      callbacks.delete(name)
    })
    stops.push(stop)
    return stop
  }
  return { callbacks, stops, subscribe }
}

test('all four streams project their generated payload into the existing View and unsubscribe', () => {
  const client = createQueryClient()
  const original = scenarioByName('models-download-in-progress').view
  const prepared = scenarioByName('preparing').view.progress
  const download = original.models?.download
  const update = scenarioByName('update-progress').view.update.progress
  expect(prepared).toBeTruthy()
  expect(download).toBeTruthy()
  expect(update).toBeTruthy()
  client.setQueryData(viewKey, original)
  const bridge = eventBridge()
  const stop = subscribeViewEvents(client, bridge.subscribe)
  expect([...bridge.callbacks.keys()]).toEqual([
    'prepare:progress',
    'download:progress',
    'log:line',
    'update:progress',
  ])
  bridge.callbacks.get('prepare:progress')?.(prepared)
  bridge.callbacks.get('download:progress')?.(download)
  bridge.callbacks.get('log:line')?.('Fixture log line')
  bridge.callbacks.get('update:progress')?.(update)
  const cached = client.getQueryData<View>(viewKey)
  expect(cached?.progress).toEqual(prepared)
  expect(cached?.models?.download).toEqual(download)
  expect(cached?.logs?.at(-1)).toBe('Fixture log line')
  expect(cached?.update.progress).toEqual(update)
  expect(cached?.update.state).toBe(original.update.state)
  expect(original.logs ?? []).not.toContain('Fixture log line')
  stop()
  expect(bridge.callbacks.size).toBe(0)
  bridge.stops.forEach((unsubscribe) =>
    expect(unsubscribe).toHaveBeenCalledOnce(),
  )
  client.clear()
})

test('events before GetView do not create a partial successful View', () => {
  const client = createQueryClient()
  const bridge = eventBridge()
  const stop = subscribeViewEvents(client, bridge.subscribe)
  bridge.callbacks.forEach((receive) => receive('Fixture early event'))
  expect(client.getQueryData(viewKey)).toBeUndefined()
  stop()
  client.clear()
})

test('event names match the authoritative Go event declarations', () => {
  const source = readFileSync(
    new URL('../../../events.go', import.meta.url),
    'utf8',
  )
  const names = [...source.matchAll(/\w+Event\s*=\s*"([^"]+)"/g)]
    .map((match) => match[1])
    .sort()
  const bridge = eventBridge()
  const stop = subscribeViewEvents(createQueryClient(), bridge.subscribe)
  expect([...bridge.callbacks.keys()].sort()).toEqual(names)
  stop()
})

test('the scenario loader contains all integrated generated Views with honest provenance', () => {
  expect(scenarios).toHaveLength(43)
  for (const name of [
    'onboarding-96-64',
    'onboarding-96-72',
    'onboarding-128-96',
  ]) {
    expect(scenarioByName(name).view._source).toBe('fixture')
  }
  for (const name of ['memory-gpu-headroom', 'memory-default-setup']) {
    expect(scenarioByName(name).view._source).toBe('fixture')
  }
  expect(
    scenarios.every(({ view }) =>
      ['recorded', 'fixture'].includes(view._source ?? ''),
    ),
  ).toBe(true)
  expect(scenarios.some(({ name }) => name === 'ready-busy')).toBe(true)
  expect(scenarios.some(({ name }) => name.includes('placeholder'))).toBe(false)
  expect(() => scenarioByName('missing-fixture')).toThrow('Missing scenario')
})
