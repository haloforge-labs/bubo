import { QueryClientProvider } from '@tanstack/react-query'
import { createMemoryHistory, RouterProvider } from '@tanstack/react-router'
import { afterEach } from 'vitest'
import { render } from 'vitest-browser-react'
import { createAppRouter } from '@/router'
import { connectBindings, viewKey, type View } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'
import { scenarioByName } from '../../tests/scenarios'
import runtimePlan from '../../tests/runtime-plan.json'
import laterRequest from '../../../../tests/scenarios/recordings/lrel8/status-0029-002905.317.json'
import originalPlan from '@/lib/model-plan-test-data'
import flashShortPlan from '../../tests/flash-short-plan.json'
export { default as flashShortPlanFixture } from '../../tests/flash-short-plan.json'
export { default as gemmaQatPlanFixture } from '../../tests/gemma-qat-plan.json'
export { default as lowerMemoryShortPlanFixture } from '../../tests/lower-memory-short-plan.json'
export { default as lowerMemoryPlanFixture } from '../../tests/lower-memory-plan.json'
export { default as legacyProfile } from '../../tests/legacy-profile.json'

const cleanups: (() => void)[] = []
afterEach(() =>
  cleanups
    .splice(0)
    .reverse()
    .forEach((cleanup) => cleanup()),
)

export function coreView(name = 'ready-idle'): View {
  return structuredClone(scenarioByName(name).view)
}

// A stopped server with the download folder facts Go reports (E19): the
// default folder on C: with 41 GiB free, or a chosen folder on its own drive.
export function folderView(folder?: string): View {
  const view = coreView('stopped')
  const GiB = 1024 ** 3
  view.download_folder = folder
    ? {
        path: folder,
        is_default: false,
        drive: folder.slice(0, 2),
        free_bytes: 892 * GiB,
        change: { allowed: true, reason_code: '', reason: '' },
      }
    : {
        path: 'C:\\Users\\fixture\\AppData\\Local\\Bubo\\models',
        is_default: true,
        drive: 'C:',
        free_bytes: 41 * GiB,
        change: { allowed: true, reason_code: '', reason: '' },
      }
  return view
}

type Row = NonNullable<View['models']>['prepared_models'][number]

function row(base: Row, folder: string, state = 'prepared'): Row {
  return {
    ...base,
    manifest_path: `${folder}\\bubo-package.json`,
    prepared_state: state,
  }
}

// Two folders hold the same package (the CLI returns two rows), the first one in
// the default folder and running state set by the caller.
export function packagesView(): { view: View; rows: Row[] } {
  const view = coreView('stopped')
  const base = coreView('ready-idle').models!.prepared_models[0]!
  const rows = [
    {
      ...row(
        base,
        'C:\\Users\\fixture\\AppData\\Local\\Bubo\\models\\' + base.id,
      ),
      repair_output: 'C:\\Users\\fixture\\AppData\\Local\\Bubo',
    },
    {
      ...row(base, 'D:\\Bubo downloads\\models\\' + base.id),
      repair_output: 'D:\\Bubo downloads',
    },
  ]
  view.models!.prepared_models = rows
  view.models!.scanned = [] // a world of two package folders and no loose file
  view.prepared_models = rows
  view.saved.manifest_path = undefined
  view.current_model = {
    catalogue_id: base.id,
    display_name: base.display_name,
    model_path: '',
    manifest_path: rows[0]!.manifest_path!,
  }
  return { view, rows }
}

// A second, earlier completion for showing time-separated request points.
export function withEarlierRequest(view: View): View {
  const earlier = coreView('ready-busy').requests[0]!
  const later = laterRequest.payload.requests.find((row) => row.id === 'r3')!
  view.requests = [structuredClone(earlier), { ...earlier, ...later }]
  view.last_completed_request = structuredClone(
    laterRequest.payload.last_completed_request,
  )

  return view
}

// The recorded plan of the default model. notInstalled gives it to the first
// downloadable catalogue entry that is not on this computer, as a Discover row.
export function recordedPlan(notInstalled = false) {
  if (notInstalled) {
    const plan = structuredClone(
      originalPlan.plan,
    ) as unknown as View['preparation'][number]
    return { ...plan, licence: plan.licence! }
  }
  const plan = coreView('preparing').preparation[0]!
  const entry = coreView().models!.catalogue[0]!
  return { ...plan, id: entry.id, kind: 'download', licence: entry.licence }
}

export function recordedRuntimePlan() {
  // The lifecycle state can be arranged, but the plan is the CLI's real one.
  return structuredClone(
    runtimePlan.plan,
  ) as unknown as View['preparation'][number]
}

export function mountCorePage(
  path: '/' | '/models' | '/onboarding' | '/settings',
  initial: View,
) {
  let current = initial
  const client = createQueryClient()
  client.setQueryData(viewKey, current)
  const disconnect = connectBindings({ getView: async () => current })
  const router = createAppRouter()
  router.update({ history: createMemoryHistory({ initialEntries: [path] }) })
  const screen = render(
    <div id="shell-capture">
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </div>,
  )
  cleanups.push(() => {
    disconnect()
    client.clear()
  })
  return {
    client,
    screen,
    replaceView: (view: View) => {
      current = view
      client.setQueryData(viewKey, view)
    },
  }
}

// The CLI's actual C: plans share one destination volume and one safety reserve.
export function combinedShortSpaceView() {
  const view = coreView('not-prepared')
  const runtime = recordedRuntimePlan()
  const model = {
    ...(structuredClone(
      flashShortPlan.plan,
    ) as unknown as View['preparation'][number]),
    kind: 'model',
  }
  view.preparation = [runtime, model].map((plan) => ({
    ...plan,
    volumes: plan.volumes.map((volume) => ({
      ...volume,
      name: Object.values(plan.paths)[0]!.slice(0, 3),
    })),
  }))
  view.approval = {
    plan_digest: runtime.plan_digest,
    components: ['runtime', 'model'],
  }
  return view
}
