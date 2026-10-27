import { queryOptions, useQuery, type QueryClient } from '@tanstack/react-query'
import { useSyncExternalStore } from 'react'
import { z } from 'zod'
import { asGUIFailure } from '@/lib/gui-errors'
import * as App from '../../wailsjs/go/main/App'
import type { supervisor } from '../../wailsjs/go/models'
import { EventsOn } from '../../wailsjs/runtime/runtime'
import { queryClient } from '@/lib/query-client'

// Wails returns JSON records; its generated class conversion methods are not
// serialized. Derive the wire fields recursively instead of copying a View shape.
type Wire<T> = T extends readonly (infer Item)[]
  ? Wire<Item>[]
  : T extends object
    ? {
        [
          Key in keyof T as T[Key] extends (...args: never[]) => unknown
            ? never
            : Key
        ]: Wire<T[Key]>
      }
    : T
export type View = Wire<supervisor.View>
export const viewKey = ['view'] as const

export interface ViewTransport {
  getView: () => Promise<View>
}

const desktopTransport: ViewTransport = { getView: App.GetView }
let transport = desktopTransport

// Tests replay generated Views without installing Wails globals.
export function connectBindings(next: ViewTransport) {
  const previous = transport
  transport = next
  return () => {
    if (transport === next) transport = previous
  }
}

export function viewRefetchInterval(hidden: boolean) {
  return hidden ? 10_000 : 2_000
}

export function viewOptions(hidden = false) {
  return queryOptions({
    queryKey: viewKey,
    queryFn: () => transport.getView(),
    refetchInterval: viewRefetchInterval(hidden),
    refetchIntervalInBackground: true,
  })
}

export function useView() {
  const hidden = useSyncExternalStore(
    subscribeVisibility,
    () => document.hidden,
    () => false,
  )
  return useQuery(viewOptions(hidden))
}

function subscribeVisibility(changed: () => void) {
  document.addEventListener('visibilitychange', changed)
  return () => document.removeEventListener('visibilitychange', changed)
}

const actionErrorSchema = z.object({ code: z.string(), remedy: z.string() })
const actionResultSchema = z.object({ error: actionErrorSchema.nullish() })

// One error boundary for bindings and composed actions. Preserve typed CLI errors
// intact; native transport failures receive the shared GUI wording fallback.
export function requireActionSuccess<Result>(result: Result): Result {
  const parsed = actionResultSchema.safeParse(result)
  if (parsed.success && parsed.data.error) {
    if (result && typeof result === 'object' && 'error' in result)
      throw result.error
  }
  if (
    !parsed.success &&
    result &&
    typeof result === 'object' &&
    'error' in result &&
    result.error != null
  )
    throw {
      code: 'protocol_mismatch',
      remedy: 'The desktop action returned an invalid error record.',
    }
  return result
}

export function bindAction<Args extends unknown[], Result>(
  action: (...args: Args) => Promise<Result>,
  client: QueryClient = queryClient,
) {
  return async (...args: Args): Promise<Wire<Result>> => {
    try {
      // The native bridge serializes data, never generated class methods.
      return requireActionSuccess(await action(...args)) as Wire<Result>
    } catch (error) {
      if (actionErrorSchema.safeParse(error).success) throw error
      throw asGUIFailure({
        code: 'binding_unavailable',
        remedy:
          'The desktop connection did not complete this action. Try again.',
      })
    } finally {
      await client.invalidateQueries({ queryKey: viewKey })
    }
  }
}

// GetView is the sole read above. Every other generated method is wrapped here,
// preserving the generated contract for each user action.
export const actions = {
  AddModelFolder: bindAction(App.AddModelFolder),
  Approve: bindAction(App.Approve),
  BrowseDownloadFolder: bindAction(App.BrowseDownloadFolder),
  BrowseModelFile: bindAction(App.BrowseModelFile),
  BrowseModelFolder: bindAction(App.BrowseModelFolder),
  Cancel: bindAction(App.Cancel),
  CancelDownload: bindAction(App.CancelDownload),
  CancelUpdate: bindAction(App.CancelUpdate),
  CheckModel: bindAction(App.CheckModel),
  CheckUpdate: bindAction(App.CheckUpdate),
  ClearReleaseHistory: bindAction(App.ClearReleaseHistory),
  ConfirmDownload: bindAction(App.ConfirmDownload),
  CopyText: bindAction(App.CopyText),
  Decline: bindAction(App.Decline),
  Diagnose: bindAction(App.Diagnose),
  DismissError: bindAction(App.DismissError),
  DismissModelJob: bindAction(App.DismissModelJob),
  DismissUpdate: bindAction(App.DismissUpdate),
  Download: bindAction(App.Download),
  DownloadUpdate: bindAction(App.DownloadUpdate),
  Estimate: bindAction(App.Estimate),
  Exit: bindAction(App.Exit),
  InstallUpdate: bindAction(App.InstallUpdate),
  RecoverUpdate: bindAction(App.RecoverUpdate),
  Manage: bindAction(App.Manage),
  PatchPreferences: bindAction(
    (
      patch: Wire<supervisor.PreferencesPatch>,
    ): Promise<Wire<supervisor.ActionResult>> =>
      App.PatchPreferences(patch as supervisor.PreferencesPatch),
  ),
  Prepare: bindAction(App.Prepare),
  PreviewBundle: bindAction(App.PreviewBundle),
  RefreshModels: bindAction(App.RefreshModels),
  RemoveModelFolder: bindAction(App.RemoveModelFolder),
  RepairPackage: bindAction(App.RepairPackage),
  OpenModelFolder: bindAction(App.OpenModelFolder),
  OpenExternal: bindAction(App.OpenExternal),
  ResetKey: bindAction(App.ResetKey),
  Restart: bindAction(App.Restart),
  RevealKey: bindAction(App.RevealKey),
  SaveBundle: bindAction(App.SaveBundle),
  SelectInstance: bindAction(App.SelectInstance),
  SetVisible: bindAction(App.SetVisible),
  SetSwitchOverrides: bindAction(
    (
      patch: Wire<supervisor.SwitchOverridesPatch>,
    ): Promise<Wire<supervisor.ActionResult>> =>
      App.SetSwitchOverrides(patch as supervisor.SwitchOverridesPatch),
  ),
  Start: bindAction(App.Start),
  Stop: bindAction(App.Stop),
  TakeTrayAction: bindAction(App.TakeTrayAction),
}

// Names and payloads mirror gui/wails/events.go; pages only consume the View.
const events = {
  prepare: 'prepare:progress',
  download: 'download:progress',
  log: 'log:line',
  update: 'update:progress',
} as const

export function subscribeViewEvents(
  client: QueryClient = queryClient,
  subscribe: (
    name: string,
    receive: (payload: unknown) => void,
  ) => () => void = EventsOn,
) {
  function on<Payload>(
    name: string,
    reduce: (view: View, payload: Payload) => View,
  ) {
    // The supervisor validates CLI records before this trusted Wails boundary.
    return subscribe(name, (payload) => {
      client.setQueryData<View>(viewKey, (view) =>
        view ? reduce(view, payload as Payload) : undefined,
      )
    })
  }
  const stops = [
    on<NonNullable<View['progress']>>(events.prepare, (view, progress) => ({
      ...view,
      progress,
    })),
    on<NonNullable<NonNullable<View['models']>['download']>>(
      events.download,
      (view, download) =>
        view.models ? { ...view, models: { ...view.models, download } } : view,
    ),
    on<string>(events.log, (view, line) => {
      // During first discovery a log can announce a new Go startup phase.
      // Fetch that View at once even when the hidden window polls slowly.
      if (view.startup_phase === 'starting_cli')
        void client.invalidateQueries({ queryKey: viewKey })
      return { ...view, logs: [...view.logs, line].slice(-2000) }
    }),
    on<NonNullable<View['update']['progress']>>(
      events.update,
      (view, progress) => ({
        ...view,
        update: { ...view.update, progress },
      }),
    ),
  ]
  return () => stops.forEach((stop) => stop())
}
