import type { actions, View } from '@/lib/bindings'
import { bytes } from '@/lib/format'
export { bytes, recommendedModel } from '@/lib/format'

// The space a drive is short by: GiB with one decimal, rounded up to 0.1 GiB so
// it never reads 0.0; MiB only under 0.1 GiB.
export function shortBy(shortfall: number) {
  const gib = shortfall / 1024 ** 3
  if (gib < 0.1)
    return bytes(Math.ceil(shortfall / 1024 ** 2) * 1024 ** 2, {
      unit: 'MiB',
      digits: 0,
    })
  return bytes(Math.ceil(gib * 10) * 0.1 * 1024 ** 3, {
    unit: 'GiB',
    digits: 1,
  })
}

// The shortfall the way the screen reads: the difference of the two figures as
// printed ("110.6 GiB needed, 100.0 GiB free" is "10.6 GiB"), never a raw byte
// difference that rounds to another number. At least 0.1 GiB, because the CLI
// said the space is short.
export function shortByShown(required: number, free: number) {
  const unit = required >= 1024 ** 3 ? 'GiB' : 'MiB'
  const size = unit === 'GiB' ? 1024 ** 3 : 1024 ** 2
  if (required < 1024 ** 2) return shortBy(Math.max(0, required - free))
  const tenths = Math.max(
    1,
    Math.round((required / size) * 10) - Math.round((free / size) * 10),
  )
  return `${(tenths / 10).toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })} ${unit}`
}

export type ModelPlan = NonNullable<
  NonNullable<NonNullable<View['models']>['download']>['plan']
>

export function packageStatus(state: string) {
  switch (state) {
    case 'prepared':
      return {
        word: 'On this computer',
        action: 'check',
        actionLabel: 'Check or repair files',
      } as const
    case 'partial':
      return {
        word: 'Incomplete',
        action: 'repair',
        actionLabel: 'Check or repair files',
      } as const
    case 'not_prepared':
      return {
        word: 'Not downloaded',
        action: 'download',
        actionLabel: 'Download',
      } as const
    default:
      return {
        word: state || 'Status unavailable',
        action: 'none',
        actionLabel: '',
      } as const
  }
}

// Windows paths compare without case and with either slash.
export function samePath(a?: string | null, b?: string | null) {
  const clean = (path: string) => path.replaceAll('/', '\\').toLowerCase()
  return !!a && !!b && clean(a) === clean(b)
}

export function modelPreferencePatch(
  kind: 'file' | 'manifest',
  path: string,
): Parameters<typeof actions.PatchPreferences>[0] {
  return kind === 'file'
    ? { set: { model_path: path }, clear: ['manifest_path'] }
    : { set: { manifest_path: path }, clear: ['model_path'] }
}

export function activeModelChoice(view: View) {
  const identity = view.current_model
  if (!identity) return undefined
  const choices = modelChoices(view)
  // The running package is named by its path: the same package can be in two
  // folders. The catalogue id only decides when no path was reported.
  return (
    choices.find((choice) =>
      choice.kind === 'manifest'
        ? samePath(choice.path, identity.manifest_path)
        : samePath(choice.path, identity.model_path),
    ) ??
    choices.find(
      (choice) =>
        choice.kind === 'manifest' && choice.id === identity.catalogue_id,
    )
  )
}

// The models list is the one source of prepared models: every downloaded
// catalogue package, not only the compiled default that status reports. Status
// stands in only until the first list arrives.
export function preparedRows(view: View) {
  return view.models?.prepared_models ?? view.prepared_models ?? []
}

// How a prepared row starts: its manifest, else its model file (CLI-CONTRACT).
export function preparedStart(row: ReturnType<typeof preparedRows>[number]) {
  if (row.manifest_path)
    return { kind: 'manifest' as const, path: row.manifest_path }
  if (row.model_path) return { kind: 'file' as const, path: row.model_path }
  return undefined
}

// One rule for every action on a model: pass the row's path. A package is
// estimated by its folder, a model file by itself, never by the catalogue id.
export function estimateSubject(start: {
  kind: 'file' | 'manifest'
  path: string
}) {
  return start.kind === 'manifest'
    ? start.path.replace(/[\\/][^\\/]*$/, '')
    : start.path
}

export function modelChoices(view: View) {
  const choices = [
    ...preparedRows(view).flatMap((model) => {
      const start = preparedStart(model)
      return model.prepared_state === 'prepared' && start
        ? [
            {
              name: model.display_name,
              path: start.path,
              id: model.id,
              subject: estimateSubject(start),
              kind: start.kind,
              size: model.size_bytes,
            },
          ]
        : []
    }),
    ...(view.models?.scanned ?? [])
      .filter((model) => model.runnable)
      .map((model) => ({
        name: model.model_name || model.file_name,
        path: model.path,
        id: model.path,
        subject: model.path,
        kind: 'file' as const,
        size: model.size_bytes,
      })),
  ]
  // The same package can sit in two folders, and its name alone does not tell
  // the two copies apart: a repeated name carries "drive:\…\folder", the drive
  // first so a closed select cannot cut it off. When two copies still read the
  // same, one more folder is taken from the right.
  const group = new Map<string, number[]>()
  choices.forEach((choice, index) =>
    group.set(choice.name, [...(group.get(choice.name) ?? []), index]),
  )
  const names = choices.map((choice) => choice.name)
  for (const indexes of group.values()) {
    if (indexes.length < 2) continue
    const folders = indexes.map((index) => copyFolder(choices[index]!))
    const deepest = Math.max(...folders.map((folder) => folder.length))
    let depth = 1
    let labels = folders.map((folder) => copyLabel(folder, depth))
    while (new Set(labels).size < labels.length && depth < deepest) {
      depth += 1
      labels = folders.map((folder) => copyLabel(folder, depth))
    }
    indexes.forEach((index, at) => {
      names[index] = `${labels[at]} · ${choices[index]!.name}`
    })
  }
  return choices.map((choice, index) => ({ ...choice, name: names[index]! }))
}

// The folder segments of a copy: the package folder, or the folder of a loose file.
function copyFolder(choice: { subject: string; kind: string }) {
  const parts = choice.subject.split(/[\\/]+/).filter(Boolean)
  return choice.kind === 'file' ? parts.slice(0, -1) : parts
}

// "C:\…\leaf": the drive, then the last `depth` folders.
function copyLabel(folder: string[], depth: number) {
  const drive = /^[A-Za-z]:$/.test(folder[0] ?? '') ? folder[0]! : ''
  const rest = drive ? folder.slice(1) : folder
  const tail = rest.slice(-depth)
  return `${drive}\\${tail.length < rest.length ? '…\\' : ''}${tail.join('\\')}`
}
