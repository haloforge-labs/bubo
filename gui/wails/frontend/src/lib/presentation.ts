import type { View } from '@/lib/bindings'
import { bytes } from '@/lib/format'

type Row = View['switches'][number]
// Presentation overrides stay in the GUI: release CLI binaries and wire values are unchanged.
// short is a hint of at most three words beside the control; help is the full
// text, shown in the row's About popup.
const presentation: Record<
  string,
  { label?: string; short?: string; help?: string }
> = {
  '--seed': {
    short: 'Repeatable',
    help: 'A fixed seed with the same prompt and settings gives the same answer. A request can still send its own seed.',
  },
  '--profile': {
    short: 'Engine tuning',
    help: "How Bubo tunes the engine. Default is recommended; Vanilla turns Bubo's tuning off.",
  },
  '--top-p': {
    help: 'Nucleus sampling: only the most likely next tokens whose probabilities add up to this share are considered. Lower values make answers more focused.',
  },
  '--top-k': {
    help: 'Limit how many of the most likely next tokens Bubo considers. Use the default for normal use.',
  },
  '--kv-cache-type': {
    label: 'KV cache',
    help: 'The KV cache holds the conversation. Q8: the KV cache takes about half the memory, so more or longer conversations fit; replies can differ slightly from f16. Q8 is opt in and lossy; existing caches rebuild when the format changes. Used after the server restarts.',
  },
  '--temperature': { short: 'Randomness' },
  '--repeat-last-n': { short: 'Repeat window' },
  '--min-keep': { short: 'Tokens retained' },
  '--reasoning-effort': { short: 'Thinking depth' },
  '--served-model-name': { short: 'Name clients see' },
  '--npu-prefill': {
    help: 'Use the NPU to help read long prompts. Off is recommended in 0.1.1.',
  },
  '--refusal-direction': {
    help: 'Removes refusals while the model writes, inside the engine. Off by default. It works only with models that the CLI reports as supported, and it uses the direction file that comes with Bubo. Used after the server restarts.',
  },
  '--cache-disk': {
    help: 'Save long conversations to the SSD so a restart does not need to read them again. Off is recommended to reduce SSD writes.',
  },
  '--cache-disk-bytes': {
    help: 'Limit how much SSD space saved conversations use. The default is recommended.',
  },
  '--cache-disk-staging-bytes': {
    help: 'Limit RAM used while saving conversations to the SSD. Automatic is recommended.',
  },
  '--cache-ram-bytes': {
    help: 'Limit RAM used to remember earlier prompts. The default is recommended.',
  },
  // The CLI names these rows after the switch that turns a feature off. The GUI
  // shows the feature itself, with On / Off; the wire values stay the
  // switch's false / true.
  '--no-behaviour': {
    label: 'Behaviour defaults',
    help: 'The eight built-in behaviour defaults. Keep them on unless plain sampling is required outside the exact profile.',
  },
  '--no-cache': {
    label: 'Prompt caches',
    help: 'RAM and disk prompt caches for all requests. Turn them off when every request must read its full prompt, for reproducible output.',
  },
  '--no-cache-disk': {
    label: 'Allow disk cache',
    help: 'Use the disk snapshot cache. Turn it off when the disk cache folder must be ignored.',
  },
  '--no-draft-head': {
    label: 'Draft head',
    help: 'The draft output head from the model package. Turn it off to run MTP without it.',
  },
  '--no-mtp': {
    label: 'MTP drafts',
    help: 'MTP drafts and the speculative backend. Turn them off when decoding must run without MTP.',
  },
  '--no-prompt-lookup': {
    label: 'Allow prompt lookup',
    help: 'Prompt lookup proposals. Keep them on with MTP unless lookup comparisons need them off.',
  },
  '--no-twin': {
    label: 'Decode twin',
    help: 'The faster 4-bit decode twin model for writing. Turn it off for troubleshooting or to release its model storage.',
  },
  '--no-vision': {
    label: 'Image input',
    help: 'The vision file that the package names. Turn it off to use less memory or to run without that file.',
  },
  '--host': {
    help: 'Choose which network address accepts connections. Use the local address unless other devices need access.',
  },
  '--port': { help: 'Choose the port clients use to connect to Bubo.' },
}
const packagePaths = new Set([
  '--twin-model',
  '--mmproj',
  '--dflash-model',
  '--dspark-model',
  '--draft-head',
  '--draft-head-model',
  'GUFO_DRAFT_OUT_MODEL',
  'GUFO_DRAFT_HEAD',
  'GUFO_BATCH_DRAFT_HEAD',
  'GUFO_DENSE_TWIN_MODEL',
  'BUBO_MMPROJ',
])
// Enum values the CLI publishes as bare words; the GUI shows them as words.
// Words for one setting come first; the shared words follow. This is the one map
// for the control, the collapsed summary, the pending-restart notice and the
// recovery messages.
const onOff = { false: 'On', true: 'Off' }
const valueLabels: Record<string, Record<string, string>> = {
  '--kv-cache-type': { f16: 'f16', q8_0: 'Q8' },
  '--no-behaviour': onOff,
  '--no-cache': onOff,
  '--no-cache-disk': onOff,
  '--no-draft-head': onOff,
  '--no-mtp': onOff,
  '--no-prompt-lookup': onOff,
  '--no-twin': onOff,
  '--no-vision': onOff,
}
const enumWords: Record<string, string> = {
  on: 'On',
  off: 'Off',
  auto: 'Automatic',
  normal: 'Normal',
  fast: 'Fast',
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra high',
  max: 'Max',
  latin: 'Latin',
  full: 'Full',
  mtp: 'MTP',
  dspark: 'DSPARK',
  dflash: 'DFlash',
  dflash2: 'DFlash 2',
  survival: 'Survival',
  current: 'Current',
  fit: 'Fit',
}
// The words for one value of one setting.
export function valueWord(name: string, value: unknown) {
  const text = String(value)
  return valueLabels[name]?.[text] ?? enumWords[text.toLowerCase()] ?? text
}
// The words of every value an enum setting offers.
export function guiValueLabels(name: string, values?: readonly unknown[]) {
  return {
    ...Object.fromEntries(
      (values ?? []).map((value) => [String(value), valueWord(name, value)]),
    ),
    ...valueLabels[name],
  }
}
// What the default of a turn-off switch means in the positive wording.
const positiveDefault: Record<string, string> = {
  '--no-behaviour': 'On',
  '--no-cache': 'On; exact turns it off',
  '--no-cache-disk': 'On',
  '--no-draft-head': 'On',
  '--no-mtp': 'On',
  '--no-prompt-lookup': 'On',
  '--no-twin': 'On',
  '--no-vision': 'On',
}
// A turn-off switch (--no-mtp and others) shown as the feature with On / Off.
export function isTurnOffSwitch(name: string) {
  return name in positiveDefault
}
// A pending change as the page shows it: the GUI's own label and value words for
// a setting it presents (the key is the switch name), the CLI's label and raw
// value for any other.
export function pendingChangeWords(change: {
  key: string
  label?: string
  running?: string
  next: string
}) {
  return {
    label: presentation[change.key]?.label ?? change.label ?? 'Setting',
    running: change.running ? valueWord(change.key, change.running) : '',
    next: valueWord(change.key, change.next),
  }
}
export function guiRegistryPresentation(row: Row): Row {
  const wording = presentation[row.name]
  return {
    ...row,
    ...(wording
      ? {
          label: wording.label ?? row.label,
          ...(wording.help
            ? { purpose: wording.help, note: wording.help }
            : {}),
        }
      : {}),
    ...(positiveDefault[row.name]
      ? { default_text: positiveDefault[row.name]! }
      : {}),
    ...(row.name === '--profile' ? { group: 'Engine tuning' } : {}),
    ...(['--host', '--port'].includes(row.name) ? { group: 'Network' } : {}),
    ...(row.group === 'Memory' ? { group: 'Memory and caches' } : {}),
    ...(row.group === 'Hardware and memory' ? { group: 'Hardware' } : {}),
    ...(packagePaths.has(row.name)
      ? {
          audience: 'debug',
          purpose: 'Supplied by the model package',
          note: 'Supplied by the model package',
        }
      : {}),
  }
}

// A row the registry flags experimental shows it in its label, beside the name,
// so it is visible in any cell. Prefill mode's registry entry says "Experimental."
// in its text but does not carry the flag, so it is named here.
const experimentalRows = new Set(['--prefill-mode'])
// The label of a setting: the GUI's own where it genuinely needs other words
// (a turn-off switch shown as the feature, the KV cache, In-engine abliteration),
// else the registry's.
export function settingLabel(row: Row) {
  const label = presentation[row.name]?.label ?? row.label ?? row.name
  return row.experimental || experimentalRows.has(row.name)
    ? `${label} (experimental)`
    : label
}
// The short hint of a setting (at most three words), if it has one.
export function settingHint(name: string) {
  return presentation[name]?.short
}
// The short phrase for a setting the CLI says the model cannot use; the CLI's
// whole reason goes into the About text.
export function unavailableHint(reason: string) {
  if (/already abliterated/i.test(reason)) return 'Already abliterated'
  if (/not tested/i.test(reason)) return 'Untested model'
  if (/not supported/i.test(reason)) return 'Not supported yet'
  return 'Unsupported model'
}

// The unit of a setting's number, from the name the registry gives it.
export function unitOf(name: string) {
  if (/-ms$|_MS$/.test(name)) return 'ms'
  if (/_MB$/.test(name)) return 'MiB'
  if (name === '--context') return 'tokens'
  return undefined
}

const placeholderNotes = new Set([
  'Inventory classification.',
  'unclassified',
  'Use this switch with its model family.',
])
// The explanation of a row, as paragraphs: the GUI's own text, else the CLI's
// purpose and note. Generated placeholders are no explanation and are dropped.
export function aboutParagraphs(row: Row) {
  const wording = presentation[row.name]?.help
  if (wording) return [wording]
  return [row.purpose, row.note]
    .map((text) => (text ?? '').trim())
    .filter(
      (text, index, all) =>
        text &&
        !placeholderNotes.has(text) &&
        !/^Control [a-z0-9 ]+\.$/i.test(text) &&
        all.indexOf(text) === index,
    )
}
// The default as the user reads it. A default the CLI does not report is said
// plainly, never as "as shown".
export function defaultWords(row: Row) {
  const text = (row.default_text ?? '').trim()
  if (!text || text === 'as shown') return 'Default not reported'
  // A byte budget written in digits is read in GiB or MiB.
  if (row.name.endsWith('-bytes') && /^\d+$/.test(text) && Number(text) > 0)
    return bytes(Number(text), { digits: 0 })
  return text
}

// ---------------------------------------------------------------------------
// The one presentation map. Every place that names a setting, a value, a unit,
// a default, a request result or a model state takes its words from here: rows,
// summaries, pending-restart notices, recovery messages, tooltips and charts.
// Sections below are owned by the page that needs them; add words here, never in
// a component.
// ---------------------------------------------------------------------------

// --- Requests: a request's result and its cache state ---

type Request = View['requests'][number]

// The word for how a request's prompt met the cache (the CLI reports disk,
// memory or miss). One word for the table cell and the chart tooltip.
export function cacheWord(kind: string | null | undefined) {
  if (!kind) return '—'
  if (kind === 'memory') return 'RAM hit'
  if (kind === 'disk') return 'Disk hit'
  if (kind === 'miss') return 'Miss'
  return 'Other'
}

// Why a completed request ended (the CLI reports stop, stop_sequence, length or
// cancelled). Each reported reason keeps its own meaning.
const finishWords: Record<string, string> = {
  stop: 'Completed',
  stop_sequence: 'Stop sequence',
  length: 'Token limit',
  cancelled: 'Cancelled',
}

// The Requests table's column titles: the full term behind each short header.
export const requestColumnTitle: Record<string, string> = {
  time: 'Local time',
  prompt_tokens: 'Prompt tokens',
  completion_tokens: 'Output tokens',
  ttft_ms: 'Time to first token, in milliseconds',
  decode_tps: 'Decode rate of this request, in tokens per second',
  prefill_tps: 'Prefill rate of this request, in tokens per second',
}

// A request that did not complete shows one plain sentence (ERRORS-v8, request
// rows), never the finish reason of the text it had started (a failed row can
// still carry "stop"). The raw status and code are in the cell's title.
export function requestResult(row: Request) {
  if (row.outcome === 'completed')
    return finishWords[row.finish_reason ?? ''] ?? 'Completed'
  const code = row.error_code ?? undefined
  const status = row.status ?? undefined
  if (row.outcome === 'disconnected') return 'Stopped: the client disconnected.'
  if (row.outcome === 'stream_error')
    return code === 'device_lost'
      ? 'Stopped: the GPU stopped responding.'
      : 'Stopped: the reply ended early.'
  if (row.outcome === 'refused') {
    const refusal = row.refusal ?? undefined
    const why = refusal?.code ?? code
    const needed = refusal?.needed_bytes
    const free = refusal?.free_bytes
    if (why === 'memory_pressure') return 'Stopped: Windows ran low on memory.'
    if (why === 'insufficient_memory_permanent')
      return needed == null
        ? 'Refused: needs more memory than this computer can ever provide.'
        : `Refused: needs ${bytes(needed, { unit: 'GiB' })} of memory, more than this computer can ever provide.`
    if (why === 'insufficient_memory')
      return needed == null || free == null
        ? 'Refused: not enough memory is free. Try again when more memory is free.'
        : `Refused: needs ${bytes(needed, { unit: 'GiB' })} of memory, ${bytes(free, { unit: 'GiB' })} free. Try again when more memory is free.`
    return 'Refused.'
  }
  if (code === 'context_length_exceeded')
    return 'Failed: the prompt is longer than the context.'
  if (status === 404)
    return "Failed: the client asked for a model name this server doesn't serve."
  if (code === 'queue_full' || status === 429)
    return 'Failed: the server was busy.'
  if (status === 408 || status === 504 || (code && /timeout/.test(code)))
    return 'Failed: the request took too long.'
  if (code === 'device_lost') return 'Failed: the GPU stopped responding.'
  if (status === 401) return 'Failed: the API key was missing or wrong.'
  if (status != null && status >= 400 && status < 500)
    return 'Failed: the request was not valid.'
  if (status != null && status >= 500)
    return 'Failed: the server could not answer.'
  return 'Failed.'
}

// The raw finish reason, status and error code, for the result cell's title.
export function requestResultRaw(row: Request) {
  return [
    row.outcome === 'completed' ? row.finish_reason : '',
    row.status != null ? `HTTP ${row.status}` : '',
    row.error_code ?? '',
  ]
    .filter(Boolean)
    .join(' · ')
}

// --- Models: the state of a model row ---
// One word per state of a model package, said the same in Installed and Discover.
export const modelState = {
  inUse: 'In use',
  selected: 'Selected for next start',
  incomplete: 'Incomplete',
  cannotRun: 'Cannot run',
  downloading: 'Downloading',
} as const

// The list of folders Bubo scans besides the download folder.
export const additionalModelFolders = {
  title: 'Additional model folders',
  hint: 'Scanned for models',
} as const

// What a model selection does while the server runs.
export const selectionAppliesAfterRestart = 'Selection applies after restart.'
export const stopServerToCheckMemory = 'Stop the server to check memory.'

// --- Onboarding: the download folder picker beside a space warning ---
export const downloadFolderScope = 'This choice applies to the model files.'

// --- Settings: the app's own preferences ---

export const appSettingWords = {
  startAtLogin: 'Start this app at login',
  closeToTray: 'Keep this app in the tray when closing',
  startServer: 'Start the server with this app',
  checkForUpdates: 'Check for updates at start',
  font: 'Font',
  engineFolder: 'Engine folder',
  defaultEngineFolder: 'Default engine folder',
} as const

// Actions that save at once, without the Save bar.
export const savedImmediately = 'Saved immediately'
