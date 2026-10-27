import type { View } from '@/lib/bindings'

type Row = View['switches'][number]
// Presentation overrides stay in the GUI: release CLI binaries and wire values are unchanged.
const presentation: Record<string, { label?: string; help: string }> = {
  '--seed': {
    label: 'Seed',
    help: 'A fixed seed with the same prompt and settings gives the same answer. A request can still send its own seed.',
  },
  '--profile': {
    label: 'Mode',
    help: "How Bubo tunes the engine. Default is recommended; Vanilla turns Bubo's tuning off.",
  },
  '--top-p': {
    label: 'Top-p',
    help: 'Choose how much of the likely next words to consider. Lower values make answers more focused.',
  },
  '--top-k': {
    label: 'Top-k',
    help: 'Limit how many likely next words Bubo considers. Use the default for normal use.',
  },
  '--npu-prefill': {
    help: 'Use the NPU to help read long prompts. Off is recommended in 0.1.1.',
  },
  '--no-twin': {
    label: 'Disable 4-bit helper',
    help: 'Turn off the faster 4-bit helper model for writing (for troubleshooting).',
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
  '--no-cache-disk': { help: 'Turn off saving conversations to the SSD.' },
  '--no-vision': { help: 'Turn off image input to use less memory.' },
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
export function packageManagedSetting(name: string) {
  return packagePaths.has(name)
}
export function guiRegistryPresentation(row: Row): Row {
  const wording = presentation[row.name]
  return {
    ...row,
    ...(wording
      ? {
          label: wording.label ?? row.label,
          purpose: wording.help,
          note: wording.help,
        }
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
