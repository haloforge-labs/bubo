import type { View } from '@/lib/bindings'

// The draft file and the image file have their own rows on the Server page.
export const sideFileSwitches = ['--mtp-model', '--mmproj'] as const

export function serverRowSection(row: View['switches'][number]) {
  if (!row.gui_control || row.audience !== 'user' || row.name === '--api-key')
    return undefined
  if (['--model', '--manifest', '--context', '--sessions'].includes(row.name))
    return 'Setup'
  if (['--host', '--port'].includes(row.name)) return undefined
  // The name clients see sits beside the model picker.
  if (row.name === '--served-model-name') return 'Name'
  if (row.name === '--kv-cache-type') return 'Cache'
  // The refusal switch sits beside the context cache, in the setup the user sees
  // before Start; prompt reading speed is an inference setting.
  if (row.name === '--refusal-direction') return 'Cache'
  if (row.name === '--prefill-mode') return 'Inference'
  switch (row.group) {
    case 'Server':
      return 'Network'
    case 'Behaviour':
    case 'Inference':
      return 'Inference'
    default:
      return undefined
  }
}
