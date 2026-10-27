import type { View } from '@/lib/bindings'

export function serverRowSection(row: View['switches'][number]) {
  if (!row.gui_control || row.audience !== 'user' || row.name === '--api-key')
    return undefined
  if (['--model', '--manifest', '--context', '--sessions'].includes(row.name))
    return 'Setup'
  if (['--host', '--port'].includes(row.name)) return undefined
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
