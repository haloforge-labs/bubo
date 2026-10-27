// Accepted client alias for the model ID reported by the CLI.
export const modelAlias = 'bubo'

// The CLI's "none" placeholder selects the built-in list, never a filename.
function isMarkerList(name: string) {
  return name === 'GUFO_THINK_MARKER_LIST'
}

export function markerListUnset(name: string, value: unknown) {
  return (
    isMarkerList(name) &&
    ['', 'none'].includes(String(value).trim().toLowerCase())
  )
}

export function unsetSettingText(name: string) {
  return isMarkerList(name)
    ? 'Custom marker-list file: not set (built-in list)'
    : 'Not set'
}
