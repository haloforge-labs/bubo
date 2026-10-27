export const NO_LINES = 'No log lines yet.'
export const NO_MATCH = 'No lines match the filter.'

export function shownLines(
  lines: readonly string[] | undefined,
  filter: string,
) {
  const query = filter.toLocaleLowerCase()
  return (lines ?? []).filter((line) =>
    line.toLocaleLowerCase().includes(query),
  )
}

export function logText(lines: readonly string[] | undefined, filter: string) {
  if (!lines?.length) return NO_LINES
  return shownLines(lines, filter).join('\n') || NO_MATCH
}

export function copyLabel(filter: string) {
  return filter ? 'Copy shown lines' : 'Copy all'
}
