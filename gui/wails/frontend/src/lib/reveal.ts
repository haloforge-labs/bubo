import { useEffect } from 'react'
import { useLocation, useNavigate } from '@tanstack/react-router'

// A recovery link that names a setting says so in its hash: `setting:<switch>`.
// The page the link opens shows that setting: it opens the tab and the folds
// that hold it, then scrolls to its row.
const prefix = 'setting:'

export function settingHash(name: string) {
  return `${prefix}${name}`
}

export function useRevealedSetting() {
  const hash = useLocation({ select: (location) => location.hash })
  return hash.startsWith(prefix) ? hash.slice(prefix.length) : undefined
}

// The page's own choices (a tab, a filter) end a reveal: the next recovery link
// then changes the hash again, even for the setting it named before.
export function useEndReveal() {
  const navigate = useNavigate()
  const revealed = useRevealedSetting()
  return () => {
    if (revealed) void navigate({ to: '.', hash: undefined, replace: true })
  }
}

// Open every fold above the row and scroll to it, once the row exists.
export function useRevealSetting(name: string | undefined, ready: unknown) {
  useEffect(() => {
    if (!name || !ready) return
    const row = document.querySelector<HTMLElement>(
      `[data-setting="${CSS.escape(name)}"]`,
    )
    if (!row) return
    for (
      let fold = row.closest('details');
      fold;
      fold = fold.parentElement?.closest('details') ?? null
    )
      fold.open = true
    row.scrollIntoView({ block: 'center' })
  }, [name, ready])
}
