// The interface typeface: "serif" (the default) or "sans". One attribute on the
// root element swaps the typeface tokens in app.css, so every surface follows.
export type InterfaceFont = 'serif' | 'sans'

const remembered = 'bubo-font'

export function interfaceFont(saved?: string): InterfaceFont {
  return saved === 'sans' ? 'sans' : 'serif'
}

// Applies the font at once and remembers it, so the next start draws in this
// font before the saved settings arrive. The saved settings stay the truth.
export function applyFont(font: InterfaceFont) {
  document.documentElement.dataset.font = font
  try {
    localStorage.setItem(remembered, font)
  } catch {
    // Storage can be blocked; the saved settings still decide.
  }
}

export function restoreFont() {
  try {
    document.documentElement.dataset.font = interfaceFont(
      localStorage.getItem(remembered) ?? undefined,
    )
  } catch {
    document.documentElement.dataset.font = 'serif'
  }
}
