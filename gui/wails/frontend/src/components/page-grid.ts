// One grid policy for every page. Columns follow the width of the page itself
// (the grid's own container), never the window: the sidebar and the padding make
// the window a poor guide. A column is never narrower than the shared minimum
// (16rem: label, control and the ⓘ of a row), so a page lays out as many columns
// as fit, and a page that is narrower than two of them has one.
export const pageGrid =
  'grid grid-cols-[repeat(auto-fit,minmax(min(100%,16rem),1fr))] gap-x-4 gap-y-2'
// The same rule for figures (a rate and its caption): they fit in a narrower column.
export const figureGrid =
  'grid grid-cols-[repeat(auto-fit,minmax(min(100%,9rem),1fr))] gap-x-6 gap-y-2'
