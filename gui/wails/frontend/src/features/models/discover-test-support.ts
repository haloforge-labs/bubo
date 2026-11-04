import { expect } from 'vitest'
import { page } from 'vitest/browser'

// Discover is folded while there are installed models and open when there are
// none: a test that reads the catalogue opens it first, whichever way it starts.
export async function openDiscover() {
  const region = page.getByRole('region', { name: 'Discover models' })
  await expect.element(region).toBeInTheDocument()
  const details = region.element().querySelector('details')!
  if (!details.open)
    await region.getByRole('button', { name: /^Discover/ }).click()
  await expect.element(region.getByRole('table')).toBeVisible()
  return region
}
