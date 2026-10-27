import { afterEach, expect, test, vi } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { bridgeReply } from '@/components/bridge-test-reply'
import { actions } from '@/lib/bindings'

afterEach(() => vi.restoreAllMocks())

test('Cyber-Tiel card shows its shared licence line once', async () => {
  const view = coreView()
  const entry = view.models!.catalogue.find((row) =>
    row.id.startsWith('cyber-tiel'),
  )!
  expect(entry.licence.note).toBe(entry.licence.prompt_line)
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: `About ${entry.display_name}`, exact: true })
    .click()
  const card = page.getByRole('dialog')
  expect(
    card.element().textContent!.split(entry.licence.prompt_line),
  ).toHaveLength(2)
})

for (const kind of ['file', 'folder'] as const) {
  test(`header Import opens the existing model ${kind} action`, async () => {
    const browse = vi
      .spyOn(actions, kind === 'file' ? 'BrowseModelFile' : 'BrowseModelFolder')
      .mockResolvedValue(bridgeReply({ message: 'Imported.' }))
    mountCorePage('/models', coreView())
    await page.getByRole('button', { name: 'Import', exact: true }).click()
    await page
      .getByRole('menuitem', {
        name: kind === 'file' ? 'Model file…' : 'Model folder…',
      })
      .click()
    expect(browse).toHaveBeenCalledExactlyOnceWith()
    expect(document.getElementById('import')).not.toBeNull()
    await expect
      .element(page.getByRole('status').filter({ hasText: 'Imported.' }))
      .toBeVisible()
  })
}

test('catalogue info is keyboard reachable, shows CLI facts and omits absent optional fields', async () => {
  const view = coreView()
  const entry = view.models!.catalogue[0]!
  mountCorePage('/models', view)
  const info = page.getByRole('button', {
    name: `About ${entry.display_name}`,
    exact: true,
  })
  await expect.element(info).toBeVisible()
  ;(info.element() as HTMLButtonElement).focus()
  await userEvent.keyboard('{Enter}')
  const card = page.getByRole('dialog', { name: entry.display_name })
  await expect.element(card).toHaveTextContent(entry.licence.prompt_line)
  await expect.element(card).toHaveTextContent('Download')
  await expect.element(card).toHaveTextContent(entry.technical_name)
  await expect.element(card).toHaveTextContent(entry.about)
  await expect.element(card).not.toHaveTextContent('Role')
  await expect.element(card).not.toHaveTextContent('Default status')
  expect(card.element().textContent).not.toMatch(
    /On GPU|In system memory|not reported|\[X\]|\[Y\]/,
  )
  await userEvent.keyboard('{Escape}')
  await expect.element(card).not.toBeInTheDocument()
  expect(document.activeElement).toBe(info.element())
})

test('catalogue rows from older CLI versions omit unavailable card text', async () => {
  const view = coreView()
  const entry = view.models!.catalogue.find((row) =>
    row.id.startsWith('diffusiongemma'),
  )!
  Object.assign(entry, {
    about: '', // Older CLI rows did not supply card text.
  })
  mountCorePage('/models', view)
  await expect
    .element(
      page.getByRole('button', {
        name: `About ${entry.display_name}`,
        exact: true,
      }),
    )
    .not.toBeInTheDocument()
})

test('catalogue cards show the real CLI identity, prose and publication reason', async () => {
  const view = coreView()
  const entry = view.models!.catalogue.find((row) =>
    row.id.includes('lowmemory'),
  )!
  mountCorePage('/models', view)
  await page
    .getByRole('button', { name: `About ${entry.display_name}`, exact: true })
    .click()
  const card = page.getByRole('dialog')
  await expect.element(card).toHaveTextContent(entry.technical_name)
  await expect.element(card).toHaveTextContent(entry.about)
  await expect.element(card).not.toHaveTextContent('On GPU')
  await expect.element(card).not.toHaveTextContent('In system memory')
  await expect.element(card).not.toHaveTextContent('not published')
  await expect
    .element(page.getByRole('link', { name: 'arXiv 2604.18556' }))
    .toBeVisible()
})
