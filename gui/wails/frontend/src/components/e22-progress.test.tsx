import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { bytes } from '@/lib/format'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())



















function preparing() {
  const view = coreView('preparing')
  const overall = view.progress!.overall!
  return { view, overall }
}

// The models job of the same download, carrying the same whole-job number.
function modelsJob() {
  const { view, overall } = preparing()
  const job = coreView('models-download-in-progress').models!.download!
  view.models!.download = { ...job, overall }
  return { view, overall }
}

const whole = (overall: { bytes_done: number; bytes_total: number }) =>
  `${bytes(overall.bytes_done)} of ${bytes(overall.bytes_total)} · ${Math.round((overall.bytes_done / overall.bytes_total) * 100)} %`

const pages = ['/onboarding', '/', '/models'] as const

for (const path of pages) {
  test(`5a: ${path} prints the whole-download line`, async () => {
    const { view, overall } = modelsJob()
    mountCorePage(path, view)
    const line = page.getByText(whole(overall)).first()
    await expect.element(line).toBeVisible()
    // The line starts with the numbers: no "Current file" scope in front.
    expect(
      line.element().closest('p')!.textContent!.startsWith(whole(overall)),
    ).toBe(true)
  })

  test(`5a: ${path} says "Current file" when no whole total is known`, async () => {
    const { view } = modelsJob()
    delete view.progress!.overall
    delete view.models!.download!.overall
    const progress = view.progress!
    mountCorePage(path, view)
    await expect
      .element(
        page
          .getByText(
            `Current file: ${bytes(progress.bytes_done)} of ${bytes(progress.bytes_total!)}`,
          )
          .first(),
      )
      .toBeVisible()
  })
}

test('5b: Get started shows the rate and the time left once a second frame arrives', async () => {
  const { view, overall } = preparing()
  const { replaceView } = mountCorePage('/onboarding', structuredClone(view))
  await expect.element(page.getByText(whole(overall))).toBeVisible()
  expect(document.body.textContent).not.toContain('/s')
  await new Promise((resolve) => setTimeout(resolve, 300))
  const next = structuredClone(view)
  next.progress!.overall!.bytes_done = overall.bytes_done + 300 * 1024 ** 2
  next.progress!.bytes_done += 300 * 1024 ** 2
  replaceView(next)
  const line = page.getByText(/MiB\/s/)
  await expect.element(line).toHaveTextContent(/about \d+ (s|min) left$/)
})
