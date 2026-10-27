import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { renderView, fixture } from '@/components/test-view'
import LogsPage from './index'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

let cleanup: (() => void) | undefined
afterEach(() => cleanup?.())

test('the shell collects GPU observations on other pages under only gpu-history', async () => {
  const view = coreView()
  view.gpu = { use_pct: 0 }
  const mounted = mountCorePage('/models', view)
  const key = ['gpu-history', view.instance?.instance_id ?? view.generation]
  await expect
    .poll(
      () =>
        mounted.client
          .getQueryData<{ gpu?: { use_pct?: number } }[]>(key)
          ?.at(-1)?.gpu?.use_pct,
    )
    .toBe(0)
  const count = mounted.client.getQueryData<unknown[]>(key)!.length
  const next = structuredClone(view)
  next.gpu = { use_pct: 80 }
  mounted.replaceView(next)
  await expect
    .poll(() => mounted.client.getQueryData<unknown[]>(key)?.length)
    .toBe(count + 1)
  expect(
    mounted.client.getQueryData<{ gpu?: { use_pct?: number } }[]>(key)?.at(-1)
      ?.gpu?.use_pct,
  ).toBe(80)
  expect(
    mounted.client.getQueryCache().findAll({ queryKey: ['server-history'] }),
  ).toHaveLength(0)
})

test('GPU history uses the existing instance history above log lines, preserving missing metrics', async () => {
  const view = fixture()
  const mounted = renderView(<LogsPage />, view)
  cleanup = mounted.cleanup
  const at = Date.now()
  mounted.client.setQueryData(
    ['gpu-history', view.instance?.instance_id ?? view.generation],
    Array.from({ length: 5 }, (_, index) => ({
      at: at - (4 - index) * 2000,
      gpu: { use_pct: index * 20, memory_used_bytes: (10 + index) * 1024 ** 3 },
    })),
  )
  const section = page.getByRole('region', { name: 'GPU history' })
  await expect.element(section).toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'GPU use history' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('figure', { name: 'GPU memory history' }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('GPU temperature history')
  const lines = document.querySelector('[aria-label="Log lines"]')
  expect(lines).not.toBeNull()
  expect(
    section.element().compareDocumentPosition(lines!) &
      Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy()
})

test('without GPU observations Logs shows no fabricated history', async () => {
  cleanup = renderView(<LogsPage />, fixture()).cleanup
  await expect
    .element(page.getByRole('heading', { name: 'Logs', exact: true }))
    .toBeVisible()
  expect(document.querySelector('.recharts-wrapper')).toBeNull()
})
