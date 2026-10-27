import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

const sentence = 'The GPU stopped responding. Bubo restarted the server.'

test('an automatic restart after a lost GPU is said plainly, in no red box', async () => {
  mountCorePage('/', coreView('gpu-restarted'))
  await expect.element(page.getByText(sentence, { exact: true })).toBeVisible()
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Stop server', exact: true }))
    .toBeVisible()
})

test('a server nobody restarted says nothing about the GPU', async () => {
  mountCorePage('/', coreView('ready-idle'))
  await expect
    .element(page.getByRole('heading', { name: 'Server', exact: true }))
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Bubo restarted the server')
})

test('after a second lost GPU the error stays and the user starts it again', async () => {
  mountCorePage('/', coreView('gpu-lost-stopped'))
  await expect
    .element(page.getByText('The GPU stopped responding again.'))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  expect(document.body.textContent).not.toContain('Bubo restarted the server')
})
