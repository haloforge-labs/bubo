import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from './test-view'
import { CopyText } from './copy-text'
import { useState } from 'react'
import { actions } from '@/lib/bindings'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('compact copy is an outlined icon button and changes to Copied after the exact copy', async () => {
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(
    <CopyText
      compact
      text="https://example.test/fixture"
      label="Copy fixture address"
    />,
    fixture(),
  ).cleanup
  const button = page.getByRole('button', { name: 'Copy fixture address' })
  await expect.element(button).toBeVisible()
  expect(button.element().querySelector('svg')).not.toBeNull()
  expect(getComputedStyle(button.element()).borderTopStyle).toBe('solid')
  await button.click()
  expect(copy).toHaveBeenCalledExactlyOnceWith('https://example.test/fixture')
  await expect.element(button).toHaveTextContent('Copied')
  await expect.element(page.getByRole('status')).toHaveTextContent('Copied.')
})

test('clipboard failure stays a failure and never shows Copied', async () => {
  vi.spyOn(actions, 'CopyText').mockRejectedValue({
    code: 'fixture_clipboard',
    remedy: 'Fixture clipboard unavailable.',
  })
  cleanup = renderView(<CopyText text="fixture" />, fixture()).cleanup
  await page.getByRole('button', { name: 'Copy', exact: true }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture clipboard unavailable.')
  await expect.element(page.getByRole('status')).not.toBeInTheDocument()
})

test('Copied resets when the address changes and the next click copies the new address', async () => {
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  function ChangingAddress() {
    const [text, setText] = useState('first fixture address')
    return (
      <>
        <CopyText text={text} />
        <button onClick={() => setText('second fixture address')}>
          Change fixture address
        </button>
      </>
    )
  }
  cleanup = renderView(<ChangingAddress />, fixture()).cleanup
  const button = page.getByRole('button', { name: 'Copy', exact: true })
  await button.click()
  await expect.element(button).toHaveTextContent('Copied')
  await page.getByRole('button', { name: 'Change fixture address' }).click()
  await expect.element(button).toHaveTextContent('Copy')
  await expect.element(page.getByRole('status')).not.toBeInTheDocument()
  await button.click()
  expect(copy.mock.calls).toEqual([
    ['first fixture address'],
    ['second fixture address'],
  ])
})
