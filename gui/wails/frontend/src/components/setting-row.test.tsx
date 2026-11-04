import { afterEach, expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { SettingRow } from '@/components/setting-row'

afterEach(() => {
  document.body.innerHTML = ''
})

// One policy decides where the hint sits: from 22rem of the row's own width it is
// beside the control; narrower, it is behind the ⓘ. The app's root font is 18 px,
// so 22rem is 396 px (a 16 px assumption made rows of 352-395 px stack the hint).
function row(width: number) {
  return render(
    <div style={{ width }}>
      <SettingRow
        name="demo"
        label="Demo"
        htmlFor="demo"
        control={<input id="demo" />}
        hint="Short hint"
        about={<p>About text</p>}
      />
    </div>,
  )
}

test('a row narrower than 22rem keeps its hint behind the info button, wider keeps it beside the control', async () => {
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize)
  const narrow = await row(Math.floor(22 * rem) - 20)
  const hints = () =>
    [...document.querySelectorAll('[data-setting="demo"] *')].filter(
      (node) => node.textContent === 'Short hint',
    )
  await expect.element(page.getByRole('group', { name: 'Demo' })).toBeVisible()
  // Behind the ⓘ: the one copy is the popup's, never a visible line of the row.
  expect(
    hints().every((node) => node.closest('details') !== null),
    'hint is in the popup',
  ).toBe(true)
  await narrow.unmount()
  await row(Math.ceil(22 * rem) + 20)
  await expect.element(page.getByRole('group', { name: 'Demo' })).toBeVisible()
  expect(
    hints().some((node) => node.closest('details') === null),
    'hint is beside the control',
  ).toBe(true)
})
