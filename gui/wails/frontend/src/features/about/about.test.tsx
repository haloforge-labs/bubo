import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import { UpdateBanner } from '@/components/update-banner'
import AboutPage from './index'
import notices from '@/lib/third-party'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

test('a real failed update check shows the Update failed banner', async () => {
  const failed = fixture('update-failed')
  cleanup = renderView(<UpdateBanner update={failed.update} />, failed).cleanup
  await expect
    .element(page.getByRole('link', { name: /Update failed/ }))
    .toBeVisible()
})

test('a failed updater start or check is no Update failed pill; About says why in one sentence', async () => {
  const view = fixture('update-check-failed')
  cleanup = renderView(
    <>
      <div aria-label="Header update">
        <UpdateBanner update={view.update} />
      </div>
      <AboutPage />
    </>,
    view,
  ).cleanup
  await expect
    .element(
      page.getByText(
        "Bubo can't check for updates: the update address could not be verified.",
        { exact: true },
      ),
    )
    .toBeVisible()
  await expect
    .element(page.getByRole('link', { name: /Update failed/ }))
    .not.toBeInTheDocument()
  // The cause appears once in the labelled error panel; raw text stays in Details.
  await expect
    .element(page.getByRole('alert').getByText('Error', { exact: true }))
    .toBeVisible()
  // Every can't-check state has the same retry.
  await expect
    .element(page.getByRole('button', { name: 'Try again' }))
    .toBeVisible()
})

test('rollback pending names the previous version, and the privacy line shows only beside Check', async () => {
  const view = fixture('update-rollback_pending')
  cleanup = renderView(
    <>
      <div aria-label="Header update">
        <UpdateBanner update={view.update} />
      </div>
      <AboutPage />
    </>,
    view,
  ).cleanup
  await expect
    .element(page.getByRole('link', { name: /Update didn't finish/ }))
    .toBeVisible()
  await expect
    .element(
      page.getByText(
        `Recover the previous version: Bubo closes and reopens as version ${view.update.previous_version}.`,
        { exact: true },
      ),
    )
    .toBeVisible()
  expect(document.body.textContent).not.toContain('Rollback pending')
  expect(document.body.textContent).not.toContain('GitHub release feed')
})

test('the privacy line is beside the Check button', async () => {
  cleanup = renderView(<AboutPage />, fixture('update-current')).cleanup
  await expect.element(page.getByText(/GitHub release feed/)).toBeVisible()
})

test('About renders the canonical inventory as component, version and licence rows', async () => {
  cleanup = renderView(<AboutPage />, fixture()).cleanup
  await page.getByText('Third-party notices', { exact: true }).click()
  await expect
    .element(page.getByLabelText('Third-party inventory'))
    .toBeVisible()
  const table = page
    .getByRole('table', { name: 'Third-party inventory' })
    .element()
  const entries = notices
    .split('\n')
    .filter((line) => line.startsWith('| '))
    .slice(2)
  expect(table.querySelectorAll('tbody tr').length).toBe(entries.length)
  for (const [index, line] of entries.entries()) {
    expect(
      Array.from(
        table.querySelectorAll('tbody tr')[index]!.querySelectorAll('td'),
        (cell) => cell.textContent,
      ),
    ).toEqual(
      line
        .split('|')
        .slice(1, 4)
        .map((cell) => cell.trim()),
    )
  }
  expect(table.textContent).not.toMatch(
    /working tree|SHA-256|release approval|\|/,
  )
  expect(document.querySelector('pre')).toBeNull()
  const container = table.closest('[data-slot="table-container"]')!
  expect(table.getBoundingClientRect().right).toBeLessThanOrEqual(
    container.getBoundingClientRect().right + 1,
  )
  expect(
    table.querySelectorAll('th')[2]!.getBoundingClientRect().right,
  ).toBeLessThanOrEqual(container.getBoundingClientRect().right + 1)
})

for (const state of [
  'none',
  'current',
  'available',
  'progress',
  'ready',
  'failed',
  'rollback_pending',
]) {
  test(`generated update ${state} is rendered with its actual fields and header banner`, async () => {
    const view = fixture(`update-${state}`)
    cleanup = renderView(
      <>
        <div aria-label="Header update">
          <UpdateBanner update={view.update} />
        </div>
        <AboutPage />
      </>,
      view,
    ).cleanup
    await expect
      .element(page.getByText('Updates', { exact: true }))
      .toBeVisible()
    // Two versions are shown only when they are compared.
    if (view.update.available_version && state !== 'rollback_pending')
      await expect
        .element(
          page.getByText(`Installed: ${view.update.current_version}`, {
            exact: false,
          }),
        )
        .toBeVisible()
    else expect(document.body.textContent).not.toContain('Installed:')
    expect(
      document.querySelector('[data-update]')?.getAttribute('data-update'),
    ).toBe(state)
    if (state === 'none')
      await expect
        .element(page.getByRole('button', { name: 'Check for updates' }))
        .toBeVisible()
    if (state === 'available')
      await expect
        .element(page.getByRole('button', { name: 'Download update' }))
        .toBeVisible()
    if (state === 'progress')
      await expect.element(page.getByRole('progressbar')).toBeVisible()
    if (state === 'ready')
      await expect
        .element(page.getByRole('button', { name: 'Install and restart' }))
        .toBeVisible()
    if (view.update.error)
      await expect
        .element(
          page
            .getByRole('region', { name: 'About', exact: true })
            .getByRole('alert'),
        )
        .toHaveTextContent(view.update.error.remedy)
    if (view.update.notes && state !== 'rollback_pending') {
      await page
        .getByText(
          view.update.available_version
            ? `Update to ${view.update.available_version}`
            : 'Update details',
          {
            exact: true,
          },
        )
        .click()
      await expect
        .element(page.getByText(view.update.notes, { exact: true }))
        .toBeVisible()
    }
  })
}

test.each([
  // An unattempted check has no sentence above its own button.
  ['update-none', ''],
  ['update-current', 'Bubo is up to date.'],
] as const)(
  '%s distinguishes an unattempted check from a completed current check',
  async (name, wording) => {
    cleanup = renderView(<AboutPage />, fixture(name)).cleanup
    if (wording)
      await expect
        .element(page.getByText(wording, { exact: true }))
        .toBeVisible()
    else
      expect(document.body.textContent).not.toContain(
        'Check for updates to see whether',
      )
    await expect
      .element(
        page.getByRole('button', { name: 'Check for updates', exact: true }),
      )
      .toBeEnabled()
  },
)

test('all five update bindings retain unsupported results without success or invented transitions', async () => {
  for (const [state, label, action] of [
    ['none', 'Check for updates', 'CheckUpdate'],
    ['available', 'Download update', 'DownloadUpdate'],
    ['progress', 'Cancel download', 'CancelUpdate'],
    ['ready', 'Install and restart', 'InstallUpdate'],
    ['available', 'Later', 'DismissUpdate'],
  ] as const) {
    const view = fixture(`update-${state}`)
    const mock = vi.spyOn(actions, action).mockResolvedValue({
      error: {
        code: 'unsupported',
        remedy: 'Updater is not attached.',
        component: '',
        stage: '',
        retryable: false,
      },
    } as Awaited<ReturnType<(typeof actions)[typeof action]>>)
    const mounted = renderView(<AboutPage />, view)
    await page
      .getByRole('button', {
        name: label === 'Download update' ? /^Download update/ : label,
        exact: true,
      })
      .click()
    if (action === 'CheckUpdate') {
      // A failed check has one check-specific panel. Its raw cause stays in Details.
      expect(page.getByRole('alert').elements()).toHaveLength(1)
      await page.getByText('Details', { exact: true }).click()
      await expect
        .element(page.getByText('Updater is not attached.', { exact: true }))
        .toBeVisible()
    } else {
      await expect
        .element(page.getByRole('alert'))
        .toHaveTextContent('unsupported: Updater is not attached.')
    }
    expect(mock).toHaveBeenCalledTimes(1)
    expect(
      document.querySelector('[data-update]')?.getAttribute('data-update'),
    ).toBe(state)
    await (await mounted.screen).unmount()
    mounted.cleanup()
    mock.mockRestore()
  }
})

test('last result wording comes directly from View and About exposes links and third party notices', async () => {
  const view = fixture('update-none')
  view.update.last_result_notice = 'Updated to fixture-version.'
  cleanup = renderView(<AboutPage />, view).cleanup
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent(view.update.last_result_notice)
  await expect
    .element(page.getByRole('link', { name: 'Website', exact: true }))
    .toHaveAttribute('href', 'https://haloforgelabs.com')
  await expect
    .element(page.getByRole('link', { name: 'Documentation', exact: true }))
    .toHaveAttribute('href', 'https://haloforgelabs.com/docs/')
  await expect
    .element(
      page.getByRole('link', { name: 'Source and support', exact: true }),
    )
    .toHaveAttribute('href', 'https://github.com/haloforge-labs/bubo')
  await page.getByText('Third-party notices', { exact: true }).click()
  await expect
    .element(page.getByRole('table', { name: 'Third-party inventory' }))
    .toBeVisible()
})

// E21w W1: an update whose helper stopped before it changed anything. The
// updater's result has the code update_interrupted. The user reads one plain
// sentence, the installed version is unchanged, and Try again is the way on.
test('an update that stopped before it changed anything says so once and offers Try again', async () => {
  const sentence =
    'The update stopped before it changed anything. Your installed version is unchanged.'
  const view = fixture('update-failed')
  view.update.error = {
    code: 'update_interrupted',
    component: 'update',
    stage: '',
    remedy: sentence,
    retryable: true,
  }
  view.update.last_result_notice = sentence
  view.update.check_failed = false
  cleanup = renderView(<AboutPage />, view).cleanup
  await expect.element(page.getByRole('alert')).toHaveTextContent(sentence)
  // No developer words and no dead button: the card's own Try again is the one way on.
  expect(document.body.textContent).not.toMatch(/helper|handoff|journal/i)
  await expect
    .element(page.getByRole('button', { name: 'Try again' }))
    .toBeVisible()
  await expect
    .element(page.getByRole('button', { name: 'Dismiss' }))
    .toBeVisible()
})
