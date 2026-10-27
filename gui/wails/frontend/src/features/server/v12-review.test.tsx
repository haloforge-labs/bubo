import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import {
  coreView,
  mountCorePage,
  legacyProfile,
} from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { MemoryFit } from './components/memory-fit'
import { renderView } from '@/components/test-view'

afterEach(() => vi.restoreAllMocks())

test('Mode uses only CLI GUI profiles and resolves the saved stock alias', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.profile = legacyProfile.profile
  view.switch_overrides.find((row) => row.name === '--profile')!.saved =
    view.saved.profile
  mountCorePage('/settings', view)
  await page.getByRole('tab', { name: 'Advanced' }).click()
  const mode = page.getByRole('combobox', { name: 'Mode', exact: true })
  await expect
    .poll(() =>
      Array.from((mode.element() as HTMLSelectElement).options).map(
        (option) => option.text,
      ),
    )
    .toEqual(
      view.profiles
        .filter((profile) => profile.gui)
        .map((profile) => profile.label),
    )
  await expect
    .poll(() => (mode.element() as HTMLSelectElement).selectedOptions[0]?.text)
    .toBe(view.profiles.find((profile) => profile.name === 'vanilla')!.label)
})

test('largest context never offers a CLI row that fits but is not recommended', async () => {
  const view = coreView('stopped-saved-estimate')
  const memory = view.saved_estimate.memory_estimate!
  const row = memory.recommended_context!.by_sessions.find(
    (row) => row.sessions === memory.sessions,
  )!
  row.fits = true
  row.recommended = false
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Use largest context' }))
    .not.toBeInTheDocument()
  await expect
    .element(
      page.getByText(
        `no recommended context for ${row.sessions} ${row.sessions === 1 ? 'session' : 'sessions'}.`,
        { exact: false },
      ),
    )
    .toBeVisible()
})

test('no context for the selected session count still offers a reported fewer-session fix', async () => {
  const view = coreView('no-fit')
  const onSessions = vi.fn()
  const fit = {
    ...view.fit,
    class: 'memory_does_not_fit_any',
    sessions: 2,
    context_tokens: 65536,
  }
  const memory = coreView('stopped-saved-estimate').saved_estimate
    .memory_estimate!
  const estimate = {
    ...memory,
    recommended_context: {
      ...memory.recommended_context!,
      by_sessions: [
        {
          ...memory.recommended_context!.by_sessions[0]!,
          sessions: 1,
          context_per_session: 65536,
          recommended: true,
          fits: true,
        },
      ],
    },
  }
  const mounted = renderView(
    <MemoryFit
      view={view}
      fit={fit}
      estimate={estimate}
      dirty={false}
      onContext={vi.fn()}
      onSessions={onSessions}
    />,
    view,
  )
  try {
    await page.getByRole('button', { name: 'Use fewer sessions (1)' }).click()
    expect(onSessions).toHaveBeenCalledExactlyOnceWith(1)
  } finally {
    mounted.cleanup()
  }
})

test('busy port refuses non-integer and out-of-range input before any write', async () => {
  const view = coreView('stopped-saved-estimate')
  view.failure = {
    code: 'port_occupied',
    remedy: '',
    source: 'fixture',
    time: '',
    forced: false,
  }
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  mountCorePage('/', view)
  const field = page.getByLabelText('New server port')
  const button = page.getByRole('button', { name: 'Save port', exact: true })
  for (const value of ['', 'abc', '0', '-1', '65536', '9.5', '1e3']) {
    await field.fill(value)
    await expect.element(button).toBeDisabled()
  }
  expect(save).not.toHaveBeenCalled()
  for (const value of ['1', '65535']) {
    await field.fill(value)
    await expect.element(button).toBeEnabled()
  }
})
