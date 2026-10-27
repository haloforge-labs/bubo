import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { coreView, legacyProfile } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('V12 review: saved stock displays the CLI Vanilla label', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.profile = legacyProfile.profile
  view.switch_overrides.find((row) => row.name === '--profile')!.saved =
    view.saved.profile
  await capturePage({
    feature: 'settings',
    scenario: 'v12-review-vanilla',
    view,
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click()
      const mode = page.getByRole('combobox', { name: 'Mode', exact: true })
      await expect
        .poll(
          () => (mode.element() as HTMLSelectElement).selectedOptions[0]?.text,
        )
        .toBe('Vanilla (no Bubo tuning)')
      mode.element().scrollIntoView({ block: 'center' })
    },
  })
})

test('V12 review: an unrecommended context is not offered', async () => {
  const view = coreView('stopped-saved-estimate')
  const memory = view.saved_estimate.memory_estimate!
  const row = memory.recommended_context!.by_sessions.find(
    (row) => row.sessions === memory.sessions,
  )!
  row.recommended = false
  await capturePage({
    feature: 'server',
    scenario: 'v12-review-no-recommended-context',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('button', { name: 'Use largest context' }))
        .not.toBeInTheDocument()
      page
        .getByLabelText('Context per session')
        .element()
        .scrollIntoView({ block: 'center' })
    },
  })
})

test('V12 review: fewer sessions remain available when no context fits the selected count', async () => {
  const view = coreView('no-fit')
  const memory = coreView('stopped-saved-estimate').saved_estimate
    .memory_estimate!
  view.fit = {
    ...view.fit,
    class: 'memory_does_not_fit_any',
    sessions: 2,
    context_tokens: 65536,
  }
  view.saved_estimate.memory_estimate = {
    ...memory,
    fits: false,
    sessions: 2,
    recommended_context: {
      ...memory.recommended_context!,
      by_sessions: [
        {
          ...memory.recommended_context!.by_sessions[0]!,
          sessions: 1,
          context_per_session: 65536,
          fits: true,
          recommended: true,
        },
        {
          ...memory.recommended_context!.by_sessions[0]!,
          sessions: 2,
          context_per_session: 4096,
          fits: false,
          recommended: false,
        },
      ],
    },
  }
  await capturePage({
    feature: 'server',
    scenario: 'v12-review-fewer-sessions',
    view,
    prepare: async () => {
      await expect
        .element(page.getByRole('button', { name: 'Use fewer sessions (1)' }))
        .toBeVisible()
      page
        .getByRole('region', { name: 'Memory fit' })
        .element()
        .scrollIntoView({ block: 'start' })
    },
  })
})

test('V12 review: invalid busy port explains the accepted range', async () => {
  const view = coreView('stopped-saved-estimate')
  view.failure = {
    code: 'port_occupied',
    remedy: '',
    source: 'supervisor',
    time: '',
    forced: false,
  }
  await capturePage({
    feature: 'server',
    scenario: 'v12-review-invalid-port',
    view,
    prepare: async () => {
      await page.getByLabelText('New server port').fill('65536')
      await expect
        .element(page.getByRole('button', { name: 'Save port', exact: true }))
        .toBeDisabled()
      await expect
        .element(page.getByText('Enter a whole-number port from 1 to 65535.'))
        .toBeVisible()
    },
  })
})
