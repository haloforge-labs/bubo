import { afterEach, expect, inject, test, vi } from 'vitest'
import { commands, page } from 'vitest/browser'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

test('Server saved setup uses identity and reports restart from the View', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  const model = view.prepared_models[0]!
  view.active_model = {
    catalogue_id: model.id,
    manifest_path: model.manifest_path,
  }
  view.saved.sessions = 3
  const recommendation =
    view.memory_estimate!.recommended_context!.by_sessions.find(
      (row) => row.sessions === view.saved.sessions && row.fits,
    )!
  view.saved.context = recommendation.context_per_session
  view.restart_needed = [
    { key: 'sessions', label: 'Sessions', running: '1', next: '3' },
  ]
  view.saved_estimate = {
    memory_estimate: {
      ...view.memory_estimate!,
      fits: true,
      sessions: view.saved.sessions,
      context_tokens: view.saved.context,
    },
  }
  view.actions.restart = { allowed: true, reason_code: '', reason: '' }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Restart server', exact: true }))
    .toBeEnabled()
  await document.fonts.ready
  await commands.captureFeature('server', 'setup-saved-restart')
})

test('Server setup typed rejection stays at the editing form', async () => {
  document.documentElement.dataset.theme = inject('theme')
  const view = coreView()
  view.active_model = {
    catalogue_id: view.prepared_models[0]!.id,
    manifest_path: view.prepared_models[0]!.manifest_path,
  }
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: { memory_estimate: { ...view.memory_estimate!, fits: true } },
    }),
  )
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({
    error: {
      code: 'fixture_rejected',
      component: 'fixture',
      stage: 'save',
      retryable: false,
      remedy: 'Review the server setup.',
    },
  })
  mountCorePage('/', view)
  await page
    .getByRole('spinbutton', { name: 'Sessions', exact: true })
    .fill('3')
  await page.getByRole('button', { name: 'Save server setup' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Review the server setup.')
  await document.fonts.ready
  await commands.captureFeature('server', 'setup-rejected')
})
