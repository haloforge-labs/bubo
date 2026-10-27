import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'
import { actions } from '@/lib/bindings'
import { folderPath } from '@/lib/format'
import { bridgeReply } from '@/components/bridge-test-reply'

afterEach(() => vi.restoreAllMocks())

test('Check again refreshes a refused draft estimate rather than the saved setup', async () => {
  const view = coreView('stopped-saved-estimate')
  view.saved.manifest_path = view.prepared_models[0]!.manifest_path
  const memory = view.saved_estimate.memory_estimate!
  const context = memory.recommended_context!.minimum_context_tokens
  const estimate = vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: {
        memory_estimate: {
          ...memory,
          context_tokens: context,
          fits: false,
          fit_reason: 'Fixture draft refusal.',
        },
      },
    }),
  )
  const refresh = vi.spyOn(actions, 'RefreshModels').mockResolvedValue({})
  mountCorePage('/', view)
  await page
    .getByRole('spinbutton', { name: 'Context per session' })
    .fill(String(context))
  await expect.poll(() => estimate.mock.calls.length).toBe(1)
  await page.getByRole('button', { name: 'Check again', exact: true }).click()
  await expect.poll(() => estimate.mock.calls.length).toBe(2)
  expect(refresh).not.toHaveBeenCalled()
})

test('an unchecked saved fit keeps the allowed Start action and raw details', async () => {
  const view = coreView('stopped-saved-estimate')
  const memory = view.saved_estimate.memory_estimate!
  view.saved.context = memory.context_tokens
  view.saved.sessions = memory.sessions
  view.saved_estimate = {
    reason: 'Fixture setup is not estimated.',
    error: {
      code: 'estimate_unavailable',
      remedy: 'Fixture setup is not estimated.',
      component: 'models',
      stage: 'estimate',
      retryable: false,
    },
  }
  view.actions.start = {
    allowed: true,
    reason_code: 'fit_unchecked',
    reason: 'The server checks memory when it starts.',
  }
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeEnabled()
  await page.getByText('Memory check details', { exact: true }).click()
  await expect
    .element(
      page.getByText('estimate_unavailable: Fixture setup is not estimated.', {
        exact: true,
      }),
    )
    .toBeVisible()
})

test('running identity and path stay read-only while the saved picker is for next start', async () => {
  const view = coreView()
  const saved = view.prepared_models[0]!
  view.saved.manifest_path = saved.manifest_path
  view.current_model = {
    catalogue_id: '',
    display_name: 'serving.gguf',
    model_path: 'D:/fixture/serving.gguf',
    manifest_path: '',
  }
  mountCorePage('/', view)
  const identity = page.getByRole('region', {
    name: 'Model in use',
    exact: true,
  })
  await expect.element(identity).toHaveTextContent('Model: serving.gguf')
  await expect
    .element(identity)
    .toHaveTextContent(folderPath(view.current_model.model_path))
  await expect
    .element(page.getByRole('combobox', { name: 'Model for next start' }))
    .toHaveValue(saved.manifest_path!)
  await expect.element(identity.getByRole('combobox')).not.toBeInTheDocument()
})

test('external address falls back to the exact connection URL once on Server', async () => {
  const view = coreView('external-server')
  view.endpoints = undefined
  expect(view.connection?.endpoint).toBeTruthy()
  const copy = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  mountCorePage('/', view)
  const addresses = page.getByRole('region', { name: 'Addresses', exact: true })
  await expect.element(addresses).toHaveTextContent(view.connection!.endpoint)
  await addresses.getByRole('button', { name: 'Copy OpenAI address' }).click()
  expect(copy).toHaveBeenCalledExactlyOnceWith(view.connection!.endpoint)
  expect(
    page.getByRole('button', { name: 'Copy OpenAI address' }).elements(),
  ).toHaveLength(1)
  await expect
    .element(page.getByLabelText('Server addresses'))
    .not.toBeInTheDocument()
})

test('reported completed-request rates retain their scope', async () => {
  const view = coreView('ready-busy')
  mountCorePage('/', view)
  const request = view.requests.at(-1)!
  const reported = page.getByRole('region', { name: 'Reported request rates' })
  await expect.element(reported).toHaveTextContent('Last request ·')
  expect(reported.element().textContent).not.toContain(request.id)
  await expect.element(reported).toHaveTextContent('ago)')
  expect(reported.element().textContent).not.toContain('this request')
  await expect
    .element(reported)
    .toHaveTextContent(request.decode_tps!.toFixed(1))
})

test('the saved refusal sits by the inputs and its recommended context is a one-click edit', async () => {
  const view = coreView('memory-smaller-context')
  view.saved.manifest_path = view.prepared_models[0]!.manifest_path
  const memory = view.saved_estimate.memory_estimate!
  const suggested = view.fit.suggested_context!
  vi.spyOn(actions, 'Estimate').mockResolvedValue(
    bridgeReply({
      estimate: {
        memory_estimate: { ...memory, context_tokens: suggested, fits: true },
      },
      fit: {
        ...view.fit,
        class: 'fits',
        sentence: 'Fits.',
        suggested_context: undefined,
      },
    }),
  )
  mountCorePage('/', view)
  await expect
    .element(page.getByRole('button', { name: 'Start server', exact: true }))
    .toBeDisabled()
  // The saved and draft verdicts share one panel beside the setup inputs.
  await expect.element(page.getByLabelText('Memory fit')).toBeVisible()
  // Nothing changed, so Check again could not change the answer.
  await expect
    .element(page.getByRole('button', { name: 'Check again', exact: true }))
    .not.toBeInTheDocument()
  await page
    .getByRole('button', {
      name: `Use ${suggested.toLocaleString()} tokens`,
      exact: true,
    })
    .click()
  await expect
    .element(page.getByRole('spinbutton', { name: 'Context per session' }))
    .toHaveValue(suggested.toLocaleString())
  // The draft's own verdict appears in the form once it is known.
  await expect.element(page.getByLabelText('Memory fit')).toBeVisible()
  expect(
    document
      .querySelector('form')!
      .contains(page.getByLabelText('Memory fit').element()),
  ).toBe(true)
})

test('GPU timing is supplied by the View and powers use whole watts', async () => {
  const view = coreView('ready-busy')
  view.gpu = {
    use_pct: 0,
    chip_power_w: 12.5,
    age_seconds: 0.25,
    window_seconds: 1.5,
  }
  mountCorePage('/', view)
  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await page.getByRole('tab', { name: 'Advanced' }).click()
  await expect
    .element(page.getByText('GPU sample:', { exact: false }))
    .toHaveTextContent('0.3 s old · over 1.5 s')
  await expect
    .element(
      page.getByRole('figure', {
        name: 'Chip power (CPU, GPU, NPU)',
        exact: true,
      }),
    )
    .toHaveTextContent('13 W')
})
