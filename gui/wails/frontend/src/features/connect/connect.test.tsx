import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { actions, bindAction, viewKey, type View } from '@/lib/bindings'
import { fixture, renderView } from '@/components/test-view'
import resetFrame from '@/../../../tests/scenarios/recordings/legacy-key/live-lan-key-reset.json'
import failedFrame from '@/../../../tests/scenarios/recordings/legacy-key/live-lan-key-store-failed.json'
import ConnectPage from './index'
import { snippetWithKey } from './key-controls'
import { fillSnippet } from './snippets'

let cleanup = () => {}
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

function networkView(): View {
  const view = fixture('lan-on')
  const lan = resetFrame.lan
  view.key = {
    lan_access: lan.lan_access,
    lan_active: lan.lan_active,
    present: lan.api_key_present,
    mask: lan.api_key_mask,
    copy_with_key: lan.copy_with_key,
    restart_needed: lan.key_restart_needed,
    addresses: lan.addresses,
  }
  return view
}

// The recorded status lists client templates, and each one has its own
// "Copy address and key" button. These tests mean the one in the API key group.
function apiKeyCopy() {
  return page
    .getByRole('region', { name: 'API key', exact: true })
    .getByRole('button', { name: 'Copy address and key', exact: true })
}

test('addresses are copied exactly as supplied and missing URLs stay unavailable', async () => {
  const view = fixture()
  view.endpoints = {
    openai_base_url: 'https://example.test/openai',
    anthropic_base_url: 'https://example.test/anthropic',
  }
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, view).cleanup
  await page
    .getByRole('button', { name: 'Copy OpenAI address', exact: true })
    .click()
  expect(clipboard).toHaveBeenLastCalledWith(view.endpoints.openai_base_url)
  await page
    .getByRole('button', { name: 'Copy Anthropic address', exact: true })
    .click()
  expect(clipboard).toHaveBeenLastCalledWith(view.endpoints.anthropic_base_url)
})

test('LAN off never reveals or copies a key and uses partial preferences for LAN', async () => {
  const reveal = vi.spyOn(actions, 'RevealKey')
  const save = vi.spyOn(actions, 'PatchPreferences').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, fixture()).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Copy address and key' }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Show', exact: true }))
    .not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Copy OpenAI address' }))
    .toBeEnabled()
  await page.getByRole('switch', { name: 'Local network access' }).click()
  expect(save).toHaveBeenCalledExactlyOnceWith({
    set: { lan_access: true },
    clear: [],
  })
  expect(reveal).not.toHaveBeenCalled()
})

test('ported live LAN key reset frame stays masked; only Copy address and key reaches the clipboard', async () => {
  const planted = 'PLANTED-KEY-TEST-ONLY'
  const reveal = vi
    .spyOn(actions, 'RevealKey')
    .mockResolvedValue({ value: planted })
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const view = networkView()
  const original = structuredClone(view)
  cleanup = renderView(<ConnectPage />, view).cleanup
  await expect
    .element(page.getByTestId('key-value'))
    .toHaveTextContent(resetFrame.lan.api_key_mask)
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('used after restart')
  expect(document.body.textContent).not.toContain(planted)
  await page.getByRole('button', { name: 'Copy snippet', exact: true }).click()
  expect(clipboard).toHaveBeenLastCalledWith(view.pi_snippet)
  expect(reveal).not.toHaveBeenCalled()
  await apiKeyCopy().click()
  await expect.poll(() => clipboard.mock.calls.at(-1)?.[0]).toContain(planted)
  expect(reveal).toHaveBeenCalledTimes(1)
  expect(document.body.textContent).not.toContain(planted)
  expect(view).toEqual(original)
})

test('a revealed key clears on Hide and unmount and is never copied by Show', async () => {
  const planted = 'PLANTED-KEY-TEST-ONLY'
  vi.spyOn(actions, 'RevealKey').mockResolvedValue({ value: planted })
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  const mounted = renderView(<ConnectPage />, networkView())
  cleanup = mounted.cleanup
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect.element(page.getByTestId('key-value')).toHaveTextContent(planted)
  expect(clipboard).not.toHaveBeenCalled()
  await page.getByRole('button', { name: /Hide/ }).click()
  await expect
    .element(page.getByTestId('key-value'))
    .not.toHaveTextContent(planted)
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect.element(page.getByTestId('key-value')).toHaveTextContent(planted)
  await (await mounted.screen).unmount()
  expect(document.body.textContent).not.toContain(planted)
})

test('Keep showing holds the key until Hide', async () => {
  vi.spyOn(actions, 'RevealKey').mockResolvedValue({
    value: 'PLANTED-KEY-TEST-ONLY',
  })
  cleanup = renderView(<ConnectPage />, networkView()).cleanup
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect
    .element(page.getByTestId('key-value'))
    .toHaveTextContent('PLANTED-KEY-TEST-ONLY')
  vi.useFakeTimers({
    toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'],
  })
  await page.getByRole('button', { name: 'Keep showing' }).click()
  await vi.advanceTimersByTimeAsync(31_000)
  await expect
    .element(page.getByTestId('key-value'))
    .toHaveTextContent('PLANTED-KEY-TEST-ONLY')
  await page.getByRole('button', { name: /Hide/ }).click()
  vi.useRealTimers()
  await expect
    .element(page.getByTestId('key-value'))
    .not.toHaveTextContent('PLANTED-KEY-TEST-ONLY')
})

test('Show automatically hides the key after 30 seconds', async () => {
  vi.spyOn(actions, 'RevealKey').mockResolvedValue({
    value: 'PLANTED-KEY-TEST-ONLY',
  })
  cleanup = renderView(<ConnectPage />, networkView()).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Show', exact: true }))
    .toBeVisible()
  vi.useFakeTimers({
    toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'],
  })
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect
    .element(page.getByTestId('key-value'))
    .toHaveTextContent('PLANTED-KEY-TEST-ONLY')
  await vi.advanceTimersByTimeAsync(30_001)
  vi.useRealTimers()
  await expect
    .element(page.getByTestId('key-value'))
    .not.toHaveTextContent('PLANTED-KEY-TEST-ONLY')
})

test('a pending Show response cannot reveal a key after LAN turns off', async () => {
  let resolve: (key: string) => void = () => {}
  vi.spyOn(actions, 'RevealKey').mockImplementation(
    () =>
      new Promise((done) => {
        resolve = (value) => done({ value })
      }),
  )
  const mounted = renderView(<ConnectPage />, networkView())
  cleanup = mounted.cleanup
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  const off = fixture()
  mounted.client.setQueryData(viewKey, off)
  await expect
    .element(page.getByRole('button', { name: 'Copy address and key' }))
    .not.toBeInTheDocument()
  resolve('PLANTED-LATE-KEY-TEST-ONLY')
  await expect
    .poll(() => document.body.textContent)
    .not.toContain('PLANTED-LATE-KEY-TEST-ONLY')
})

test('hiding the page clears a shown key', async () => {
  vi.spyOn(actions, 'RevealKey').mockResolvedValue({
    value: 'PLANTED-KEY-TEST-ONLY',
  })
  cleanup = renderView(<ConnectPage />, networkView()).cleanup
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect
    .element(page.getByTestId('key-value'))
    .toHaveTextContent('PLANTED-KEY-TEST-ONLY')
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
  document.dispatchEvent(new Event('visibilitychange'))
  await expect
    .element(page.getByTestId('key-value'))
    .not.toHaveTextContent('PLANTED-KEY-TEST-ONLY')
})

test('secret transport failures stay typed and never echo the rejected payload', async () => {
  const planted = 'PLANTED-ERROR-KEY-TEST-ONLY'
  vi.spyOn(actions, 'RevealKey').mockImplementation(
    bindAction(async () => {
      throw planted
    }),
  )
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, networkView()).cleanup
  await page.getByRole('button', { name: 'Show', exact: true }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Bubo could not connect to the desktop service')
  await apiKeyCopy().click()
  await expect
    .element(page.getByRole('alert').last())
    .toHaveTextContent('binding_unavailable:')
  expect(document.body.textContent).not.toContain(planted)
  expect(clipboard).not.toHaveBeenCalled()
})

test('ported key store failure stays off without Copy address and key', async () => {
  const view = fixture()
  view.key.unavailable = failedFrame.lan.unavailable
  cleanup = renderView(<ConnectPage />, view).cleanup
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent("Windows couldn't read or save the API key.")
  // The raw text is under Details, with the code.
  const raw = page.getByText(
    `key_store_failed: ${failedFrame.lan.unavailable}`,
    { exact: true },
  )
  expect(raw.element().closest('details')).not.toBeNull()
  expect(
    page.getByRole('button', { name: 'Try again' }).elements(),
  ).toHaveLength(0)
  await expect
    .element(page.getByRole('switch', { name: 'Local network access' }))
    .not.toBeChecked()
  await expect
    .element(page.getByRole('button', { name: 'Copy address and key' }))
    .not.toBeInTheDocument()
})

test('Reset is confirmed and canceled reset never calls the binding', async () => {
  const reset = vi.spyOn(actions, 'ResetKey').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, networkView()).cleanup
  await page.getByRole('button', { name: 'Reset key…', exact: true }).click()
  await expect.element(page.getByRole('dialog')).toBeVisible()
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  expect(reset).not.toHaveBeenCalled()
  await page.getByRole('button', { name: 'Reset key…', exact: true }).click()
  await page.getByRole('button', { name: 'Reset key', exact: true }).click()
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
  expect(reset).toHaveBeenCalledTimes(1)
})

test('unsupported keyed snippet is a typed failure without inventing client configuration', () => {
  expect(() => snippetWithKey('{"client":"fixture"}', 'PLANTED-KEY')).toThrow()
})

test('CLI templates substitute endpoints, preserving literal code braces; a keyless server gets its default key, a keyed one keeps the placeholder', () => {
  const view = fixture('cli-page-facts-loopback')
  for (const snippet of view.client_snippets!) {
    expect(snippet.requires_key).toBe(false)
    const shown = fillSnippet(
      view,
      snippet.template,
      '',
      undefined,
      snippet.default_api_key,
    )
    expect(shown).not.toMatch(/\{(?:openai|anthropic)_base_url\}/)
    // The CLI's loopback dummy key fills the template; no placeholder shows.
    expect(snippet.default_api_key).toBeTruthy()
    expect(shown).not.toContain('{api_key}')
    expect(shown).toContain(snippet.default_api_key!)
  }
  // With a key required the placeholder stays until the explicit keyed copy.
  const keyed = fixture('cli-page-facts-authenticated')
  for (const snippet of keyed.client_snippets!) {
    expect(snippet.requires_key).toBe(true)
    const plain = fillSnippet(keyed, snippet.template)
    expect(plain).toContain('{api_key}')
    expect(fillSnippet(keyed, snippet.template, '', 'PLANTED-KEY')).toBe(
      plain.replaceAll('{api_key}', 'PLANTED-KEY'),
    )
  }
  view.endpoints = undefined
  expect(() => fillSnippet(view, '{openai_base_url}')).toThrow()
})

test('the Connect page shows and copies the default key of a keyless server, never a placeholder', async () => {
  const view = fixture('cli-page-facts-loopback')
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, view).cleanup
  const snippet = view.client_snippets![0]!
  const section = page.getByRole('region', { name: snippet.title, exact: true })
  await expect.element(section).toHaveTextContent(snippet.default_api_key!)
  expect(document.body.textContent).not.toContain('{api_key}')
  await section.getByRole('button', { name: 'Copy', exact: true }).click()
  const copied = clipboard.mock.calls.at(-1)![0]
  expect(copied).toContain(snippet.default_api_key!)
  expect(copied).not.toContain('{api_key}')
})

test('CLI client templates copy selected LAN endpoints and include a key only on explicit keyed copy', async () => {
  const view = fixture('cli-page-facts-authenticated')
  view.key = {
    ...networkView().key,
    addresses: [
      {
        adapter: 'Fixture LAN',
        ipv4: '192.0.2.10',
        url: 'https://lan.example.test:9443',
      },
    ],
  }
  const original = structuredClone(view)
  const copy = vi
    .spyOn(actions, 'RevealKey')
    .mockResolvedValue({ value: 'PLANTED-KEY' })
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, view).cleanup
  await page
    .getByRole('combobox', { name: 'Address for client templates' })
    .selectOptions('https://lan.example.test:9443')
  const snippet = view.client_snippets![0]!
  const section = page.getByRole('region', { name: snippet.title, exact: true })
  await section.getByRole('button', { name: 'Copy', exact: true }).click()
  expect(clipboard).toHaveBeenLastCalledWith(
    fillSnippet(view, snippet.template, 'https://lan.example.test:9443'),
  )
  await section
    .getByRole('button', { name: 'Copy address and key', exact: true })
    .click()
  expect(copy).toHaveBeenCalledOnce()
  expect(clipboard).toHaveBeenLastCalledWith(
    fillSnippet(
      view,
      snippet.template,
      'https://lan.example.test:9443',
      'PLANTED-KEY',
    ),
  )
  expect(document.body.textContent).not.toContain('PLANTED-KEY')
  expect(view).toEqual(original)
})

test('a LAN preference rejection stays next to the toggle and does not enable key controls', async () => {
  vi.spyOn(actions, 'PatchPreferences').mockResolvedValue(bridgeFailure())
  cleanup = renderView(<ConnectPage />, fixture()).cleanup
  await page.getByRole('switch', { name: 'Local network access' }).click()
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('fixture_rejected: Check network preferences.')
  await expect
    .element(page.getByRole('switch', { name: 'Local network access' }))
    .not.toBeChecked()
})

function bridgeFailure() {
  return {
    error: {
      code: 'fixture_rejected',
      component: 'fixture',
      stage: 'save',
      retryable: false,
      remedy: 'Check network preferences.',
    },
  }
}

test('client templates with LAN off offer only plain copy and never fetch a key', async () => {
  const view = fixture('cli-page-facts-authenticated')
  const key = vi.spyOn(actions, 'RevealKey')
  const clipboard = vi.spyOn(actions, 'CopyText').mockResolvedValue({})
  cleanup = renderView(<ConnectPage />, view).cleanup
  await expect
    .element(page.getByRole('button', { name: 'Copy address and key' }))
    .not.toBeInTheDocument()
  const snippet = view.client_snippets![0]!
  await page
    .getByRole('region', { name: snippet.title, exact: true })
    .getByRole('button', { name: 'Copy', exact: true })
    .click()
  expect(clipboard).toHaveBeenLastCalledWith(
    fillSnippet(view, snippet.template),
  )
  expect(key).not.toHaveBeenCalled()
})
