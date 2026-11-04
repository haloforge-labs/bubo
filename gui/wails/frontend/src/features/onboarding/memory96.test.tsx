import { afterEach, expect, test, vi } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

afterEach(() => vi.restoreAllMocks())

for (const split of ['64', '72']) {
  test(`96 GiB / ${split} VGM keeps three real options and their refusal reasons`, async () => {
    const view = coreView(`onboarding-96-${split}`)
    mountCorePage('/onboarding', view)
    const select = page.getByLabelText('Model to download')
    await expect
      .element(select)
      .toHaveValue('huihui-qwen38-flash-next-lowmemory-v1')
    const element = select.element() as HTMLSelectElement
    const options = [...element.options]
    expect(options).toHaveLength(3)
    const full = options.filter(
      (option) => option.value !== 'huihui-qwen38-flash-next-lowmemory-v1',
    )
    expect(full).toHaveLength(2)
    for (const option of full) {
      expect(option.disabled).toBe(true)
      expect(option.textContent).toContain(
        'Does not fit this computer: needs about 91 GiB of GPU memory. Use the lower-memory model.',
      )
    }
    expect(options.find((option) => option.selected)?.disabled).toBe(false)
  })
}

test('128 GiB keeps its existing first-run choice and enabled full-size options', async () => {
  const view = coreView('onboarding-128-96')
  mountCorePage('/onboarding', view)
  const select = page.getByLabelText('Model to download')
  await expect
    .element(select)
    .toHaveValue('huihui-qwen38-flash-next-ud-q4-k-xl-v1')
  for (const option of [...(select.element() as HTMLSelectElement).options]) {
    expect(option.disabled).toBe(false)
    expect(option.textContent).not.toContain('Does not fit this computer')
  }
})

test('first-run choices leave out a supported entry that has its own family group', async () => {
  const view = coreView('onboarding-96-64')
  const template = view.models!.catalogue.find(
    (entry) => entry.id === 'huihui-qwen38-flash-next-lowmemory-v1',
  )!
  view.models!.catalogue.push({
    ...template,
    id: 'fixture-coder-entry',
    display_name: 'Fixture Coder',
    selection_label: 'Fixture Coder',
    family_group: 'fixture-coder-group',
  })
  mountCorePage('/onboarding', view)
  const select = page.getByLabelText('Model to download')
  await expect.element(select).toBeVisible()
  const element = select.element() as HTMLSelectElement
  expect(element.options).toHaveLength(3)
  expect([...element.options].map((option) => option.value)).not.toContain(
    'fixture-coder-entry',
  )
})
