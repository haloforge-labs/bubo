import { afterEach, expect, test, vi } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { actions } from '@/lib/bindings'
import { bridgeReply } from '@/components/bridge-test-reply'
import {
  coreView,
  mountCorePage,
  recordedPlan,
} from '@/components/core-page-test-support'
import { PlanSummary } from '@/components/model-download'
import type { ModelPlan } from '@/components/model-facts'

const GiB = 1024 ** 3

afterEach(() => vi.restoreAllMocks())

// E23 failure list: the file is already in the package folder and failed its
// pin, yet the consent currently looks exactly like a fresh download.
test('repair consent names the replaced file and counts its download bytes', async () => {
  const base = recordedPlan()
  const plan = {
    ...base,
    files: [
      {
        ...base.files[0]!,
        file_id: 'damaged',
        path: 'C:/fixture/damaged.gguf',
        size: GiB,
        state: 'download',
        replaces: 'SHA-256 differs from the pin',
      },
      {
        ...base.files[0]!,
        file_id: 'good',
        path: 'C:/fixture/good.gguf',
        size: 2 * GiB,
        state: 'reuse',
        replaces: '',
      },
    ],
    required_bytes: 5 * GiB,
    free_bytes: 10 * GiB,
    volumes: base.volumes.map((volume) => ({
      ...volume,
      required_bytes: 5 * GiB,
      free_bytes: 10 * GiB,
    })),
  }
  render(<PlanSummary plan={plan} />)

  const repairs = page.getByRole('list', { name: 'Files to replace' })
  await expect.element(repairs).toBeVisible()
  await expect.element(repairs).toHaveTextContent('damaged.gguf')
  await expect
    .element(repairs)
    .toHaveTextContent('replaces a file that did not match')
  expect(repairs.element().textContent).not.toContain('good.gguf')
  await expect.element(page.getByText('To download: 1.0 GiB')).toBeVisible()
  await expect.element(page.getByText(/Needs 5.0 GiB/)).toBeVisible()
})

test('a fresh download has no replacement notice', async () => {
  const base = recordedPlan()
  render(<PlanSummary plan={base} />)
  await expect
    .element(page.getByRole('list', { name: 'Files to replace' }))
    .not.toBeInTheDocument()
})

// S11 failure list was recorded in REPORT-S11-20261023-0346.md before these
// tests: reported transfer bytes and replacement metadata select the wording,
// and both actions keep the existing confirmation/decline callbacks and their
// pending behaviour. S12 corrects the replacement fixtures to contract states.
function healthyPlan(): ModelPlan {
  const base = recordedPlan()
  return {
    ...base,
    kind: 'repair',
    ready: true,
    files: base.files.map((file) => ({
      ...file,
      state: 'reuse',
      replaces: '',
    })),
    required_bytes: 0,
    free_bytes: 0,
    volumes: [],
    enough_space: true,
  }
}

async function showConsent(plan: ModelPlan, view = coreView('stopped')) {
  const entry = view.models!.catalogue.find((row) => row.id === plan.id)
  view.models!.download = {
    ...coreView('models-download-in-progress').models!.download!,
    id: plan.id,
    kind: plan.kind,
    state: 'awaiting_confirm',
    plan,
  }
  mountCorePage('/models', view)
  const dialog = page.getByRole('dialog')
  await expect.element(dialog).toBeVisible()
  return { dialog, name: entry?.display_name ?? 'model' }
}

async function expectCheck(plan: ModelPlan, view = coreView('stopped')) {
  const { dialog, name } = await showConsent(plan, view)
  await expect
    .element(
      dialog.getByRole('heading', { name: `Check ${name}`, exact: true }),
    )
    .toBeVisible()
  await expect
    .element(
      dialog.getByText('All files match. Nothing to download.', {
        exact: true,
      }),
    )
    .toBeVisible()
  expect(dialog.getByRole('button').elements()).toHaveLength(1)
  await expect
    .element(dialog.getByRole('button', { name: 'OK', exact: true }))
    .toBeEnabled()
  for (const text of [
    'To download',
    'Needs ',
    ' has ',
    'Free space after download',
    'By downloading you accept',
    'Read licence texts',
    'Not now',
  ]) {
    expect(dialog.element().textContent).not.toContain(text)
  }
  expect(
    dialog.getByRole('button', { name: 'Download', exact: true }).elements(),
  ).toHaveLength(0)
  return dialog
}

test('S11: healthy Check shows only OK and keeps every planned file', async () => {
  const plan = healthyPlan()
  const dialog = await expectCheck(plan)
  await dialog
    .getByText(`Files (${plan.files.length})`, { exact: true })
    .click()
  const files = dialog.getByRole('list', { name: 'Download files' })
  await expect.element(files).toBeVisible()
  expect(files.getByRole('listitem').elements()).toHaveLength(plan.files.length)
  for (const file of plan.files) {
    await expect
      .element(files)
      .toHaveTextContent(file.path.split(/[\\/]/).at(-1)!)
  }
})

test.each([
  ['S1 replacement', 'repair', 'download', 'SHA-256 differs from the pin'],
  ['resume', 'repair', 'resume', ''],
  [
    'replacement download',
    'download',
    'download',
    'SHA-256 differs from the pin',
  ],
  ['plain download', 'download', 'download', ''],
])(
  'S11: nonzero %s keeps the transfer consent',
  async (_label, kind, state, replaces) => {
    const base = healthyPlan()
    const plan = {
      ...base,
      kind,
      files: base.files.map((file, index) =>
        index === 0 ? { ...file, size: GiB, state, replaces } : file,
      ),
      required_bytes: 5 * GiB,
      free_bytes: 10 * GiB,
    }
    const { dialog, name } = await showConsent(plan)
    await expect
      .element(
        dialog.getByRole('heading', {
          name: `${kind === 'repair' ? 'Repair' : 'Download'} ${name}`,
          exact: true,
        }),
      )
      .toBeVisible()
    await expect.element(dialog).toHaveTextContent('To download: 1.0 GiB')
    await expect.element(dialog).toHaveTextContent('Needs 5.0 GiB')
    await expect.element(dialog).toHaveTextContent('has 10.0 GiB free')
    await expect
      .element(dialog)
      .toHaveTextContent('Free space after download is not reported by Bubo.')
    await expect
      .element(dialog)
      .toHaveTextContent(
        'By downloading you accept the licence terms for these model files.',
      )
    await expect
      .element(
        dialog.getByText(`Read licence texts (${plan.licences.length})`, {
          exact: true,
        }),
      )
      .toBeVisible()
    expect(dialog.getByRole('button').elements()).toHaveLength(
      kind === 'download' ? 3 : 2,
    )
    if (kind === 'download') {
      await expect
        .element(
          dialog.getByRole('button', { name: 'Change folder…', exact: true }),
        )
        .toBeVisible()
    }
    await expect
      .element(dialog.getByRole('button', { name: 'Not now', exact: true }))
      .toBeEnabled()
    await expect
      .element(dialog.getByRole('button', { name: 'Download', exact: true }))
      .toBeEnabled()
    expect(dialog.element().textContent).not.toContain('All files match.')
    const repairs = dialog.getByRole('list', { name: 'Files to replace' })
    if (replaces) {
      await expect
        .element(repairs)
        .toHaveTextContent('replaces a file that did not match')
      await expect
        .element(repairs)
        .toHaveTextContent(plan.files[0]!.path.split(/[\\/]/).at(-1)!)
      expect(repairs.getByRole('listitem').elements()).toHaveLength(1)
    } else {
      await expect.element(repairs).not.toBeInTheDocument()
    }
  },
)

test('S11: required_bytes zero does not hide a nonzero download', async () => {
  const plan = healthyPlan()
  plan.files[0] = { ...plan.files[0]!, size: GiB, state: 'download' }
  const { dialog } = await showConsent(plan)
  await expect.element(dialog).toHaveTextContent('To download: 1.0 GiB')
  await expect
    .element(dialog.getByRole('button', { name: 'Download', exact: true }))
    .toBeEnabled()
  expect(
    dialog.getByRole('button', { name: 'OK', exact: true }).elements(),
  ).toHaveLength(0)
})

test.each(['download', 'resume'])(
  'S11: a zero-sized %s entry still has no bytes to transfer',
  async (state) => {
    const plan = healthyPlan()
    plan.files[0] = { ...plan.files[0]!, size: 0, state }
    await expectCheck(plan)
  },
)

test.each([
  {
    label: 'unreported file states',
    kind: 'download',
    state: '',
    size: GiB,
    replaces: '',
    summary: 'Planned files: 3.0 GiB',
  },
  {
    label: 'import replacement',
    kind: 'repair',
    state: 'import',
    size: GiB,
    replaces: 'SHA-256 differs from the pin',
    summary: 'To download: 0 B',
  },
  {
    label: 'zero-sized replacement',
    kind: 'repair',
    state: 'download',
    size: 0,
    replaces: 'SHA-256 differs from the pin',
    summary: 'To download: 0 B',
  },
])(
  'S12: $label keeps the transfer consent',
  async ({ kind, state, size, replaces, summary }) => {
    const base = healthyPlan()
    const plan = {
      ...base,
      kind,
      files: base.files.slice(0, 2).map((file, index) => ({
        ...file,
        size: index === 0 ? size : 2 * GiB,
        state: index === 0 || !state ? state : 'reuse',
        replaces: index === 0 ? replaces : '',
      })),
      required_bytes: 5 * GiB,
      free_bytes: 10 * GiB,
    }
    const { dialog, name } = await showConsent(plan)
    await expect
      .element(
        dialog.getByRole('heading', {
          name: `${kind === 'repair' ? 'Repair' : 'Download'} ${name}`,
          exact: true,
        }),
      )
      .toBeVisible()
    await expect.element(dialog).toHaveTextContent(summary)
    await expect.element(dialog).toHaveTextContent('Needs 5.0 GiB')
    await expect.element(dialog).toHaveTextContent('has 10.0 GiB free')
    await expect
      .element(dialog)
      .toHaveTextContent(
        'By downloading you accept the licence terms for these model files.',
      )
    await expect
      .element(
        dialog.getByText(`Read licence texts (${plan.licences.length})`, {
          exact: true,
        }),
      )
      .toBeVisible()
    await expect
      .element(dialog.getByRole('button', { name: 'Not now', exact: true }))
      .toBeEnabled()
    await expect
      .element(dialog.getByRole('button', { name: 'Download', exact: true }))
      .toBeEnabled()
    expect(dialog.element().textContent).not.toContain('All files match.')
    expect(
      dialog.getByRole('button', { name: 'OK', exact: true }).elements(),
    ).toHaveLength(0)
    if (replaces) {
      await expect
        .element(dialog.getByRole('list', { name: 'Files to replace' }))
        .toHaveTextContent('replaces a file that did not match')
    }
  },
)

test('S11: an empty plan has no transfer and does not invent a Files list', async () => {
  const dialog = await expectCheck({ ...healthyPlan(), files: [] })
  expect(dialog.element().textContent).not.toContain('Files (')
  expect(
    dialog.getByRole('list', { name: 'Download files' }).elements(),
  ).toHaveLength(0)
})

test('S11: OK needs no download licence, destination, or free-space approval', async () => {
  const plan = {
    ...healthyPlan(),
    licences: [],
    licence: undefined,
    paths: {},
    enough_space: false,
  }
  const view = coreView('stopped')
  view.models!.catalogue = []
  // The catalogue fallback must not reintroduce the licence guard either.
  const confirm = vi
    .spyOn(actions, 'ConfirmDownload')
    .mockResolvedValue(bridgeReply({}))
  const dialog = await expectCheck(plan, view)
  await dialog.getByRole('button', { name: 'OK', exact: true }).click()
  expect(confirm).toHaveBeenCalledExactlyOnceWith(plan.id)
})

function pendingReply() {
  type Reply = Awaited<ReturnType<typeof actions.ConfirmDownload>>
  let finish!: (result: Reply) => void
  const promise = new Promise<Reply>((resolve) => {
    finish = resolve
  })
  return { promise, finish: () => finish(bridgeReply({})) }
}

test.each(['OK', 'Download'])(
  'S11: %s uses the same confirm action and stays open until it returns',
  async (label) => {
    const plan = label === 'OK' ? healthyPlan() : recordedPlan()
    const pending = pendingReply()
    const confirm = vi
      .spyOn(actions, 'ConfirmDownload')
      .mockReturnValue(pending.promise)
    const cancel = vi.spyOn(actions, 'CancelDownload')
    const { dialog } = await showConsent(plan)
    const button = dialog.getByRole('button', { name: label, exact: true })
    await button.click()
    expect(confirm).toHaveBeenCalledExactlyOnceWith(plan.id)
    await expect.element(button).toBeDisabled()
    await expect.element(dialog).toBeVisible()
    await userEvent.keyboard('{Escape}')
    expect(cancel).not.toHaveBeenCalled()
    await expect.element(dialog).toBeVisible()
    pending.finish()
    await expect.element(dialog).not.toBeInTheDocument()
  },
)

test('S11: Escape declines a healthy Check and waits for that action', async () => {
  const pending = pendingReply()
  const cancel = vi
    .spyOn(actions, 'CancelDownload')
    .mockReturnValue(pending.promise)
  const confirm = vi.spyOn(actions, 'ConfirmDownload')
  const { dialog } = await showConsent(healthyPlan())
  await userEvent.keyboard('{Escape}')
  expect(cancel).toHaveBeenCalledExactlyOnceWith()
  expect(confirm).not.toHaveBeenCalled()
  await expect.element(dialog).toBeVisible()
  pending.finish()
  await expect.element(dialog).not.toBeInTheDocument()
})

test('S11: a failed OK keeps the consent open with the existing action error', async () => {
  vi.spyOn(actions, 'ConfirmDownload').mockResolvedValue(
    bridgeReply({
      error: {
        code: 'integrity_failed',
        remedy: 'Fixture verification failure.',
      },
    }),
  )
  const { dialog } = await showConsent(healthyPlan())
  await dialog.getByRole('button', { name: 'OK', exact: true }).click()
  await expect.element(dialog).toBeVisible()
  await expect
    .element(dialog.getByRole('alert'))
    .toHaveTextContent('The downloaded files did not pass verification.')
  await expect
    .element(dialog.getByRole('button', { name: 'OK', exact: true }))
    .toBeEnabled()
})
