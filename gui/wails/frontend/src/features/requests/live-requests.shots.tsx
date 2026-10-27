import { expect, inject, test } from 'vitest'
import { page } from 'vitest/browser'
import { capturePage } from '@/components/page-capture'
import { liveRequestView } from './live-request-test-support'
// These Views must be exported by the supervisor recording replay. A missing
// export fails the capture instead of silently substituting invented rates.
for (const phase of ['reading', 'writing', 'completed'] as const) {
  test(`Requests ${phase} from the production supervisor replay`, async () => {
    const view = inject('liveRequestViews')?.[phase]
    expect(
      view,
      'BUBO_LIVE_SHOTS_INPUT_DIRECTORY production View',
    ).toBeDefined()
    await capturePage({
      feature: 'requests',
      scenario: `live-${phase}`,
      view: view!,
      prepare: async () => {
        await expect
          .element(
            page.getByRole('figure', { name: /^Live request rates/ }).first(),
          )
          .toBeVisible()
        if (phase !== 'completed') {
          await expect
            .element(
              page.getByRole('region', { name: 'Live requests', exact: true }),
            )
            .toHaveTextContent(phase === 'reading' ? 'Reading' : 'Writing')
        } else {
          await expect
            .element(
              page.getByRole('figure', {
                name: 'Decode per completed request',
                exact: true,
              }),
            )
            .toBeVisible()
        }
      },
    })
  })
}

test('Requests known-rate layout with explicitly arranged observation rates', async () => {
  await capturePage({
    feature: 'requests',
    scenario: 'live-writing-arranged-known-rate',
    view: liveRequestView(),
    prepare: async () => {
      await expect
        .element(
          page.getByRole('region', { name: 'Live requests', exact: true }),
        )
        .toHaveTextContent('54.0')
      await expect
        .element(
          page.getByRole('figure', { name: /^Live request rates/ }).first(),
        )
        .toBeVisible()
    },
  })
})

for (const phase of ['reading', 'writing', 'completed'] as const) {
  test(`Server header ${phase} from the production supervisor replay`, async () => {
    const view = inject('liveRequestViews')?.[phase]
    expect(
      view,
      'BUBO_LIVE_SHOTS_INPUT_DIRECTORY production View',
    ).toBeDefined()
    await capturePage({
      feature: 'server',
      scenario: `live-header-${phase}`,
      view: view!,
      prepare: async () => {
        await expect
          .element(page.getByLabelText('Server status'))
          .toHaveTextContent(
            phase === 'writing'
              ? 'Writing'
              : phase === 'reading'
                ? 'Reading prompt'
                : 'Idle',
          )
      },
    })
  })
}
