import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { fixture, renderView } from '@/components/test-view'
import AboutPage from '@/features/about'

test.runIf(import.meta.env.VITE_V8_GATE === '1')(
  'F14 rollback pending offers an enabled Recover control on About',
  async () => {
    const view = fixture('update-rollback_pending')
    expect(view.update.state).toBe('rollback_pending')
    const mounted = renderView(<AboutPage />, view)
    try {
      await expect
        .element(page.getByRole('button', { name: 'Recover', exact: true }))
        .toBeEnabled()
    } finally {
      mounted.cleanup()
    }
  },
)
