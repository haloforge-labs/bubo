import { test } from 'vitest'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

for (const name of ['ready-busy', 'ready-idle']) {
  test(`Requests recorded ${name}`, () =>
    capturePage({ feature: 'requests', scenario: name, view: fixture(name) }))
}
