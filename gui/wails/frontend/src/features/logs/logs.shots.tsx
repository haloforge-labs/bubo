import { test } from 'vitest'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'

for (const name of ['ready-idle', 'failed-with-remedy']) {
  test(`Logs recorded ${name}`, () =>
    capturePage({ feature: 'logs', scenario: name, view: fixture(name) }))
}
