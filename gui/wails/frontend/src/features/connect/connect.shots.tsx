import { test } from 'vitest'
import { fixture } from '@/components/test-view'
import { capturePage } from '@/components/page-capture'
import resetFrame from '@/../../../tests/scenarios/recordings/legacy-key/live-lan-key-reset.json'
import failedFrame from '@/../../../tests/scenarios/recordings/legacy-key/live-lan-key-store-failed.json'

for (const name of ['ready-idle', 'lan-on']) {
  test(`Connect recorded ${name}`, () =>
    capturePage({ feature: 'connect', scenario: name, view: fixture(name) }))
}
test('Connect legacy recorded key reset', () => {
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
  return capturePage({
    feature: 'connect',
    scenario: 'live-lan-key-reset',
    view,
  })
})
test('Connect legacy recorded key store failure', () => {
  const view = fixture()
  view.key.unavailable = failedFrame.lan.unavailable
  return capturePage({
    feature: 'connect',
    scenario: 'live-lan-key-store-failed',
    view,
  })
})
