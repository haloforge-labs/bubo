import type { RegistryRow, SwitchRow } from '@/lib/registry'

// Synthetic extension, never a copy of a real switch's facts.
export const dummyRow = {
  name: '--fixture-size',
  purpose: 'Choose a fixture size.',
  audience: 'user',
  group: 'Other',
  gui_control: true,
  allowed: { type: 'integer', minimum: 3, maximum: 17 },
} satisfies RegistryRow & Pick<SwitchRow, 'allowed'>
