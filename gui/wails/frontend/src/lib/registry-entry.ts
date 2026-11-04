import type { View } from '@/lib/bindings'
import {
  fieldFromRow,
  schemaFromRegistry,
  UnsupportedRegistryError,
  type RegistryRow,
  type RegistryValue,
} from '@/lib/registry'
import {
  effectiveValue,
  registryDefault,
  registryRow,
} from '@/lib/registry-view'
import {
  aboutParagraphs,
  guiValueLabels,
  isTurnOffSwitch,
  settingLabel,
} from '@/lib/presentation'
import { profileAliases } from '@/generated/profile-aliases'

type Row = View['switches'][number]

// The sampling numbers are always editable: an empty field shows the default as
// its placeholder, so there is no unset state to enter first.
// The transport marker the supervisor sends in place of a redacted value.
const redactedMarker = '[hidden]'

export const samplingRows = ['--temperature', '--top-p', '--top-k', '--seed']

// One registry row prepared for display and editing: the row as the page presents
// it, its saved/running state, the editable form of the row, its current value,
// and the reason it cannot be edited. Server and Settings both read rows through
// this, so a setting is the same setting on both pages.
export interface RegistryEntry {
  row: Row
  state: View['switch_overrides'][number] | undefined
  // The setting is a model choice: it is made on Models, not here.
  modelRoute: boolean
  // The CLI does not let the GUI write this setting.
  blocked: boolean
  // The reason the row cannot be shown as a control.
  error: string
  converted?: RegistryRow
  value?: RegistryValue
  // The CLI redacts the value (a secret): it is not a value to show or save.
  hidden?: boolean
}

export function prepareRow(row: Row, view: View): RegistryEntry {
  const state = view.switch_overrides?.find((entry) => entry.name === row.name)
  const modelRoute =
    state?.write_via === 'preference' &&
    (state.preference_field === 'model_path' ||
      state.preference_field === 'manifest_path')
  const blocked =
    !state ||
    (!!state.write_block &&
      !(
        state.write_via === 'preference' &&
        state.write_block === 'supervisor_owned'
      ))
  const base = { row, state, modelRoute: !!modelRoute, blocked }
  try {
    const converted = registryRow(row)
    converted.label = settingLabel(row)
    const turnOff =
      isTurnOffSwitch(row.name) && converted.allowed.type === 'boolean'
    // The wire value of a turn-off switch is its flag; the user picks the
    // feature's state, so a plain On / Off choice replaces the inverted switch.
    if (turnOff)
      converted.allowed = {
        ...converted.allowed,
        type: 'enum',
        values: [false, true],
      }
    const field = fieldFromRow(converted)
    if (field.booleanValues?.every((value) => value !== undefined)) {
      converted.valueLabels = {
        ...converted.valueLabels,
        [String(field.booleanValues[0])]: 'Off',
        [String(field.booleanValues[1])]: 'On',
      }
    }
    converted.valueLabels = {
      ...converted.valueLabels,
      ...guiValueLabels(row.name, converted.allowed.values),
    }
    // A fixed seed has no named values; -1 is shown as the placeholder "Random".
    if (row.name === '--seed') converted.valueLabels = {}
    converted.purpose = aboutParagraphs(row).join('\n\n')
    if (row.name === '--profile') {
      const profiles = view.profiles.filter((profile) => profile.gui)
      converted.allowed = {
        type: 'enum',
        values: profiles.map((profile) => profile.name),
      }
      converted.valueLabels = Object.fromEntries(
        profiles.map((profile) => [profile.name, profile.label]),
      )
    }
    schemaFromRegistry([converted])
    let value =
      effectiveValue(view, row.name) ??
      (row.name === '--profile' ? registryDefault(row) : undefined)
    // A random seed is -1 on the wire; the field is empty and says "Random".
    if (row.name === '--seed' && String(value) === '-1') value = ''
    if (row.name === '--profile' && typeof value === 'string')
      value = profileAliases[value] ?? value
    if (value === redactedMarker)
      return { ...base, converted, value: undefined, hidden: true, error: '' }
    return {
      ...base,
      converted,
      value:
        value ??
        // A small choice shows its default as the selected value instead of an
        // unset field.
        (samplingRows.includes(row.name)
          ? ''
          : turnOff
            ? false
            : converted.allowed.format === 'effort:bias comma list' ||
                converted.allowed.type === 'enum'
              ? registryDefault(row)
              : undefined),
      error: '',
    }
  } catch (cause) {
    if (!(cause instanceof UnsupportedRegistryError)) throw cause
    return { ...base, error: cause.message }
  }
}
