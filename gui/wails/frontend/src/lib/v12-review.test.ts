import { expect, test } from 'vitest'
import { guiError } from './gui-errors'

test.each(['invalid_arguments', 'port_occupied', 'setting_rejected'])(
  '%s recovery opens Server setup',
  (code) => {
    expect(guiError({ code, remedy: '' }).recovery).toBe('server-setup')
    expect(guiError({ code, remedy: '' }).action).toBe('Review server setup')
  },
)
