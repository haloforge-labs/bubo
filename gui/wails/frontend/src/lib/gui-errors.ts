import type { View } from '@/lib/bindings'
import { bytes, number } from '@/lib/format'

export type GUIFailure = Pick<
  NonNullable<View['failure']>,
  'code' | 'remedy'
> & {
  label?: string
  serverStopped?: boolean
  detail?: string
  again?: boolean
  memory?: {
    needed_bytes: number
    available_bytes: number
    sessions: number
    context: number
  }
}
type Recovery =
  | 'none'
  | 'prepare'
  | 'models'
  | 'settings'
  | 'connect'
  | 'about'
  | 'logs'
  | 'retry'
  | 'restart'
  | 'recover'
  | 'server'
  | 'server-setup'
  | 'download'
  | 'vcredist'
export type Wording = {
  message: string
  action: string
  recovery: Recovery
  // A line under the message and its button (the CLI's own remedy text stays
  // in Details).
  below?: string
}

// The CLI names the damaged file in its detail: "<path>: <reason>; delete it
// and run bubo prepare again" (src/cli/prepare/model_prepare.cpp, Different()).
export function damagedFile(
  detail?: string,
): { path: string; name: string } | undefined {
  const match =
    /^((?:[A-Za-z]:[\\/]|\\\\)[^:]*): [^;]*; delete it and run bubo prepare again$/.exec(
      (detail ?? '').trim(),
    )
  if (!match) return undefined
  const path = match[1]!
  const name = path.slice(
    Math.max(path.lastIndexOf('\\'), path.lastIndexOf('/')) + 1,
  )
  return name ? { path, name } : undefined
}

// The one sentence for a downloaded file that failed its pin. Today the user
// deletes the file and downloads it again through "Check or repair files".
function damagedFileSentence(name: string) {
  return `The downloaded file "${name}" did not pass verification. Delete that file, then choose "Check or repair files" to download it again.`
}

// Each code has one visible sentence and at most one recovery control.
const wording: Record<string, Wording> = {
  not_prepared: {
    message: 'The files for this setup are not ready.',
    action: 'Download the engine files',
    recovery: 'prepare',
  },
  runtime_missing: {
    message: "Bubo's engine files are missing.",
    action: 'Download the engine files',
    recovery: 'prepare',
  },
  model_missing: {
    message: 'No model is installed.',
    action: 'Download or choose a file',
    recovery: 'models',
  },
  invalid_arguments: {
    message: 'A server setting needs attention.',
    action: 'Review server setup',
    recovery: 'server-setup',
  },
  port_occupied: {
    message: 'Another app is using the server address.',
    action: 'Review server setup',
    recovery: 'server-setup',
  },
  device_lost: {
    message: 'The GPU stopped responding.',
    action: 'View logs',
    recovery: 'logs',
  },
  startup_timeout: {
    message: 'The server took too long to start.',
    action: 'View logs',
    recovery: 'logs',
  },
  query_timeout: {
    message: 'Bubo took too long to respond.',
    action: 'Try again',
    recovery: 'retry',
  },
  stop_timeout: {
    message: 'The server took too long to stop.',
    action: 'View logs',
    recovery: 'logs',
  },
  protocol_mismatch: {
    message: 'Bubo and its runtime use different versions.',
    action: 'Review Bubo version',
    recovery: 'about',
  },
  insufficient_space: {
    message: 'There is not enough space on the destination drive.',
    action: 'Try again',
    recovery: 'retry',
  },
  plan_changed: {
    message: 'The download plan changed. Review it again before downloading.',
    action: 'Try again',
    recovery: 'retry',
  },
  network: {
    message: 'The download could not connect.',
    action: 'Try again',
    recovery: 'retry',
  },
  filesystem: {
    message: 'Bubo could not read or write a file.',
    action: 'Try again',
    recovery: 'retry',
  },
  configuration: {
    message: 'Update settings need attention.',
    action: 'Review updates',
    recovery: 'about',
  },
  url_refused: {
    message: 'The download address could not be verified.',
    action: 'View logs',
    recovery: 'logs',
  },
  invalid_manifest: {
    message: 'The update information could not be verified.',
    action: 'Try again',
    recovery: 'retry',
  },
  downgrade_refused: {
    message: 'This update is older than the installed version.',
    action: 'Review updates',
    recovery: 'about',
  },
  stale_manifest: {
    message: 'The update information is out of date.',
    action: 'Try again',
    recovery: 'retry',
  },
  intermediate_required: {
    message: 'An earlier update must be installed first.',
    action: 'Review updates',
    recovery: 'about',
  },
  check_required: {
    message: 'Updates have not been checked.',
    action: 'Check for updates',
    recovery: 'about',
  },
  busy: {
    message: 'Another Bubo action is still running.',
    action: '',
    recovery: 'none',
  },
  size_mismatch: {
    message: 'The downloaded file has an unexpected size.',
    action: 'Try again',
    recovery: 'retry',
  },
  sha256_mismatch: {
    message: 'The downloaded file did not pass verification.',
    action: 'Try again',
    recovery: 'retry',
  },
  invalid_resume: {
    message: 'The partial download could not be resumed.',
    action: 'Try again',
    recovery: 'retry',
  },
  canceled: {
    message: 'The action was cancelled.',
    action: 'Try again',
    recovery: 'retry',
  },
  invalid_archive: {
    message: 'The update package could not be opened.',
    action: 'Try again',
    recovery: 'retry',
  },
  process_wait: {
    message: 'Bubo is waiting for its running tasks to finish.',
    action: 'Try again',
    recovery: 'retry',
  },
  swap_failed: {
    message: 'The update files could not be installed.',
    action: 'View logs',
    recovery: 'logs',
  },
  health_failed: {
    message: 'The updated app could not start correctly.',
    action: 'View logs',
    recovery: 'logs',
  },
  rollback_pending: {
    message:
      'Recover the previous version: Bubo stops the server, closes and reopens the previous version.',
    action: 'Recover',
    recovery: 'recover',
  },
  unexpected_failure: {
    message: 'Bubo hit an unexpected problem.',
    action: 'View logs',
    recovery: 'logs',
  },
  cancelled: {
    message: 'The action was cancelled.',
    action: 'Try again',
    recovery: 'retry',
  },
  source_missing: {
    message: 'The required download source is missing.',
    action: 'Download the engine files',
    recovery: 'prepare',
  },
  transport_failed: {
    message: 'Bubo could not download the files.',
    action: 'Try again',
    recovery: 'retry',
  },
  // integrity_failed has several CLI sources. Only the one that names a file
  // with "delete it and run bubo prepare again" is a file the CLI will never
  // replace; guiError() gives that one the delete sentence and no Try again. A
  // supplied file that does not match is the user's own file, and a partial
  // that changed during its check can pass on a second run, so the others keep
  // this row. If the CLI gains a repair, change damagedFileSentence and
  // guiError's integrity_failed branch only.
  integrity_failed: {
    message: 'The downloaded files did not pass verification.',
    action: 'Try again',
    recovery: 'retry',
  },
  storage_failed: {
    message: 'Bubo could not save the files.',
    action: 'Try again',
    recovery: 'retry',
  },
  prepare_locked: {
    message: 'Another preparation is already running.',
    action: 'View logs',
    recovery: 'logs',
  },
  prepare_internal: {
    message: 'Bubo could not finish preparing its files.',
    action: 'View logs',
    recovery: 'logs',
  },
  control_access: {
    message: 'Bubo could not manage this server.',
    action: 'View logs',
    recovery: 'logs',
  },
  diagnostic_failed: {
    message: 'Bubo could not finish diagnostics.',
    action: 'Try again',
    recovery: 'retry',
  },
  binding_unavailable: {
    message: 'Bubo could not connect to the desktop service.',
    action: 'Try again',
    recovery: 'retry',
  },
  download_folder_invalid: {
    message: 'Choose a folder on a local drive.',
    action: '',
    recovery: 'none',
  },
  download_folder_unwritable: {
    message: "Bubo can't write to this folder. Choose another folder.",
    action: '',
    recovery: 'none',
  },
  download_folder_locked: {
    message:
      "The download folder can't change while the server or a download is running.",
    action: '',
    recovery: 'none',
  },
  model_folder_invalid: {
    message: 'The model folder path is invalid.',
    action: 'View logs',
    recovery: 'logs',
  },
  model_folder_not_allowed: {
    message: 'This folder is not in the model library.',
    action: 'View logs',
    recovery: 'logs',
  },
  model_folder_unavailable: {
    message: 'The model folder is unavailable.',
    action: 'View logs',
    recovery: 'logs',
  },
  folder_open_failed: {
    message: 'Bubo could not open the model folder.',
    action: 'Try again',
    recovery: 'retry',
  },
  // Windows could not start bubo.exe (exit 0xC0000135, no output).
  vcredist_missing: {
    message: 'Bubo needs the Microsoft Visual C++ runtime.',
    action: 'Download the runtime',
    recovery: 'vcredist',
    below: 'Install it, then open Bubo again.',
  },
  // Serve exit 19: the sessions at this context do not fit in memory. The
  // numbers, when the CLI sends them, make the sentence exact (guiError).
  memory_insufficient: {
    message:
      "The server didn't start: the sessions at this context need more memory than is available.",
    action: '',
    recovery: 'none',
    below: 'Lower Context per session or Sessions, or close other apps.',
  },
  // Not an error: the supervisor restarted the server once after exit status 75.
  device_lost_restarted: {
    message: 'The GPU stopped responding. Bubo restarted the server.',
    action: '',
    recovery: 'none',
  },
  internal_error: {
    message:
      'Something went wrong inside Bubo. Restart Bubo; if it happens again, save a support bundle on the Logs page.',
    action: 'View logs',
    recovery: 'logs',
  },
  estimate_needs_restart: {
    message: 'Bubo checks memory for this change after the server restarts.',
    action: 'Restart',
    recovery: 'restart',
  },
  estimate_no_model: {
    message: 'Choose an installed model to check memory.',
    action: 'Open Models',
    recovery: 'models',
  },
  // Saved settings that the engine can check only when the server starts.
  estimate_needs_start: {
    message: 'Not checked: you can still start the server.',
    action: '',
    recovery: 'none',
  },
  estimate_unavailable: {
    message: "Bubo couldn't check memory for this setup.",
    action: 'Try again',
    recovery: 'retry',
  },
  tray_unavailable: {
    message:
      "The system tray isn't available, so Bubo can't keep running after the window closes. Keep this window open while the server runs.",
    action: '',
    recovery: 'none',
  },
  update_installing: {
    message: 'Bubo is installing an update. Wait for it to finish.',
    action: '',
    recovery: 'none',
  },
  closing: { message: 'Bubo is closing.', action: '', recovery: 'none' },
  initializing: {
    message: 'Bubo is still starting up.',
    action: 'Try again',
    recovery: 'retry',
  },
  use_connect_for_lan: {
    message: 'Change local-network access on the Connect page.',
    action: 'Open Connect',
    recovery: 'connect',
  },
  setting_removed: {
    message:
      'A setting saved by an older version of Bubo no longer exists. It was removed from your saved settings.',
    action: '',
    recovery: 'none',
  },
  setting_read_only: {
    message: "This setting can't be changed from the app.",
    action: '',
    recovery: 'none',
  },
  login_start_failed: {
    message: 'Bubo couldn\'t update the "start at login" setting.',
    action: 'Open Settings',
    recovery: 'settings',
  },
  update_path_redirected: {
    message:
      "Bubo can't update itself in this folder, because Windows redirects it (for example OneDrive or a mapped drive). Move Bubo to a normal folder, then update.",
    action: '',
    recovery: 'none',
  },
  update_folder_read_only: {
    message:
      "Bubo can't update itself, because its folder is read-only. Move Bubo to a folder you can write to, then update.",
    action: '',
    recovery: 'none',
  },
  update_version_invalid: {
    message: "The update information has a version number Bubo can't read.",
    action: 'Try again',
    recovery: 'retry',
  },
  update_interrupted: {
    message:
      'The update stopped before it changed anything. Your installed version is unchanged.',
    action: '',
    recovery: 'none',
  },
  download_stalled: {
    message:
      "The download stopped making progress. Check your internet connection. If you use a proxy, Bubo can't use it yet.",
    action: 'Try again',
    recovery: 'retry',
  },
  lan_unavailable: {
    message: "Local network access isn't available in this copy of Bubo.",
    action: '',
    recovery: 'none',
  },
  setting_rejected: {
    message: "The engine didn't accept one of your settings.",
    action: 'Review server setup',
    recovery: 'server-setup',
  },
  update_external_server: {
    message:
      'A server started outside Bubo is running. Stop it where it was started, then install the update.',
    action: 'Try again',
    recovery: 'retry',
  },
  updater_starting: {
    message: 'Bubo is still getting ready to check for updates.',
    action: 'Try again',
    recovery: 'retry',
  },
  updater_too_old: {
    message:
      "This Bubo can't read the newest update information. Download the latest Bubo from the Bubo website.",
    action: '',
    recovery: 'none',
  },
  key_not_created: {
    message:
      'There is no API key yet. Turn on local network access on the Connect page to create one.',
    action: 'Open Connect',
    recovery: 'connect',
  },
  key_store_failed: {
    message: "Windows couldn't read or save the API key.",
    action: 'Try again',
    recovery: 'retry',
  },
  clipboard_failed: {
    message:
      "Windows didn't allow copying. Select the text and copy it yourself.",
    action: '',
    recovery: 'none',
  },
  server_exited: {
    message: 'The server stopped unexpectedly.',
    action: 'View logs',
    recovery: 'logs',
  },
  cli_unavailable: {
    message: "Bubo couldn't run its command-line tool.",
    action: 'Try again',
    recovery: 'retry',
  },
  timeout: {
    message: 'Bubo took too long to respond.',
    action: 'Try again',
    recovery: 'retry',
  },
  external_read_only: {
    message:
      'This server was started outside Bubo. Stop it where it was started.',
    action: '',
    recovery: 'none',
  },
  memory_does_not_fit: {
    message: "This setup doesn't fit in memory.",
    action: 'Open Server setup',
    recovery: 'server-setup',
  },
  memory_tight: {
    message: 'Memory is tight. Review the saved setup on the Server page.',
    action: 'Open Server setup',
    recovery: 'server-setup',
  },
  readiness_unavailable: {
    message: 'Bubo is still checking its files.',
    action: 'Try again',
    recovery: 'retry',
  },
  fit_unchecked: {
    message: "Bubo can't check memory while its own server is using it.",
    action: '',
    recovery: 'none',
  },
  model_file_invalid: {
    message: 'Choose a model file that ends in .gguf.',
    action: '',
    recovery: 'none',
  },
  memory_does_not_fit_any: {
    message:
      "This model doesn't fit in memory at any context size. Choose a smaller model.",
    action: 'Open Models',
    recovery: 'models',
  },
  model_unusable: {
    message: "Bubo can't use this model file.",
    action: '',
    recovery: 'none',
  },
  model_folder_warning: {
    message: 'Bubo skipped some files in this folder.',
    action: '',
    recovery: 'none',
  },
  npu_paused: {
    message: 'The NPU is paused, so the GPU handles prompts for now.',
    action: '',
    recovery: 'none',
  },
  cli_missing: {
    message: 'Bubo needs to be reinstalled.',
    action: 'Open download page',
    recovery: 'download',
  },
  server_stalled: {
    message: 'The server stopped responding. Restart it to continue.',
    action: '',
    recovery: 'none',
  },
  page_error: {
    message: 'This page hit a problem. Go to the Server page, or restart Bubo.',
    action: 'Open Server',
    recovery: 'server',
  },
  startup_slow: {
    message: 'Bubo is taking longer than usual to start. It keeps trying.',
    action: 'View logs',
    recovery: 'logs',
  },
  not_applicable: {
    message: 'This action is not available right now.',
    action: '',
    recovery: 'none',
  },
  support_bundle_saved: {
    message: 'Support bundle saved locally.',
    action: '',
    recovery: 'none',
  },
  settings_saved: {
    message: 'Restart to apply these settings.',
    action: '',
    recovery: 'none',
  },
}

// A failed start of the updater, or a failed check: no update was started. Two
// codes have a download sentence in the table; a check names the update server.
const checkReasons: Record<string, string> = {
  network: "it couldn't reach the update server.",
  url_refused: 'the update address could not be verified.',
}

export function checkFailureSentence(error: GUIFailure): string {
  if (error.code === 'update_information_missing')
    return 'No update information was found. Bubo checks again later.'
  if (error.code === 'update_server_error')
    return 'The update server could not provide update information. Bubo checks again later.'
  const own = checkReasons[error.code]
  if (own) return `Bubo can't check for updates: ${own}`
  const reason = guiError(error).message
  const keep = /^(Bubo|Windows)\b/.test(reason)
  const text = keep ? reason : reason.charAt(0).toLowerCase() + reason.slice(1)
  return `Bubo can't check for updates: ${text}`
}

// One plain sentence for each server warning code the app knows. A code it
// does not know shows no line (the raw text is under Details of a known one).
const warningSentences: Record<string, string> = {
  queue_pressure: 'Other GPU work is slowing the server.',
}
export function warningSentence(code: string): string | undefined {
  return warningSentences[code]
}

export function guiError(error: GUIFailure): Wording {
  if (
    error.code === 'source_missing' &&
    error.detail?.includes('hosting is unpublished')
  )
    return { message: error.detail, action: '', recovery: 'none' }
  if (error.code === 'rollback_pending')
    return {
      ...wording.rollback_pending!,
      // Bubo stops the server only when one is running.
      message: `Recover the previous version: Bubo ${error.serverStopped ? '' : 'stops the server, '}closes and reopens ${error.label ? `as version ${error.label}` : 'the previous version'}.`,
    }
  if (error.code === 'memory_insufficient' && error.memory) {
    const { sessions, context, needed_bytes, available_bytes } = error.memory
    return {
      ...wording.memory_insufficient!,
      message: `The server didn't start: needs ${bytes(needed_bytes, { unit: 'GiB', digits: 0 })} for ${sessions} ${sessions === 1 ? 'session' : 'sessions'} at ${number(context, 0)} tokens; ${bytes(available_bytes, { unit: 'GiB', digits: 0 })} is available.`,
    }
  }
  if (error.code === 'integrity_failed') {
    const file = damagedFile(error.detail)
    if (file)
      return {
        ...wording.integrity_failed!,
        message: damagedFileSentence(file.name),
        action: '',
        recovery: 'none',
      }
  }
  if (error.code === 'device_lost' && error.again)
    return {
      ...wording.device_lost!,
      message: 'The GPU stopped responding again.',
    }
  if (error.code === 'setting_read_only' && error.label)
    return {
      ...wording.setting_read_only!,
      message: `The setting "${error.label}" can't be changed from the app.`,
    }
  return wording[error.code] ?? wording.internal_error!
}

export function asGUIFailure(error: unknown): GUIFailure | undefined {
  if (!error) return undefined
  if (typeof error === 'object' && 'code' in error && 'remedy' in error)
    return {
      code: String(error.code),
      remedy: String(error.remedy),
      label:
        'label' in error && typeof error.label === 'string'
          ? error.label
          : undefined,
      serverStopped:
        'serverStopped' in error && typeof error.serverStopped === 'boolean'
          ? error.serverStopped
          : undefined,
      detail:
        'detail' in error && typeof error.detail === 'string' && error.detail
          ? error.detail
          : undefined,
      again: 'again' in error && error.again === true ? true : undefined,
      memory:
        'memory' in error && error.memory && typeof error.memory === 'object'
          ? (error.memory as GUIFailure['memory'])
          : undefined,
    }
  return {
    code: 'internal_error',
    remedy: error instanceof Error ? error.message : String(error),
  }
}
