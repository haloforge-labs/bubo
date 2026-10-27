import observations from '../../tests/recorded-observations.json'
import {
  coreView,
  folderView,
  packagesView,
  recordedPlan,
} from './core-page-test-support'
import originalPlan from '../../tests/original-plan.json'

export function recordedFolderView(chosen?: string) {
  const view = folderView(chosen)
  const plan = chosen ? recordedPlan() : originalPlan.plan
  const model = plan.paths.model!
  const folder = model.slice(0, model.lastIndexOf('\\models\\'))
  view.download_folder = {
    ...view.download_folder,
    path: `${folder}\\models`,
    drive: folder.slice(0, 2),
    free_bytes: plan.free_bytes,
  }
  return view
}

export function recordedPackagesView() {
  const result = packagesView()
  const real = coreView('stopped').models!.prepared_models
  result.rows[0]!.manifest_path = real[0]!.manifest_path
  result.rows[0]!.model_path = real[0]!.model_path
  result.rows[0]!.repair_output = real[0]!.repair_output
  const second = recordedPlan().paths.model!
  result.rows[1]!.manifest_path = `${second}\\bubo-package.json`
  result.rows[1]!.repair_output = second.slice(
    0,
    second.lastIndexOf('\\models\\'),
  )
  return result
}

// Preserve the measured values and intervals; arrange the historical series at
// the current browser clock so the normal five-minute collector can display it.
export function recordedGPU(at = Date.now()) {
  const last = observations.gpu.at(-1)!
  const end = Date.parse(last.gpu.sample_time)
  return {
    current: structuredClone(last.gpu),
    samples: observations.gpu.map(({ gpu }) => ({
      at: at + Date.parse(gpu.sample_time) - end,
      gpu: structuredClone(gpu),
    })),
  }
}

export function recordedTransfer(at = Date.now()) {
  const { transfer } = observations
  const plan = coreView('preparing').preparation[0]!
  const total = plan.files.reduce((sum, file) => sum + file.size, 0)
  const position = plan.files.findIndex(
    (file) => file.file_id === transfer.file_id,
  )
  const prior = plan.files
    .slice(0, position)
    .reduce((sum, file) => sum + file.size, 0)
  return {
    file: {
      file_id: transfer.file_id,
      bytes_done: transfer.bytes_done,
      bytes_total: transfer.bytes_total,
    },
    overall: { bytes_done: prior + transfer.bytes_done, bytes_total: total },
    samples: [
      { bytes: prior, at: at - transfer.elapsed_ms },
      { bytes: prior + transfer.bytes_done, at },
    ],
  }
}
