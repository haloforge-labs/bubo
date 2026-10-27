import { actions } from '@/lib/bindings'
import { requireActionSuccess } from '@/lib/bindings'

type PlanActions = Pick<
  typeof actions,
  'Download' | 'CheckModel' | 'ConfirmDownload' | 'CancelDownload'
> & { RepairPackage?: typeof actions.RepairPackage }

// Both paths return the same typed plan/error envelope.
export async function openModelPlan(
  api: PlanActions,
  id: string,
  kind: 'download' | 'repair',
  // The --output the CLI named on the package's row (repair_output): a repair
  // of a package acts where that package is. None: the CLI default folder.
  output?: string,
) {
  if (output && kind === 'repair' && !api.RepairPackage)
    throw {
      code: 'internal_error',
      remedy: 'The package repair action is missing.',
    }
  const result = await (kind === 'repair'
    ? output
      ? api.RepairPackage!(id, output)
      : api.CheckModel(id, true)
    : api.Download(id))
  requireActionSuccess(result)
  if (!result.plan)
    throw {
      code: 'protocol_mismatch',
      remedy: 'The CLI did not return a download plan.',
    }
  return result.plan
}

export async function confirmModelPlan(api: PlanActions, id: string) {
  requireActionSuccess(await api.ConfirmDownload(id))
}

export async function cancelModelPlan(api: PlanActions) {
  requireActionSuccess(await api.CancelDownload())
}
