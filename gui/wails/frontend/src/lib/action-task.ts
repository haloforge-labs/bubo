import { useMutation } from '@tanstack/react-query'
import { requireActionSuccess } from '@/lib/bindings'
import { asGUIFailure } from '@/lib/gui-errors'

// All desktop actions and composed flows share typed failure handling and never
// retry a mutation automatically. run supports flows that need a success flag.
export function useActionTask() {
  const mutation = useMutation<unknown, unknown, () => Promise<unknown>>({
    mutationFn: async (task) => requireActionSuccess(await task()),
    retry: false,
  })
  async function run(task: () => Promise<unknown>) {
    try {
      await mutation.mutateAsync(task)
      return true
    } catch {
      return false
    }
  }
  return {
    ...mutation,
    run,
    pending: mutation.isPending,
    error: mutation.isError ? asGUIFailure(mutation.error) : undefined,
    completed: mutation.isSuccess,
  }
}
