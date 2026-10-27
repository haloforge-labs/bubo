import { ErrorNote } from '@/components/error-note'
import { requireActionSuccess } from '@/lib/bindings'
import { Controller, useForm } from 'react-hook-form'
import {
  useEffect,
  useId,
  useImperativeHandle,
  useState,
  type ReactNode,
  type Ref,
} from 'react'
import { Link } from '@tanstack/react-router'
import { useSetupRegistry } from './setup-registry'
import { MemoryFit } from './memory-fit'
import { zodResolver } from '@hookform/resolvers/zod'
import { useQuery } from '@tanstack/react-query'
import { actions, type View } from '@/lib/bindings'
import { ActionError, useActionTask } from '@/components/action-feedback'
import {
  estimateSubject,
  preparedRows,
  preparedStart,
  modelChoices,
  modelPreferencePatch,
  activeModelChoice,
} from '@/components/model-facts'
import { Button } from '@/components/ui/button'
import { NumberInput } from '@/components/number-input'
import { Label } from '@/components/ui/label'
import { PathText } from '@/components/path-text'
import { activeModelName, recommendedModel } from '@/lib/format'
import { guiError } from '@/lib/gui-errors'
import {
  numberConstraint,
  contextPreset,
  numericSetting,
  serverFormSchema,
} from '@/features/server/context'

export interface ServerSetupHandle {
  save: () => Promise<unknown>
}

// A memory line needs a verdict. "Checking memory…" and "In use" are not one.
function hasVerdict(fit?: View['fit']) {
  return !!fit && fit.class !== 'unavailable' && fit.class !== 'in_use'
}

function fitIsUnchecked(view: View) {
  return [view.actions.start, view.actions.restart].some(
    (action) => action.reason_code === 'fit_unchecked',
  )
}

export function ServerSetup({
  view,
  onDirtyChange,
  actionRef,
  suggestionSignal = 0,
  addresses,
}: {
  view: View
  onDirtyChange: (dirty: boolean) => void
  actionRef?: Ref<ServerSetupHandle>
  suggestionSignal?: number
  addresses?: ReactNode
}) {
  try {
    serverFormSchema(view)
  } catch {
    return (
      <div data-code="not_reported" className="space-y-3">
        <p>Server setup settings have not been reported yet.</p>
        {!view.failure?.switch &&
          (fitIsUnchecked(view) || !view.serving) &&
          view.failure?.code !== 'memory_insufficient' &&
          hasVerdict(view.fit) && (
            <p aria-label="Memory fit" className="text-sm">
              {view.fit.sentence}
            </p>
          )}
        {addresses}
      </div>
    )
  }
  // Remount after saved overrides change; polling must not overwrite an editing draft.
  return (
    <SetupForm
      key={JSON.stringify([
        view.saved.context,
        view.saved.sessions,
        view.saved.manifest_path,
        view.saved.model_path,
        view.saved.profile,
      ])}
      view={view}
      onDirtyChange={onDirtyChange}
      actionRef={actionRef}
      suggestionSignal={suggestionSignal}
      addresses={addresses}
    />
  )
}

function SetupForm({
  view,
  onDirtyChange,
  actionRef,
  suggestionSignal,
  addresses,
}: {
  view: View
  onDirtyChange: (dirty: boolean) => void
  actionRef?: Ref<ServerSetupHandle>
  suggestionSignal: number
  addresses?: ReactNode
}) {
  const choices = modelChoices(view)
  const context = numericSetting(view, '--context')
  const sessions = numericSetting(view, '--sessions')
  const savedMemory = view.saved_estimate?.memory_estimate
  const initialContext =
    view.saved.context ??
    savedMemory?.context_tokens ??
    context.value ??
    context.default
  const initialSessions =
    view.saved.sessions ??
    savedMemory?.sessions ??
    sessions.value ??
    sessions.default
  const initialModel =
    view.saved.model_path ||
    view.saved.manifest_path ||
    activeModelChoice(view)?.path ||
    view.current_model?.model_path ||
    view.current_model?.manifest_path ||
    view.prepared_default ||
    ''
  const form = useForm({
    resolver: zodResolver(serverFormSchema(view, null)),
    defaultValues: {
      context: initialContext,
      sessions: initialSessions,
      model: initialModel,
    },
  })
  const registry = useSetupRegistry(view)
  const [inferenceExpanded, setInferenceExpanded] = useState(
    view.saved.inference_expanded ?? false,
  )
  const inferenceId = useId()
  const disclosure = useActionTask()
  function toggleInference() {
    const next = !inferenceExpanded
    setInferenceExpanded(next)
    void disclosure
      .run(() =>
        actions.PatchPreferences({
          set: { inference_expanded: next },
          clear: [],
        }),
      )
      .then((saved) => {
        if (!saved) setInferenceExpanded(!next)
      })
  }
  const draft = form.watch()
  useEffect(() => {
    if (suggestionSignal && view.fit.suggested_context != null)
      form.setValue('context', view.fit.suggested_context, {
        shouldDirty: true,
        shouldValidate: true,
      })
  }, [suggestionSignal, view.fit.suggested_context, form])
  const dirty = form.formState.isDirty
  useEffect(() => {
    onDirtyChange(dirty || registry.dirty)
    return () => onDirtyChange(false)
  }, [dirty, registry.dirty, onDirtyChange])
  // A first saved estimate arrives asynchronously; hydrate a pristine form only.
  useEffect(() => {
    if (
      !dirty &&
      (form.getValues('context') == null ||
        form.getValues('sessions') == null ||
        form.getValues('model') !== initialModel)
    )
      form.reset({
        context: initialContext,
        sessions: initialSessions,
        model: initialModel,
      })
  }, [initialContext, initialSessions, initialModel, dirty, form])
  const { dirtyFields } = form.formState
  const needsEstimate = dirty
  const command = useActionTask()
  const selectedModel = choices.find((choice) => choice.path === draft.model)
  const selectedPrepared = preparedRows(view).find(
    (model) => preparedStart(model)?.path === draft.model,
  )
  const activeIdentity = view.current_model
  const activePath = activeIdentity?.model_path || activeIdentity?.manifest_path
  const selectedStart = selectedPrepared && preparedStart(selectedPrepared)
  const subject =
    selectedModel?.subject ||
    (selectedStart && estimateSubject(selectedStart)) ||
    (draft.model === activePath
      ? estimateSubject(
          activeIdentity?.model_path
            ? { kind: 'file', path: activeIdentity.model_path }
            : { kind: 'manifest', path: activeIdentity?.manifest_path ?? '' },
        )
      : '')
  const estimate = useQuery({
    queryKey: ['server-estimate', subject, draft.context, draft.sessions],
    queryFn: async () => {
      if (!subject)
        throw {
          code: 'estimate_no_model',
          remedy: 'Select an installed model to check memory fit.',
        }
      const result = await actions.Estimate(
        subject,
        Number(draft.context),
        Number(draft.sessions),
      )
      requireActionSuccess(result)
      if (!result.estimate)
        throw {
          code: 'estimate_unavailable',
          remedy: 'A memory estimate was not reported.',
        }
      return { ...result.estimate, fit: result.fit }
    },
    enabled:
      needsEstimate &&
      !registry.setupChanged &&
      serverFormSchema(view, null).safeParse(draft).success,
    staleTime: Infinity,
    retry: false,
  })
  const currentEstimate = needsEstimate
    ? estimate.data?.memory_estimate
    : view.saved_estimate
      ? view.saved_estimate.memory_estimate
      : view.memory_estimate
  const table = registry.setupChanged
    ? undefined
    : (currentEstimate?.recommended_context ??
      (draft.model === initialModel
        ? (savedMemory ?? view.memory_estimate)?.recommended_context
        : undefined))
  const largest = contextPreset(table, Number(draft.sessions))
  const fit = needsEstimate ? estimate.data?.fit : view.fit
  const contextAllowed = numberConstraint(context.row)
  const sessionsAllowed = numberConstraint(sessions.row)
  const running = view.serving
  const unchecked = fitIsUnchecked(view)
  // The Start area already says a fit problem, with its one fix.
  // The engine cannot judge this saved change until the server starts.
  // While files download or are missing there is nothing to check.
  const downloading =
    view.phase === 'downloading_model' ||
    view.phase === 'preparing_engine' ||
    view.phase === 'needs_engine' ||
    view.phase === 'needs_model'
  const checkedAtStart =
    !running &&
    !dirty &&
    (view.saved_estimate?.error?.code === 'estimate_needs_restart' ||
      view.saved_estimate?.error?.code === 'estimate_no_verdict')

  async function saveSetup() {
    if (!(await form.trigger()))
      throw {
        code: 'setting_rejected',
        remedy: 'Check the highlighted settings.',
      }
    if (!dirty && !registry.dirty) return {}
    const values = form.getValues()
    const registryPatch = registry.patch()
    if (!registryPatch)
      throw {
        code: 'setting_rejected',
        remedy: 'Check the highlighted settings.',
      }
    const modelTable =
      estimate.data?.memory_estimate?.recommended_context ??
      (values.model === initialModel
        ? (view.saved_estimate?.memory_estimate ?? view.memory_estimate)
            ?.recommended_context
        : null)
    const checked = serverFormSchema(
      view,
      registry.setupChanged ? null : (modelTable ?? null),
    ).safeParse(values)
    if (!checked.success) {
      for (const issue of checked.error.issues) {
        if (issue.path[0] === 'context' || issue.path[0] === 'sessions')
          form.setError(issue.path[0], { message: issue.message })
      }
      throw {
        code: 'setting_rejected',
        remedy: 'Check the highlighted settings.',
      }
    }

    const choice = choices.find((choice) => choice.path === values.model)
    if (values.model !== initialModel && !choice)
      throw {
        code: 'estimate_no_model',
        remedy: 'Choose an installed model.',
      }
    const patch =
      choice && dirtyFields.model
        ? modelPreferencePatch(choice.kind, choice.path)
        : { set: {}, clear: [] }
    const result = await actions.PatchPreferences({
      ...patch,
      clear: [...patch.clear, ...registryPatch.clear],
      ...(registryPatch.switch_values.length
        ? { switch_values: registryPatch.switch_values }
        : {}),
      set: {
        ...patch.set,
        ...(dirtyFields.context ? { context: Number(values.context) } : {}),
        ...(dirtyFields.sessions ? { sessions: Number(values.sessions) } : {}),
      },
    })
    requireActionSuccess(result)
    form.reset(values)
    registry.reset()
    return result
  }
  useImperativeHandle(actionRef, () => ({ save: saveSetup }))
  return (
    <div id="server-setup" className="space-y-4">
      <form
        noValidate
        className="space-y-4"
        onSubmit={(event) => {
          event.preventDefault()
          command.mutate(saveSetup)
        }}
      >
        <div className="grid items-start gap-4 sm:grid-cols-[2fr_1fr_1fr]">
          <div className="space-y-2">
            <Label htmlFor="server-model">
              {running ? 'Model for next start' : 'Model'}
            </Label>
            <select
              id="server-model"
              className="h-9 w-full min-w-0 rounded-md border bg-background px-2 text-sm"
              {...form.register('model')}
            >
              {!choices.some((choice) => choice.path === initialModel) && (
                <option value={initialModel}>
                  {initialModel
                    ? initialModel === view.prepared_default
                      ? preparedRows(view).find(
                          (model) =>
                            preparedStart(model)?.path === initialModel,
                        )?.display_name ||
                        activeModelName(view) ||
                        recommendedModel(view)?.display_name ||
                        'Default model'
                      : activeModelName(view) || 'Saved model'
                    : 'Choose an installed model'}
                </option>
              )}
              {choices.map((choice) => (
                <option key={choice.path} value={choice.path}>
                  {choice.name}
                </option>
              ))}
            </select>
            <Link
              className="block text-sm underline"
              to="/models"
              hash="import"
            >
              Not listed? Add a model you already have.
            </Link>
            {draft.model && (
              <details className="text-sm text-muted-foreground">
                <summary>Model file</summary>
                <PathText path={draft.model} />
              </details>
            )}
          </div>
          <div className="space-y-2">
            <Label htmlFor="server-context">Context per session</Label>
            <Controller
              control={form.control}
              name="context"
              render={({ field }) => (
                <NumberInput
                  id="server-context"
                  min={table?.minimum_context_tokens ?? contextAllowed.minimum}
                  max={table?.context_max_tokens ?? contextAllowed.maximum}
                  step={contextAllowed.type === 'integer' ? 1 : undefined}
                  {...field}
                  value={
                    typeof field.value === 'number' ? field.value : undefined
                  }
                  onValueChange={field.onChange}
                  aria-invalid={!!form.formState.errors.context}
                />
              )}
            />
            <p className="text-sm text-muted-foreground">
              Largest that fits now:{' '}
              {largest?.fits && largest.recommended !== false ? (
                <>
                  {largest.context_per_session.toLocaleString()} tokens for{' '}
                  {largest.sessions}{' '}
                  {largest.sessions === 1 ? 'session' : 'sessions'}{' '}
                  <Button
                    type="button"
                    variant="link"
                    aria-label="Use largest context"
                    onClick={() =>
                      form.setValue('context', largest.context_per_session, {
                        shouldDirty: true,
                        shouldValidate: true,
                      })
                    }
                  >
                    Use
                  </Button>
                </>
              ) : registry.setupChanged ? (
                'save this Mode to check.'
              ) : largest ? (
                `no recommended context for ${largest.sessions} ${largest.sessions === 1 ? 'session' : 'sessions'}.`
              ) : (
                'not reported for this setup.'
              )}
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="server-sessions">Sessions</Label>
            <Controller
              control={form.control}
              name="sessions"
              render={({ field }) => (
                <NumberInput
                  id="server-sessions"
                  min={sessionsAllowed.minimum}
                  max={table?.max_sessions ?? sessionsAllowed.maximum}
                  step={sessionsAllowed.type === 'integer' ? 1 : undefined}
                  {...field}
                  value={
                    typeof field.value === 'number' ? field.value : undefined
                  }
                  onValueChange={field.onChange}
                  aria-invalid={!!form.formState.errors.sessions}
                />
              )}
            />
          </div>
        </div>
        {!registry.setupChanged &&
          !view.failure?.switch &&
          (view.failure?.code !== 'memory_insufficient' || needsEstimate) &&
          !checkedAtStart &&
          !downloading &&
          (needsEstimate && estimate.isFetching ? (
            <p aria-label="Memory fit" className="text-sm">
              Checking memory…
            </p>
          ) : (
            <MemoryFit
              view={view}
              fit={fit}
              estimate={currentEstimate}
              modelId={selectedModel?.id ?? selectedPrepared?.id}
              dirty={dirty || registry.setupChanged}
              onContext={(value) =>
                form.setValue('context', value, {
                  shouldDirty: true,
                  shouldValidate: true,
                })
              }
              onSessions={(value) =>
                form.setValue('sessions', value, {
                  shouldDirty: true,
                  shouldValidate: true,
                })
              }
            />
          ))}
        <section aria-label="Network" className="space-y-3 border-t pt-4">
          <h2 className="text-lg">Network</h2>
          {registry.render('Network')}
          <p className="text-sm">
            Client access, local network and API key:{' '}
            <Link to="/connect" className="underline">
              Connect
            </Link>
          </p>
        </section>
        {addresses}
        <section aria-label="Inference" className="space-y-3 border-t pt-4">
          <h2>
            <Button
              type="button"
              variant="ghost"
              className="h-auto w-full justify-between px-0 text-lg"
              aria-expanded={inferenceExpanded}
              aria-controls={inferenceId}
              disabled={disclosure.isPending}
              onClick={toggleInference}
            >
              Inference
              <span aria-hidden="true">{inferenceExpanded ? '−' : '+'}</span>
            </Button>
          </h2>
          {!inferenceExpanded && (
            <p
              className="truncate text-sm text-muted-foreground"
              title={registry.summary}
            >
              {registry.summary}
            </p>
          )}
          <div id={inferenceId} hidden={!inferenceExpanded}>
            {registry.render('Inference')}
          </div>
        </section>
        {Object.entries(form.formState.errors).map(([name, error]) => (
          <ErrorNote key={name} className="text-sm text-destructive">
            <span>{String(error.message)}</span>
          </ErrorNote>
        ))}
        {checkedAtStart && (
          <p className="text-sm text-muted-foreground">
            {guiError({ code: 'estimate_needs_start', remedy: '' }).message}
          </p>
        )}
        {!running &&
          needsEstimate &&
          currentEstimate?.fits === false &&
          fit?.class !== 'memory_does_not_fit_any' && (
            <Button
              type="button"
              variant="outline"
              disabled={command.isPending || view.models?.refreshing}
              onClick={() =>
                command.mutate(
                  needsEstimate
                    ? () => estimate.refetch({ throwOnError: true })
                    : actions.RefreshModels,
                )
              }
            >
              Check again
            </Button>
          )}
        {running && (dirty || !!view.restart_needed?.length) && (
          <div className="space-y-1 text-sm">
            <p>Changes will be used after restart.</p>
            {view.restart_needed?.map((change) => (
              <p key={change.key}>
                {change.label || 'Setting'}: {change.running || 'not reported'}{' '}
                → {change.next}
              </p>
            ))}
          </div>
        )}
        <div className="sticky bottom-0 z-10 flex items-center gap-3 border-t bg-background py-3">
          <Button
            type="submit"
            disabled={
              (!dirty && !registry.dirty) ||
              command.isPending ||
              (!!subject &&
                needsEstimate &&
                !registry.setupChanged &&
                !estimate.data)
            }
          >
            Save server setup
          </Button>
        </div>
        {registry.error && <ErrorNote>{registry.error}</ErrorNote>}
        <Button
          type="button"
          variant="outline"
          disabled={command.isPending}
          onClick={() =>
            command.mutate(() =>
              actions.PatchPreferences({
                set: {},
                clear: ['context', 'sessions', 'model_path', 'manifest_path'],
              }),
            )
          }
        >
          Reset to default
        </Button>
      </form>
      <ActionError
        error={
          command.error ||
          disclosure.error ||
          (!view.failure?.switch &&
            (estimate.error ||
              (!dirty &&
                !unchecked &&
                !checkedAtStart &&
                view.saved_estimate?.error)))
        }
      />
      {!view.failure?.switch &&
        !dirty &&
        unchecked &&
        view.saved_estimate?.error && (
          <details className="text-xs text-muted-foreground">
            <summary>Memory check details</summary>
            <p className="font-mono">
              {view.saved_estimate.error.code}:{' '}
              {view.saved_estimate.error.remedy}
            </p>
          </details>
        )}
    </div>
  )
}
