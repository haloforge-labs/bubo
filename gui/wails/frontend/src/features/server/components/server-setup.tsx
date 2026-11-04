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
import { PathText } from '@/components/path-text'
import { SettingGroup, SettingRow } from '@/components/setting-row'
import { SideFiles } from './side-files'
import { CapabilitySource } from '@/components/registry-rows'
import { activeModelName, recommendedModel } from '@/lib/format'
import { pendingChangeWords } from '@/lib/presentation'
import { useRevealedSetting, useRevealSetting } from '@/lib/reveal'
import { serverRowSection } from '@/lib/server-settings'
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

// A saved model the scan has not listed yet is named by its own file (or its
// package folder), never by the model that happens to be active.
function savedModelLabel(path: string) {
  const parts = path.split(/[\\/]/).filter(Boolean)
  if (/^bubo-package\.json$/i.test(parts.at(-1) ?? '')) parts.pop()
  return parts.at(-1) || 'Saved model'
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
  const [showModelFile, setShowModelFile] = useState(false)
  const [inferenceExpanded, setInferenceExpanded] = useState(
    view.saved.inference_expanded ?? false,
  )
  const inferenceId = useId()
  // A recovery link that names an Inference setting shows it: the section opens
  // (for this visit only; the saved choice is not changed) and the row scrolls in.
  const revealed = useRevealedSetting()
  const revealedInInference =
    !!revealed &&
    view.switches.some(
      (row) => row.name === revealed && serverRowSection(row) === 'Inference',
    )
  const inferenceOpen = inferenceExpanded || revealedInInference
  useRevealSetting(revealed, inferenceOpen)
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
  // Every row that depends on the model describes the one in the picker.
  const plainFile =
    (selectedModel?.kind ??
      (draft.model === view.saved.model_path ? 'file' : '')) === 'file'
  // A helper file is checked against the saved model, so a picked model is saved
  // before the file dialog opens.
  async function saveModelChoice() {
    if (!selectedModel) return
    requireActionSuccess(
      await actions.PatchPreferences(
        modelPreferencePatch(selectedModel.kind, selectedModel.path),
      ),
    )
    form.resetField('model', { defaultValue: draft.model })
  }
  // The settings describe the model in the picker, saved or not.
  const pickedCapabilities =
    draft.model === initialModel
      ? undefined
      : (estimate.data?.capabilities ??
        estimate.data?.memory_estimate?.capabilities ??
        null)
  return (
    <CapabilitySource.Provider value={pickedCapabilities}>
      <div id="server-setup" className="space-y-4">
        <form
          noValidate
          className="@container space-y-3"
          onSubmit={(event) => {
            event.preventDefault()
            command.mutate(saveSetup)
          }}
        >
          <SettingGroup className="grid-cols-1 @min-[40rem]:grid-cols-2">
            <SettingRow
              name="model"
              label={running ? 'Next model' : 'Model'}
              htmlFor="server-model"
              wide
              about={
                <p>
                  The model the server starts with. A model you already have is
                  added on Models.
                </p>
              }
              control={
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
                          : savedModelLabel(initialModel)
                        : 'Choose an installed model'}
                    </option>
                  )}
                  {choices.map((choice) => (
                    <option key={choice.path} value={choice.path}>
                      {choice.name}
                    </option>
                  ))}
                </select>
              }
            />
            {registry.renderRow('--served-model-name')}
            <div className="col-span-full space-y-1">
              <div className="flex flex-wrap items-center gap-x-2 text-sm">
                <Link className="underline" to="/models" hash="import">
                  Add a model you have
                </Link>
                {draft.model && (
                  <>
                    <span aria-hidden="true">·</span>
                    <button
                      type="button"
                      className="text-muted-foreground"
                      aria-expanded={showModelFile}
                      onClick={() => setShowModelFile((open) => !open)}
                    >
                      Model file{' '}
                      <span aria-hidden="true">
                        {showModelFile ? '▾' : '▸'}
                      </span>
                    </button>
                  </>
                )}
                {(view.saved.model_path ||
                  view.saved.manifest_path ||
                  view.saved.context != null ||
                  view.saved.sessions != null) && (
                  // Clears exactly these four choices, and shows only while one is set.
                  <>
                    <span aria-hidden="true">·</span>
                    <button
                      type="button"
                      className="underline"
                      disabled={command.isPending}
                      onClick={() =>
                        command.mutate(() =>
                          actions.PatchPreferences({
                            set: {},
                            clear: [
                              'context',
                              'sessions',
                              'model_path',
                              'manifest_path',
                            ],
                          }),
                        )
                      }
                    >
                      Reset model, context and sessions
                    </button>
                  </>
                )}
              </div>
              {draft.model && showModelFile && <PathText path={draft.model} />}
            </div>
          </SettingGroup>
          <SettingGroup>
            <SettingRow
              name="context"
              label="Context per session"
              htmlFor="server-context"
              unit="tokens"
              about={
                <p>
                  How many tokens of conversation each session can hold. The
                  maximum depends on the memory the model, the cache and the
                  number of sessions leave.
                </p>
              }
              hint={
                largest?.fits && largest.recommended !== false ? (
                  <>
                    {Number(draft.context) > largest.context_per_session
                      ? 'Above max'
                      : 'Max'}{' '}
                    {largest.context_per_session.toLocaleString()} ·{' '}
                    <Button
                      type="button"
                      variant="link"
                      className="h-auto p-0 text-xs"
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
                  'Save this Profile to check the maximum.'
                ) : largest ? (
                  `No recommended context for ${largest.sessions} ${largest.sessions === 1 ? 'session' : 'sessions'}.`
                ) : (
                  'Max is checked at start.'
                )
              }
              control={
                <Controller
                  control={form.control}
                  name="context"
                  render={({ field }) => (
                    <NumberInput
                      id="server-context"
                      min={
                        table?.minimum_context_tokens ?? contextAllowed.minimum
                      }
                      max={table?.context_max_tokens ?? contextAllowed.maximum}
                      step={contextAllowed.type === 'integer' ? 1 : undefined}
                      {...field}
                      value={
                        typeof field.value === 'number'
                          ? field.value
                          : undefined
                      }
                      onValueChange={field.onChange}
                      aria-invalid={!!form.formState.errors.context}
                    />
                  )}
                />
              }
            />
            <SettingRow
              name="sessions"
              label="Sessions"
              htmlFor="server-sessions"
              about={
                <p>
                  How many conversations the server handles at the same time.
                  Each session has its own context.
                </p>
              }
              control={
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
                        typeof field.value === 'number'
                          ? field.value
                          : undefined
                      }
                      onValueChange={field.onChange}
                      aria-invalid={!!form.formState.errors.sessions}
                    />
                  )}
                />
              }
            />
          </SettingGroup>
          <SideFiles
            view={view}
            plainFile={plainFile}
            unsaved={
              draft.model !==
              (view.saved.model_path || view.saved.manifest_path)
            }
            saveModel={saveModelChoice}
            cacheRow={
              registry.has('Cache') ? (
                <SettingGroup>
                  {registry.renderRow('--kv-cache-type')}
                  {registry.renderRow('--refusal-direction')}
                </SettingGroup>
              ) : undefined
            }
          />
          {/* The line of a setup the engine cannot judge before it starts sits
              where the Fits line goes, above Inference, so an open Inference
              section does not push it out of sight. */}
          {checkedAtStart && (
            <p aria-label="Memory fit" className="text-sm">
              {guiError({ code: 'estimate_needs_start', remedy: '' }).message}
            </p>
          )}
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
          {addresses}

          <section aria-label="Inference" className="space-y-2 border-t pt-3">
            <h2>
              <Button
                type="button"
                variant="ghost"
                className="h-auto w-full justify-start gap-1.5 px-0 text-lg"
                aria-expanded={inferenceOpen}
                aria-controls={inferenceId}
                disabled={disclosure.isPending}
                onClick={toggleInference}
              >
                <span
                  aria-hidden="true"
                  className="inline-block w-3 text-muted-foreground"
                >
                  {inferenceOpen ? '▾' : '▸'}
                </span>
                Inference
              </Button>
            </h2>
            {!inferenceOpen && (
              <p
                className="truncate text-sm text-muted-foreground"
                title={registry.summary}
              >
                {registry.summary}
              </p>
            )}
            <div id={inferenceId} hidden={!inferenceOpen}>
              {registry.render('Inference')}
            </div>
          </section>
          {Object.entries(form.formState.errors).map(([name, error]) => (
            <ErrorNote key={name} className="text-sm text-destructive">
              <span>{String(error.message)}</span>
            </ErrorNote>
          ))}
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
              {view.restart_needed?.map((change) => {
                const words = pendingChangeWords(change)
                return (
                  <p key={change.key}>
                    {words.label}: {words.running || 'not reported'} →{' '}
                    {words.next}
                  </p>
                )
              })}
            </div>
          )}
          <div className="sticky bottom-0 z-10 flex items-center gap-3 border-t bg-background py-2">
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
    </CapabilitySource.Provider>
  )
}
