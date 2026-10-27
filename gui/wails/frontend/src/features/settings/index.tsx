import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { useView } from '@/lib/bindings'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useState } from 'react'
import { useBlocker, useLocation } from '@tanstack/react-router'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { RegistrySection } from './registry-section'
import { AppSettings } from './app-settings'
import { RejectedOverride } from '@/components/rejected-override'
import { GUIError } from '@/components/gui-error'

export default function SettingsPage() {
  const query = useView()
  const view = query.data
  const location = useLocation()
  const [tab, setTab] = useState<string>()
  const [dirty, setDirty] = useState(false)
  const [nextTab, setNextTab] = useState<string>()
  const blocker = useBlocker({
    shouldBlockFn: () => dirty,
    withResolver: true,
    enableBeforeUnload: dirty,
  })
  function stay() {
    setNextTab(undefined)
    blocker.reset?.()
  }
  return (
    <Page title="Settings">
      {query.error && (
        <ErrorNote>
          <span>{query.error.message}</span>
        </ErrorNote>
      )}
      {view ? (
        <>
          <GUIError error={view.failure} recovery={!view.failure?.switch} />
          <RejectedOverride failure={view.failure} />
          <Tabs
            value={tab ?? (location.hash === 'inference' ? 'advanced' : 'app')}
            onValueChange={(next) => {
              if (dirty) setNextTab(next)
              else setTab(next)
            }}
            className="space-y-6"
          >
            <TabsList>
              <TabsTrigger value="app">App</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            <TabsContent value="app">
              <AppSettings view={view} onDirtyChange={setDirty} />
            </TabsContent>
            <TabsContent value="advanced" className="space-y-8">
              <RegistrySection
                view={view}
                audience="advanced"
                includeUserRemainder
                onDirtyChange={setDirty}
              />
            </TabsContent>
          </Tabs>
          <Dialog
            open={nextTab !== undefined || blocker.status === 'blocked'}
            onOpenChange={(open) => {
              if (!open) stay()
            }}
          >
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Unsaved changes</DialogTitle>
                <DialogDescription>
                  Save your changes before leaving, or discard them.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={stay}>
                  Keep editing
                </Button>
                <Button
                  onClick={() => {
                    setDirty(false)
                    if (nextTab !== undefined) {
                      setTab(nextTab)
                      setNextTab(undefined)
                    } else blocker.proceed?.()
                  }}
                >
                  Discard and leave
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </>
      ) : (
        !query.error && <p>Loading settings…</p>
      )}
    </Page>
  )
}
