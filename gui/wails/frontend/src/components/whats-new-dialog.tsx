import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  hasWhatsNew,
  WhatsNewList,
  whatsNewRelease,
} from '@/features/about/whats-new'
import { actions, useView } from '@/lib/bindings'

function recordSeen(version: string) {
  // A failed save only means the dialog can show again at the next start.
  void actions
    .PatchPreferences({ set: { last_seen_version: version }, clear: [] })
    .catch(() => undefined)
}

// Shows the release text once, at the first start after an update. The saved
// last_seen_version lives in the app settings. A new installation (setup not
// finished, nothing seen yet) records the version and shows nothing: its first
// screen is setup. An install made by a version that never recorded one has
// its engine and model ready, so it counts as an update.
export function WhatsNewDialog() {
  const view = useView().data
  const version = view?.product?.version || view?.update.current_version
  const seen = view?.saved.last_seen_version
  const setupDone = !!(
    view?.runtime.readiness.runtime && view.runtime.readiness.model
  )
  const relevant = !!version && hasWhatsNew(version) && seen !== version
  const fresh = relevant && !seen && !setupDone
  const [closed, setClosed] = useState(false)
  const recorded = useRef(false)
  useEffect(() => {
    if (!fresh || recorded.current || !version) return
    recorded.current = true
    recordSeen(version)
  }, [fresh, version])
  if (!version) return null
  return (
    <Dialog
      open={relevant && !fresh && !closed}
      onOpenChange={(next) => {
        if (next) return
        setClosed(true)
        recordSeen(version)
      }}
    >
      <DialogContent
        showCloseButton={false}
        className="max-h-[80vh] overflow-y-auto"
      >
        <DialogTitle className="text-xl font-normal">
          What's new in {whatsNewRelease}
        </DialogTitle>
        <DialogDescription className="sr-only">
          Changes in this version of Bubo
        </DialogDescription>
        <WhatsNewList />
        <DialogFooter>
          <Button
            onClick={() => {
              setClosed(true)
              recordSeen(version)
            }}
          >
            OK
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
