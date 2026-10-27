import { links } from '@/lib/links'
import { ErrorNote } from '@/components/error-note'
import { Page } from '@/components/page'
import { CopyText } from '@/components/copy-text'
import { actions, useView } from '@/lib/bindings'
import { UpdateCard } from './update-card'
import { ThirdPartyNotices } from './notices'
import { WhatsNew } from './whats-new'

export default function AboutPage() {
  const query = useView()
  const view = query.data
  const version = view?.product?.version || view?.update.current_version
  return (
    <Page title="About">
      {query.error && (
        <ErrorNote>
          <span>{query.error.message}</span>
        </ErrorNote>
      )}
      {view ? (
        <div className="space-y-8">
          <UpdateCard
            update={view.update}
            availability={view.actions}
            serverRunning={view.serving}
          />
          <section aria-label="This build" className="space-y-4">
            <h2 className="text-2xl">This build</h2>
            <div className="flex flex-wrap items-center gap-2">
              <p>
                Bubo{' '}
                {version ? (
                  <span className="tabular-nums">{version}</span>
                ) : (
                  'Version not reported'
                )}
                {view.product?.revision && (
                  <>
                    {' '}
                    · <code>{view.product.revision}</code>
                  </>
                )}
              </p>
              {view.product && (
                <CopyText
                  iconOnly
                  variant="ghost"
                  label="Copy version"
                  text={`Bubo ${view.product.version} · ${view.product.revision}`}
                />
              )}
            </div>
          </section>
          <WhatsNew version={version} />
          <nav
            aria-label="Bubo links"
            className="flex flex-wrap gap-6 underline underline-offset-4"
          >
            <a
              href={version ? links.release(version) : links.releases}
              onClick={(event) => {
                event.preventDefault()
                void actions.OpenExternal(
                  version ? links.release(version) : links.releases,
                )
              }}
            >
              Release notes
            </a>
            <a
              href={links.website}
              onClick={(event) => {
                event.preventDefault()
                void actions.OpenExternal(links.website)
              }}
            >
              Website
            </a>
            <a
              href={links.docs}
              onClick={(event) => {
                event.preventDefault()
                void actions.OpenExternal(links.docs)
              }}
            >
              Documentation
            </a>
            <a
              href={links.repository}
              onClick={(event) => {
                event.preventDefault()
                void actions.OpenExternal(links.repository)
              }}
            >
              Source and support
            </a>
          </nav>
          <details className="border-t pt-4">
            <summary className="min-h-10 cursor-pointer text-xl">
              Third-party notices
            </summary>
            <div className="max-h-96 overflow-auto pt-4">
              <ThirdPartyNotices />
            </div>
          </details>
        </div>
      ) : (
        !query.error && <p>Loading build information…</p>
      )}
    </Page>
  )
}
