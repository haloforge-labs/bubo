import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { coreView, mountCorePage } from '@/components/core-page-test-support'

// A hard scroll budget. The default window is 1280 x 860 (about 829 px of page
// under the title bar), the sidebar open. A page's ratio is its height divided by
// the height of the window: 1.0 is no scrolling. A page that grows past its
// budget fails here, so the pages cannot drift back into long scrolls.
const width = 1280
const height = 829

type View = ReturnType<typeof coreView>

const pages: {
  name: string
  path: string
  budget: number
  view: () => View
  prepare?: () => Promise<void>
}[] = [
  {
    name: 'Server, stopped',
    path: '/',
    budget: 1.0,
    view: () => coreView('stopped-saved-estimate'),
  },
  {
    name: 'Server, running, Inference open',
    path: '/',
    budget: 1.5,
    view: () => {
      const view = coreView('ready-busy')
      view.saved.inference_expanded = true
      return view
    },
  },
  {
    name: 'Settings, Advanced, Debug closed',
    path: '/settings',
    budget: 2.0,
    view: () => coreView('stopped-saved-estimate'),
    prepare: async () => {
      await page.getByRole('tab', { name: 'Advanced' }).click()
    },
  },
  {
    name: 'Settings, App',
    path: '/settings',
    budget: 1.0,
    view: () => coreView('stopped-saved-estimate'),
  },
  {
    name: 'Models',
    path: '/models',
    budget: 1.0,
    view: () => coreView('ready-idle'),
  },
  {
    name: 'Connect',
    path: '/connect',
    budget: 1.0,
    view: () => coreView('ready-idle'),
  },
  {
    name: 'Connect, network access on',
    path: '/connect',
    budget: 1.0,
    view: () => coreView('lan-on'),
  },
  {
    name: 'Requests, running',
    path: '/requests',
    budget: 1.0,
    view: () => coreView('ready-busy'),
  },
  {
    name: 'Requests, empty',
    path: '/requests',
    budget: 1.0,
    view: () => {
      const view = coreView('stopped')
      view.requests = []
      return view
    },
  },
  {
    name: 'Logs',
    path: '/logs',
    budget: 1.0,
    view: () => coreView('ready-idle'),
  },
  {
    name: 'About',
    path: '/about',
    budget: 1.0,
    view: () => coreView('update-none'),
  },
  {
    // The first run: home shows Get started while nothing is prepared.
    name: 'Onboarding',
    path: '/',
    budget: 1.0,
    view: () => coreView('not-prepared'),
  },
  {
    // The model choice (three choices, the lower-memory one selected) once the
    // engine files are there. The home route would show Server for this view, so
    // the page is opened by its own route.
    name: 'Onboarding, engine ready',
    path: '/onboarding',
    budget: 1.0,
    view: () => {
      const view = coreView('onboarding-96-64')
      view.runtime.readiness.runtime = true
      return view
    },
  },
]

for (const item of pages)
  test(`${item.name} stays within ${item.budget} screens`, async () => {
    await page.viewport(width, height)
    // The memory history takes any route; the helper's type names a few.
    mountCorePage(item.path as '/', item.view())
    await expect.element(page.getByRole('heading', { level: 1 })).toBeVisible()
    if (item.prepare) await item.prepare()
    await new Promise((resolve) => setTimeout(resolve, 600))
    const ratio =
      document.documentElement.scrollHeight /
      document.documentElement.clientHeight
    console.log(
      `RATIO ${item.name}: ${ratio.toFixed(2)} (budget ${item.budget})`,
    )
    expect(
      ratio,
      `${item.name} is ${ratio.toFixed(2)} screens`,
    ).toBeLessThanOrEqual(item.budget + 0.005)
  })
