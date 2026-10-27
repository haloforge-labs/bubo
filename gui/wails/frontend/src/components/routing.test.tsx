import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
} from '@tanstack/react-router'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { afterEach, expect, test } from 'vitest'
import { page, userEvent } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { createAppRouter } from '@/router'
import { HomePage } from '@/routes'
import { connectBindings } from '@/lib/bindings'
import { createQueryClient } from '@/lib/query-client'
import { AppShell } from '@/components/app-shell'
import { scenarioByName } from '../../tests/scenarios'
import { sidebarData } from '@/components/layout/data/sidebar-data'

const clients: ReturnType<typeof createQueryClient>[] = []
const disconnects: (() => void)[] = []
afterEach(() => {
  clients.splice(0).forEach((client) => client.clear())
  disconnects
    .splice(0)
    .reverse()
    .forEach((stop) => stop())
})

test('template navigation resolves every registry page using hash history', async () => {
  const router = createAppRouter()
  router.history.replace('/')
  const client = createQueryClient()
  clients.push(client)
  disconnects.push(
    connectBindings({ getView: async () => scenarioByName('ready-idle').view }),
  )
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  const navigation = page.getByRole('navigation', { name: 'Main navigation' })
  const links = sidebarData.navGroups.flatMap((group) => group.items)
  await expect
    .poll(() => document.querySelectorAll('nav a').length)
    .toBe(links.length)
  await expect
    .element(navigation.getByRole('link', { name: 'Get started' }))
    .not.toBeInTheDocument()
  for (const { url: to, title } of links) {
    const link = navigation.getByRole('link', { name: title, exact: true })
    await link.click()
    await expect
      .element(page.getByRole('heading', { name: title, exact: true }))
      .toBeVisible()
    expect(document.location.hash).toBe(`#${to}`)
    await expect.element(link).toHaveAttribute('data-active', 'true')
    await expect.element(link).toHaveAttribute('aria-current', 'page')
    expect(document.querySelectorAll('nav [data-active="true"]')).toHaveLength(
      1,
    )
  }
  await router.navigate({ to: '/onboarding' })
  await expect
    .element(page.getByRole('heading', { name: 'Get started' }))
    .toBeVisible()
  router.history.replace('/')
})

test('the template sidebar collapses and preserves header slots', async () => {
  const routeTree = createRootRoute({
    component: () => (
      <AppShell
        serverState="Fixture state"
        addresses={{
          openai_base_url: 'https://example.test/openai',
          anthropic_base_url: 'https://example.test/anthropic',
        }}
        update={scenarioByName('update-available').view.update}
        renderUpdateBanner={(update) => (
          <span>Update: {update.available_version}</span>
        )}
      >
        <h1>Shell fixture</h1>
      </AppShell>
    ),
  })
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: ['/'] }),
  })
  const client = createQueryClient()
  clients.push(client)
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('Fixture state')
  await expect
    .element(page.getByLabelText('Server addresses'))
    .toHaveTextContent('Copy OpenAI address')
  await expect
    .element(page.getByLabelText('Server addresses'))
    .toHaveTextContent('Copy Anthropic address')
  await expect
    .element(page.getByLabelText('Application update'))
    .toHaveTextContent(
      `Update: ${scenarioByName('update-available').view.update.available_version}`,
    )
  const sidebar = document.querySelector('[data-slot="sidebar"][data-state]')
  await expect.poll(() => sidebar?.getAttribute('data-state')).toBe('expanded')
  await page.getByRole('button', { name: 'Toggle Sidebar' }).last().click()
  await expect.poll(() => sidebar?.getAttribute('data-state')).toBe('collapsed')
  await userEvent.keyboard('{Control>}b{/Control}')
  await expect.poll(() => sidebar?.getAttribute('data-state')).toBe('expanded')
})

test('the template mobile sheet closes on navigation and Escape', async () => {
  await page.viewport(390, 760)
  try {
    const router = createAppRouter()
    router.update({
      history: createMemoryHistory({ initialEntries: ['/models'] }),
    })
    const client = createQueryClient()
    clients.push(client)
    disconnects.push(
      connectBindings({
        getView: async () => scenarioByName('ready-idle').view,
      }),
    )
    render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    )
    await expect
      .element(page.getByRole('heading', { name: 'Models' }))
      .toBeVisible()
    await expect
      .element(page.getByRole('navigation', { name: 'Main navigation' }))
      .not.toBeInTheDocument()
    await page.getByRole('button', { name: 'Toggle Sidebar' }).click()
    await expect
      .element(page.getByRole('dialog', { name: 'Sidebar' }))
      .toBeVisible()
    await page.getByRole('link', { name: 'Connect', exact: true }).click()
    await expect
      .element(page.getByRole('heading', { name: 'Connect' }))
      .toBeVisible()
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
    await page.getByRole('button', { name: 'Toggle Sidebar' }).click()
    await expect.element(page.getByRole('dialog')).toBeVisible()
    await userEvent.keyboard('{Escape}')
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
    expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(390)
  } finally {
    await page.viewport(1100, 760)
  }
})

test.each([
  [false, 'Get started'],
  [true, 'Server'],
] as const)(
  'preparation input %s chooses %s',
  async (packagePrepared, title) => {
    render(<HomePage packagePrepared={packagePrepared} />)
    await expect
      .element(page.getByRole('heading', { name: title }))
      .toBeVisible()
  },
)

test('unknown preparation never implies readiness', async () => {
  render(<HomePage />)
  await expect
    .element(
      page.getByText(
        'Package preparation status is unavailable until the desktop View is connected.',
      ),
    )
    .toBeVisible()
})

test('a first call that waits shows the one plain first-start line from Go', async () => {
  render(
    <HomePage
      packagePrepared={false}
      startingCLI
      statusLabel="Starting for the first time…"
    />,
  )
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('Starting for the first time…')
})

test('a cold CLI handshake shows its typed phase before deciding onboarding', async () => {
  render(
    <HomePage
      packagePrepared={false}
      startingCLI
      statusLabel="Checking this computer…"
    />,
  )
  // The body says what the header says: one wording from Go.
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('Checking this computer…')
  await expect
    .element(page.getByRole('heading', { name: 'Onboarding' }))
    .not.toBeInTheDocument()
})

test('binding failure is visible in the shell', async () => {
  const router = createAppRouter()
  router.update({
    history: createMemoryHistory({ initialEntries: ['/models'] }),
  })
  const client = createQueryClient()
  clients.push(client)
  disconnects.push(
    connectBindings({
      getView: () =>
        Promise.reject(new Error('Fixture transport unavailable.')),
    }),
  )
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  await expect
    .element(page.getByRole('status'))
    .toHaveTextContent('Fixture transport unavailable.')
})
