# Bubo frontend shell

This is the production frontend served and embedded by Wails. Follow
[the architecture](../../ARCHITECTURE.md). Build assets are ignored; a tracked
`dist/.gitkeep` lets Go-only checks compile. `wails build` builds the real assets.

The layout is imported from `satnaing/shadcn-admin` at
`e16c87f213a5ba5e45964e9b67c792105ec74d26` (MIT), using the upstream
source. It supplies `components/layout/{app-sidebar,app-title,nav-group,header,main,types}`,
`components/layout/data/sidebar-data`, `components/ui/{sidebar,button,sheet,tooltip}`,
and `hooks/use-mobile`. The frontend adds the template's dialog, select, badge, card,
table, tabs, switch, alert, textarea and collapsible primitives. Progress comes
from this repository's CLI-imported shadcn source at `app-base`, because the
supplied template has no progress primitive. The existing form primitives remain.
See [the source manifest](src/vendor/shadcn-admin-SOURCE.md) for exact template
paths and import adaptations, and [generated GUI inventory](../../../docs/release/THIRD-PARTY-GUI.md) for licensing.

The template's sidebar provider handles desktop icon collapse and its Radix
sheet handles small screens. Navigation lives in
`src/components/layout/data/sidebar-data.ts`: Server, Models, Connect, Requests,
Logs, Settings and About. Onboarding replaces the home content through its
preparation input; it has no sidebar entry. The header retains state and address
slots. Account/team controls, authentication, demo data, search, top navigation
and theme switching are omitted; theme follows `data-theme` or the system.

To update the template, obtain an approved local source checkout at an explicit
commit, compare only the manifest's files, and reapply the documented import
adaptations and Bubo navigation/branding. Do not run the upstream scripts or copy
its package manifest. Preserve the strict guards and existing pinned packages;
add required dependencies at exact versions. Update the source manifest, licence
and notice, then run every verification command below and inspect the four PNGs.

## Develop and verify

Use Node 24.19 or later. All direct versions and the transitive lockfile are pinned.

```sh
npm ci
npm run gen:types
npm run routes
npm run dev
```

```sh
npm run typecheck
npm run lint
npm run knip
npm run format:check
npm test
npm run test:shots
npm run build
```

The build is a normal Vite folder with relative asset paths and TanStack **hash
history**. Wails serves a folder, so `vite-plugin-singlefile` is unnecessary.
Generated route files are kept in `src/generated/` alongside schema types.
`gen:types` uses the canonical `tests/cli/contract/switches.schema.json` and emits
the named `SwitchRow` type consumed by the registry helper.

Vite 8, React 19, Tailwind 4, ESLint 10 and Vitest 4 are pinned. TypeScript is
6.0.3 because typescript-eslint 8.71.2 declares a peer range below 6.1; TS 7
is outside that supported range. Playwright **1.63.0** names Chromium revision
**1243**, already installed under `%LOCALAPPDATA%/ms-playwright`. Never run
`playwright install` during frontend checks. Browser tests are headless and use only fakes;
they never launch Wails or a CLI/model server. TanStack Table is pinned to the
template's compatible 8.21.3. React Compiler is not enabled; the compiler-specific
`incompatible-library` lint warning is disabled while ordinary hook checks remain.

Vitest defines two projects: `unit` (Node) and `browser` (headless Chromium).
Set `BUBO_SHOTS_DIRECTORY` to an evidence folder and `BUBO_GPU_LOCK` to the shared
timing lock path before running screenshot tests.

`test:shots` selects `*.shots.tsx` in the browser project, with light/dark Ã—
device scale 1/1.5. The shell capture is fixed at 1100 Ã— 760 CSS pixels. PNGs go
to `C:/path/to/shots/D3/<feature>/`, outside git. The fixture loader types
all Go-generated `src/scenarios/*.json` as `View`; `_source` distinguishes
recorded CLI data and app fixtures. The shell screenshots select `ready-idle`,
a recorded scenario. There is no placeholder. Feature tests select their affected
scenarios separately; `ready-busy` is the frozen active-session scenario.

ESLint bans explicit `any`, cross-feature imports and window/global state. Only
`src/lib/bindings.ts` may import `wailsjs`. Generated `wailsjs/` and
`src/generated/` are excluded from ESLint, formatting and knip. Go-generated
scenario JSON is also excluded from formatting so regeneration stays exact. Knip has no
dependency/export suppression. Wails is outside its project roots. The fact
guard parses source literals and excludes only scenarios and generated code.
Presentation tokens `default` and `standard` are allowed; the reserved CLI profile
tokens and forbidden numeric/URL literals remain checked.

## Frozen desktop boundary for D1/D2

Only `src/lib/bindings.ts` imports generated Wails modules. `View` is derived
recursively from `supervisor.View`, excluding generated class conversion methods
because the transport returns JSON. No fields are hand-copied. A plain settings
patch is accepted by `actions.PatchPreferences` without requiring a generated
class method; all generated wire fields remain typed.

Pages import `useView`, `actions` and types **only from `@/lib/bindings`**.
`useView` calls generated `GetView`; every other generated App method is exposed
once as `actions.<GeneratedMethodName>`. Prefer structured methods such as
`Start`, `Stop`, `PatchPreferences`, `Download` and the five update actions.
Actions invalidate `['view']`
on success and rejection and retain typed error results. Do not treat an
`ActionResult.error` as success. Missing services return typed `unsupported`.

`main.tsx` installs `subscribeViewEvents` with effect cleanup. The four names
(`prepare:progress`, `download:progress`, `log:line`, `update:progress`) and payload
projections live once in the boundary. Streams update the existing View cache;
an event before the first GetView cannot invent a partial View. Polling remains
authoritative for state transitions. Tests use `connectBindings` and the injected
event subscriber without Wails globals; disconnect restores the real bridge.

`useView()` polls at 2 seconds while visible and 10 seconds while hidden;
`visibilitychange` immediately updates the Query interval. Once the generated
View is available, `ViewShell` supplies its state, endpoints and update to
`AppShell`, and the home route passes its preparation state to `HomePage`.
`AppShell.renderUpdateBanner(update)` is D2's one-line update slot. Only supplied
endpoint URLs are shown; missing URLs remain unavailable. `false` shows Onboarding,
`true` shows Server, and absence stays visibly unsupported. These display inputs
are layout contracts, not another handwritten View.

## Add a page

Add one route file, one `features/<name>/` folder and one nav entry in
`src/components/layout/data/sidebar-data.ts`. View and bindings change only if
the page needs new data. A feature never imports another feature; use `@/components` or `@/lib` for
shared code, and `./` for its own children. The router plugin generates the tree.

## Add a setting

Add or edit its row in `release_switches.json` (`label`, `purpose`, `audience`,
`group`, `gui_control`, `allowed`). Nothing in Go or TS changes: Settings renders
rows by audience (user: main panel, advanced: Advanced, debug: Debug) and group.
D1 supplies those rows from the generated View to the shared `RegistryFields`.
`fieldFromRow` chooses a control; `schemaFromAllowed` builds validation without
switch-specific branches. The synthetic registry extension is covered in both
Node and Chromium tests.

The helper uses the CLI row's optional `label`, falling back to its `name`.
Unknown formats return typed
`UnsupportedRegistryError`; the form displays the failure and offers no save.
The CLI's human-described formats (for example tuning lists) need a machine
validation contract or a later explicit UI treatment. The CLI remains the judge.

For another model, add it to the CLI catalogue. For another machine field,
change the schema + Go struct, regenerate Wails, then use it in the feature.
For another stream, add its Wails event name in Go and its subscriber in
`lib/bindings.ts`.

## Integrated page writes

Settings groups the CLI registry by audience and group. Unknown boolean values
are indeterminate. Tagged `Value` constraints are decoded before the shared
registry helper runs; unsupported formats and integers outside JavaScript's safe
range stay explicitly unsupported. App fields use React Hook Form drafts.

Server setup, model selection, app settings and LAN use the typed partial
`PatchPreferences` binding. Only edited fields are sent; clear lists restore
defaults without copying settings metadata or other saved requests.
Settings uses `SetSwitchOverrides` for registry rows, including sampling and
reasoning. Eligibility and pending/unknown restart wording come from
`View.switch_overrides`; defaults and constraints come from the registry.
The CLI's shared Behaviour group appears in both Inference and Reasoning; the
latter uses range controls for bounded numeric rows. No switch-specific routing
or write adapters are maintained. A boolean row with explicit allowed values
uses those wire values in a select; an ordinary boolean uses a checkbox.
Unsupported formats remain visible and saved requests remain clearable.
Successful writes persist requests; the CLI validates them on start. Typed save
errors stay next to their controls and asynchronous startup errors remain visible.

Connect renders supplied endpoint URLs, the CLI's Pi snippet and client templates.
Templates substitute the supplied endpoints or the LAN origin selected by the
user while preserving the CLI endpoint path; plain copies retain the key token.
Only Copy address and key fetches and substitutes a key. Show keeps
a key in component state for 30 seconds, or until Hide/leave/visibility change;
Keep showing extends that display until Hide or leave. Only Copy address and key asks
for a key to place into the existing JSON snippet and clipboard. Reset requires
its own dialog. Secrets never enter the View cache, drafts or screenshot fixtures.

About owns the update card; the shared header banner links to About. All six
update states and all five update actions preserve the supervisor's state and
typed failures. The updater runs through the existing bindings.

Run all scenario captures once with `BUBO_SHOTS_LANE=D3 npm run test:shots`.
The single screenshot runner checks the timing lock, uses headless Chromium and
accepts optional file filters. Captures use light/dark themes and both device
scales; PNGs stay under `C:/path/to/shots/D3/<feature>/` when using that example output folder.
The preserved key reset/store-failure recordings now live under
`gui/tests/scenarios/recordings/legacy-key/`. They remain raw legacy recordings;
tests combine only their key metadata with generated Views and do not call them
new canonical Views. All other scenarios are generated by the Go View builder.
The plain `npm run format:check` uses the repository's LF rule.

After capturing, run `python gui/tests/contact_sheets.py C:/path/to/shots/D3`
from the repository root. It writes one labelled PNG per page under the same
output `contact-sheets/` folder without changing the original captures.
