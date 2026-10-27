# Imported shadcn-admin source

- Repository: `satnaing/shadcn-admin`.
- Exact upstream commit: `e16c87f213a5ba5e45964e9b67c792105ec74d26`.
- Source: upstream shadcn-admin source snapshot at commit `e16c87f2`.
- Licence: MIT, Copyright (c) 2024 Sat Naing; full upstream `LICENSE` copied as
  [shadcn-admin-LICENSE.txt](shadcn-admin-LICENSE.txt).

Paths are relative to the template root and this frontend root respectively.

| Template source                              | Imported destination                         |
| -------------------------------------------- | -------------------------------------------- |
| `src/components/layout/app-sidebar.tsx`      | `src/components/layout/app-sidebar.tsx`      |
| `src/components/layout/app-title.tsx`        | `src/components/layout/app-title.tsx`        |
| `src/components/layout/nav-group.tsx`        | `src/components/layout/nav-group.tsx`        |
| `src/components/layout/header.tsx`           | `src/components/layout/header.tsx`           |
| `src/components/layout/main.tsx`             | `src/components/layout/main.tsx`             |
| `src/components/layout/types.ts`             | `src/components/layout/types.ts`             |
| `src/components/layout/data/sidebar-data.ts` | `src/components/layout/data/sidebar-data.ts` |
| `src/components/ui/sidebar.tsx`              | `src/components/ui/sidebar.tsx`              |
| `src/components/ui/button.tsx`               | `src/components/ui/button.tsx`               |
| `src/components/ui/sheet.tsx`                | `src/components/ui/sheet.tsx`                |
| `src/components/ui/tooltip.tsx`              | `src/components/ui/tooltip.tsx`              |
| `src/components/ui/dialog.tsx`               | `src/components/ui/dialog.tsx`               |
| `src/components/ui/select.tsx`               | `src/components/ui/select.tsx`               |
| `src/components/ui/badge.tsx`                | `src/components/ui/badge.tsx`                |
| `src/components/ui/card.tsx`                 | `src/components/ui/card.tsx`                 |
| `src/components/ui/table.tsx`                | `src/components/ui/table.tsx`                |
| `src/components/ui/tabs.tsx`                 | `src/components/ui/tabs.tsx`                 |
| `src/components/ui/switch.tsx`               | `src/components/ui/switch.tsx`               |
| `src/components/ui/alert.tsx`                | `src/components/ui/alert.tsx`                |
| `src/components/ui/textarea.tsx`             | `src/components/ui/textarea.tsx`             |
| `src/components/ui/collapsible.tsx`          | `src/components/ui/collapsible.tsx`          |
| `src/hooks/use-mobile.tsx`                   | `src/hooks/use-mobile.tsx`                   |
| `LICENSE`                                    | `src/vendor/shadcn-admin-LICENSE.txt`        |

## Import adaptations

- Keep the template's sidebar/provider, flat-link navigation branch, header and
  main layout. Remove team/account/auth/demo data, nested navigation and badges.
  The frontend adds collapsible for the feature pages; the flat sidebar does not use it.
- Replace template navigation data and branding with Bubo's entries and title.
  Narrow the link type to TanStack's registered routes; derive exact active state
  from the router pathname and close the mobile sheet when following a link.
- Use the already pinned `radix-ui` namespace exports instead of adding separate
  packages for the same Radix primitives. Mark `VariantProps` type-only for strict
  TypeScript. Pin `class-variance-authority` at `0.7.1` from the offline npm cache.
- Prune unused primitive declarations and exports so knip needs no suppressions:
  sidebar input/separator/submenus/actions/badges/skeleton, sheet trigger/close
  wrapper/footer, and the exported button variant helper. Keep Radix's built-in
  close button, focus handling, keyboard dismissal and accessibility primitives.
- Remove the template's cookie persistence; saved GUI preferences belong to the
  supervisor. Sidebar open state is component state. Bind the keyboard listener
  to `document` and use standard `matchMedia`, retaining cleanup and avoiding
  window globals. No application state is stored on a browser global.
- The frontend restores the upstream presentation variant/size name `default`. The fact
  guard permits `default` and `standard` as presentation tokens while retaining
  the forbidden CLI literals.
- The frontend's new imports use the same Radix namespace adaptation. Keep internal
  dialog portal/overlay and select scroll controls unexported, and the badge
  variant helper private; the remaining page-facing exports are exercised by
  component integration tests. No unused-export suppression is needed.
- The header wraps on narrow screens; retain state/address slots. Sidebar theme
  colors reference Bubo's existing semantic tokens. No theme switch is imported.
- Apply repository Prettier formatting. Future changes should remain in the
  layout/composition and theme tokens; primitive changes require an upstream
  update or a documented import compatibility requirement.

Progress is absent from the supplied template. The frontend reuses the existing shadcn/ui
CLI-imported `gui/wails/frontend/src/components/ui/progress.tsx` from repository
revision `14a8b9ecbe9de17c271062c80e5be0352584061b`, with formatting only, just as
the scaffold reused the other existing primitives. It is covered by the shadcn/ui
MIT notice in `shadcn-LICENSE.txt`. No source application or package manifest is copied.

The existing input, label, checkbox and separator were previously imported from
the Bubo `app-base` revision, with primitive notices in `shadcn-LICENSE.txt`; they were not
replaced by this import. `AppShell` composes the imported layout with the existing
feature routes and typed header slots. The frontend integrates the generated bindings
and View fixtures; D1/D2 consume that frozen boundary.
