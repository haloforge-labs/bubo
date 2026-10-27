import { DropdownMenu } from 'radix-ui'
import { ChevronDownIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function ImportMenu({
  disabled,
  browse,
}: {
  disabled: boolean
  browse: (kind: 'file' | 'folder') => void
}) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button variant="outline" disabled={disabled}>
          Import <ChevronDownIcon />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={4}
          className="z-50 min-w-40 rounded-md border bg-popover p-1 text-sm text-popover-foreground shadow-md"
        >
          {(['file', 'folder'] as const).map((kind) => (
            <DropdownMenu.Item
              key={kind}
              className="cursor-pointer rounded-sm px-3 py-2 outline-none focus:bg-accent focus:text-accent-foreground"
              onSelect={() => browse(kind)}
            >
              Model {kind}…
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}
