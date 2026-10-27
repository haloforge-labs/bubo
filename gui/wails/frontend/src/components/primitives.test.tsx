import { useState } from 'react'
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { expect, test } from 'vitest'
import { page } from 'vitest/browser'
import { render } from 'vitest-browser-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

test('dialog supports accessible description, explicit close and focus return', async () => {
  render(
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="default" size="default">
          Open fixture
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Fixture confirmation</DialogTitle>
          <DialogDescription>Review the fixture.</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button>Return</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>,
  )
  await page.getByRole('button', { name: 'Open fixture' }).click()
  await expect
    .element(page.getByRole('dialog', { name: 'Fixture confirmation' }))
    .toBeVisible()
  await expect.element(page.getByText('Review the fixture.')).toBeVisible()
  await page.getByRole('button', { name: 'Return', exact: true }).click()
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument()
  await expect
    .element(page.getByRole('button', { name: 'Open fixture' }))
    .toHaveFocus()
})

test('select, switch and textarea keep their editable state', async () => {
  render(
    <>
      <Select defaultValue="first">
        <SelectTrigger aria-label="Fixture choice">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectLabel>Fixture options</SelectLabel>
            <SelectItem value="first">First fixture</SelectItem>
            <SelectSeparator />
            <SelectItem value="second">Second fixture</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
      <Switch aria-label="Fixture enabled" />
      <Textarea aria-label="Fixture notes" />
    </>,
  )
  await page.getByRole('combobox', { name: 'Fixture choice' }).click()
  await page.getByRole('option', { name: 'Second fixture' }).click()
  await expect
    .element(page.getByRole('combobox'))
    .toHaveTextContent('Second fixture')
  await page.getByRole('switch').click()
  await expect.element(page.getByRole('switch')).toBeChecked()
  await page.getByRole('textbox').fill('Fixture text')
  await expect.element(page.getByRole('textbox')).toHaveValue('Fixture text')
})

test('tabs and collapsible expose only the requested panel', async () => {
  render(
    <Tabs defaultValue="first">
      <TabsList aria-label="Fixture panels">
        <TabsTrigger value="first">First panel</TabsTrigger>
        <TabsTrigger value="second">Second panel</TabsTrigger>
      </TabsList>
      <TabsContent value="first">First content</TabsContent>
      <TabsContent value="second">
        <Collapsible>
          <CollapsibleTrigger>More fixture details</CollapsibleTrigger>
          <CollapsibleContent>Expanded fixture</CollapsibleContent>
        </Collapsible>
      </TabsContent>
    </Tabs>,
  )
  await page.getByRole('tab', { name: 'Second panel' }).click()
  await expect.element(page.getByText('First content')).not.toBeInTheDocument()
  await page.getByRole('button', { name: 'More fixture details' }).click()
  await expect.element(page.getByText('Expanded fixture')).toBeVisible()
})

test('tooltip and status primitives compose without a page implementation', async () => {
  render(
    <Card>
      <CardHeader>
        <CardTitle>Fixture status</CardTitle>
        <CardDescription>Fixture description</CardDescription>
        <CardAction>
          <Badge variant="default">Fixture badge</Badge>
        </CardAction>
      </CardHeader>
      <CardContent>
        <Alert>
          <AlertTitle>Fixture notice</AlertTitle>
          <AlertDescription>Fixture remedy</AlertDescription>
        </Alert>
        <Progress aria-label="Fixture progress" value={null} />
      </CardContent>
      <CardFooter>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button>Fixture help</Button>
            </TooltipTrigger>
            <TooltipContent>Fixture explanation</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardFooter>
    </Card>,
  )
  await expect
    .element(page.getByRole('alert'))
    .toHaveTextContent('Fixture remedy')
  await expect
    .element(page.getByRole('progressbar', { name: 'Fixture progress' }))
    .toBeVisible()
  await page.getByRole('button', { name: 'Fixture help' }).hover()
  await expect
    .element(page.getByRole('tooltip'))
    .toHaveTextContent('Fixture explanation')
})

const data = [{ name: 'Zebra' }, { name: 'Aster' }]
const columns: ColumnDef<(typeof data)[number]>[] = [
  { accessorKey: 'name', header: 'Fixture name' },
]

function TableFixture() {
  const [sorting, setSorting] = useState<SortingState>([])
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })
  return (
    <Table>
      <TableCaption>Fixture rows</TableCaption>
      <TableHeader>
        {table.getHeaderGroups().map((group) => (
          <TableRow key={group.id}>
            {group.headers.map((header) => (
              <TableHead key={header.id}>
                <Button onClick={header.column.getToggleSortingHandler()}>
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext(),
                  )}
                </Button>
              </TableHead>
            ))}
          </TableRow>
        ))}
      </TableHeader>
      <TableBody>
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <TableCell key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
      <TableFooter>
        <TableRow>
          <TableCell>{data.length} fixtures</TableCell>
        </TableRow>
      </TableFooter>
    </Table>
  )
}

test('the pinned TanStack table sorts rows through the imported table primitives', async () => {
  render(<TableFixture />)
  await expect
    .element(page.getByRole('table'))
    .toHaveAccessibleName('Fixture rows')
  const body = () =>
    [...document.querySelectorAll('tbody td')].map((cell) => cell.textContent)
  expect(body()).toEqual(['Zebra', 'Aster'])
  await page.getByRole('button', { name: 'Fixture name' }).click()
  await expect.poll(body).toEqual(['Aster', 'Zebra'])
})
