import inventory from '@/lib/third-party'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

// The packaging inventory remains canonical. Its generated Markdown table has
// plain pipe-delimited cells; render its first three columns, without provenance.
const entries = inventory
  .split('\n')
  .filter((line) => line.startsWith('| '))
  .slice(2)
  .map((line) =>
    line
      .split('|')
      .slice(1, 4)
      .map((cell) => cell.trim()),
  )

export function ThirdPartyNotices() {
  return (
    <Table aria-label="Third-party inventory" className="table-fixed">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[55%]">Component</TableHead>
          <TableHead className="w-[20%]">Version</TableHead>
          <TableHead className="w-[25%]">Licence</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {entries.map(([component, version, licence], index) => (
          <TableRow key={`${component}-${version}-${index}`}>
            <TableCell className="whitespace-normal [overflow-wrap:anywhere]">
              {component}
            </TableCell>
            <TableCell className="tabular-nums whitespace-normal [overflow-wrap:anywhere]">
              {version}
            </TableCell>
            <TableCell className="whitespace-normal [overflow-wrap:anywhere]">
              {licence}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
