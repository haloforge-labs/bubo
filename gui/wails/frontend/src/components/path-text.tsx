import { Fragment } from 'react'

export function PathText({ path }: { path: string }) {
  return (
    <code className="text-xs [overflow-wrap:anywhere]">
      {path.split(/(?<=[\\/])/).map((part, index) => (
        <Fragment key={index}>
          {part}
          <wbr />
        </Fragment>
      ))}
    </code>
  )
}
