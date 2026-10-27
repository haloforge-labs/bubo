import { Generator, getConfig } from '@tanstack/router-generator'

await new Generator({
  config: getConfig(
    { generatedRouteTree: './src/generated/routeTree.gen.ts' },
    process.cwd(),
  ),
  root: process.cwd(),
}).run()
console.log('Route tree generated from src/routes/.')
