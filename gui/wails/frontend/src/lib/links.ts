const repository = 'https://github.com/haloforge-labs/bubo'

export const links = {
  website: 'https://haloforgelabs.com',
  docs: 'https://haloforgelabs.com/docs/',
  repository,
  releases: `${repository}/releases`,
  release: (version: string) => `${repository}/releases/tag/v${version}`,
}
