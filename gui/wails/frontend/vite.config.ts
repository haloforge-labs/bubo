import { fileURLToPath, URL } from 'node:url'
import { writeFileSync } from 'node:fs'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  plugins: [
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
      generatedRouteTree: './src/generated/routeTree.gen.ts',
    }),
    react(),
    tailwindcss(),
    {
      name: 'embed-placeholder',
      // Vite cleans stale assets; restore the tracked placeholder for Go-only builds.
      closeBundle() {
        writeFileSync(new URL('./dist/.gitkeep', import.meta.url), '')
      },
    },
  ],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: { host: 'localhost' },
})
