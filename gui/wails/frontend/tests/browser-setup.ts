import { afterEach } from 'vitest'
import { cleanup } from 'vitest-browser-react'
import '@/styles/app.css'

afterEach(() => {
  cleanup()
  delete document.documentElement.dataset.theme
})
