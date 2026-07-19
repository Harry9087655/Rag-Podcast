import type { HealthResponse } from './types.ts'

const BACKEND_URL = 'http://localhost:8000'

const app = document.querySelector<HTMLDivElement>('#app')!

fetch(`${BACKEND_URL}/health`)
  .then(async (res) => {
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    const data: HealthResponse = await res.json()
    app.textContent = `Backend + DB connected ✅ (status: ${data.status}, db: ${data.db})`
  })
  .catch((err: Error) => {
    app.textContent = `Failed to reach backend: ${err.message}`
  })
