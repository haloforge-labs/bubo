# Rebuilds frontend/dist, which Go embeds. Only its placeholder is tracked.
# Use Node 24.19+; wails build also builds the production frontend.
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot 'frontend')
try {
    npm ci
    if ($LASTEXITCODE) { throw 'npm ci failed' }
    npm run build
    if ($LASTEXITCODE) { throw 'npm run build failed' }
} finally { Pop-Location }
