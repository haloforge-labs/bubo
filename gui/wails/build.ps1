param([Parameter(Mandatory)][string]$OutputDir)
$ErrorActionPreference = 'Stop'
$env:CGO_ENABLED = '0'
$env:GOFLAGS = '-mod=readonly'
$env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = '1'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if ((go version) -ne 'go version go1.27.2 windows/amd64') { throw 'Use Go 1.27.2.' }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$wails = Join-Path (go env GOPATH) 'bin/wails.exe'
$config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'wails.json') -Raw | ConvertFrom-Json
$guiName = $config.outputfilename
if (-not $guiName.EndsWith('.exe')) { $guiName += '.exe' }
$guiExe = Join-Path $OutputDir $guiName
$helperExe = Join-Path $OutputDir 'bubo-update.exe'
Push-Location $PSScriptRoot
try {
    # Wails selects -H windowsgui for production Windows builds. Binding
    # generation is already verified by gui/tests/test.ps1; never run the app.
    & $wails build -skipbindings -m -nosyncgomod -tags production,wv2runtime.error -webview2 error -trimpath -o $guiName
    if ($LASTEXITCODE -ne 0) { throw 'Wails production build failed.' }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "build/bin/$guiName") -Destination $guiExe -Force
    $deps = @(go list -tags desktop,production,wv2runtime.error -deps .)
    if ($LASTEXITCODE -ne 0) { throw 'Production dependency listing failed.' }
    $deps | Set-Content -LiteralPath (Join-Path $OutputDir 'gui-deps.log')
    $fyne = @($deps | Where-Object { $_ -match '^fyne\.io/' })
    if ($fyne.Count) { throw "Production executable depends on $($fyne.Count) fyne.io packages." }
    Write-Output "production dependency guard: $($deps.Count) packages, 0 fyne.io packages"
    $commit = (git rev-parse HEAD).Trim()
} finally { Pop-Location }
Push-Location (Join-Path (Split-Path $PSScriptRoot) 'updater')
try {
    go build -buildvcs=false -trimpath '-ldflags=-H=windowsgui -s -w' -o $helperExe ./cmd/bubo-update
    if ($LASTEXITCODE -ne 0) { throw 'Updater helper build failed.' }
} finally { Pop-Location }
$artifacts = @(foreach ($file in @($guiExe, $helperExe)) {
    go version -m $file | Set-Content -LiteralPath (Join-Path $OutputDir ((Split-Path $file -Leaf) + '.build.txt'))
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect $file" }
    [ordered]@{ path = $file; bytes = (Get-Item -LiteralPath $file).Length; sha256 = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash; launched = $false }
})
[ordered]@{ source_commit = $commit; cgo_enabled = 0; artifacts = $artifacts } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDir 'build.json')
$artifacts | ForEach-Object { Write-Output ($_ | ConvertTo-Json -Compress) }
Write-Output 'production builds: 2 passed, 0 failed; executables never launched'
