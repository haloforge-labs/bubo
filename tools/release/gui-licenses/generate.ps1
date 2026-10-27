[CmdletBinding(DefaultParameterSetName='Tree')]
param(
  [Parameter(Mandatory,ParameterSetName='Tree')][string]$FrontendDir,
  [Parameter(Mandatory,ParameterSetName='Ref')][string]$FrontendRef,
  [Parameter(ParameterSetName='Ref')][string]$InstalledModules = '',
  [Parameter(Mandatory)][string]$Bundle,
  [string]$Report = '',
  [string]$Go = 'go'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..\..").Path
if (-not $Report) { $Report = Join-Path $root 'docs\release\THIRD-PARTY-GUI.md' }
$bundlePath = [IO.Path]::GetFullPath($Bundle)
if (Test-Path -LiteralPath $bundlePath) { throw 'Use an empty bundle path to avoid stale notices' }
New-Item -ItemType Directory -Force $bundlePath | Out-Null
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('bubo-gui-licenses-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $scratch | Out-Null
$junction = $null
$savedCGO, $savedFlags = $env:CGO_ENABLED, $env:GOFLAGS
try {
  if ($FrontendRef) {
    $provenance = (& git -C $root rev-parse $FrontendRef).Trim()
    if ($LASTEXITCODE) { throw 'Unknown frontend revision' }
    $frontendPath = 'gui/wails/frontend'
    $selected = @('package.json','package-lock.json','src/vendor/shadcn-admin-LICENSE.txt','src/fonts/Go-Mono-LICENSE.txt','src/fonts/et-book-LICENSE.txt') | ForEach-Object { "$frontendPath/$_" }
    $tar = Join-Path $scratch 'inputs.tar'
    & git -C $root archive --format=tar "--output=$tar" $FrontendRef -- @selected
    if ($LASTEXITCODE) { throw 'Cannot extract locked frontend inputs' }
    & "$env:SystemRoot\System32\tar.exe" -x -f $tar -C $scratch
    if ($LASTEXITCODE) { throw 'Cannot extract locked frontend inputs' }
    $frontend = Join-Path $scratch $frontendPath
    if (-not $InstalledModules) { $InstalledModules = Join-Path $root "$frontendPath/node_modules" }
    $junction = Join-Path $frontend 'node_modules'
    New-Item -ItemType Junction -Path $junction -Target (Resolve-Path -LiteralPath $InstalledModules).Path | Out-Null
  } else {
    $frontend = (Resolve-Path -LiteralPath $FrontendDir).Path
    $provenance = 'working tree ' + (& git -C $root rev-parse HEAD).Trim()
  }
  Push-Location $PSScriptRoot
  try {
    & npm ci --ignore-scripts --no-audit --no-fund
    if ($LASTEXITCODE) { throw 'Pinned npm licence tool install failed' }
    $tool = Join-Path $scratch 'go-licenses-offline.exe'
    & $Go build -o $tool .
    if ($LASTEXITCODE) { throw 'Pinned go-licenses adapter build failed' }
  } finally { Pop-Location }
  $env:CGO_ENABLED = '0'
  $env:GOFLAGS = '-tags=desktop,production,wv2runtime.error'
  $goData = [ordered]@{}
  foreach ($entry in @(@{binary='GUI';dir='gui\wails';package='.'},@{binary='helper';dir='gui\updater';package='./cmd/bubo-update'})) {
    $module = Join-Path $root $entry.dir
    $json = & $tool -dir $module -package $entry.package
    if ($LASTEXITCODE) { throw "go-licenses failed for $($entry.binary)" }
    $rows = @($json | ConvertFrom-Json)
    Push-Location $module
    try {
      $deps = @(& $Go list -deps -f '{{if .Module}}{{.Module.Path}}|{{.Module.Version}}{{end}}' $entry.package | Where-Object { $_ -and $_ -notlike 'bubo/gui/*' } | Sort-Object -Unique)
      if ($LASTEXITCODE) { throw 'Cannot verify linked Go modules' }
      foreach ($dep in $deps) {
        $name, $version = $dep -split '\|',2
        if (-not @($rows | Where-Object { ($_.Name -eq $name -or $_.Name.StartsWith($name+'/')) -and $_.Version -eq $version }).Count) { throw "Missing linked Go module notice: $dep" }
      }
    } finally { Pop-Location }
    $goData[$entry.binary] = $rows
    Write-Output "PASS: go-licenses $($entry.binary) libraries=$($rows.Count); linked modules=$($deps.Count)"
  }
  $goJSON = Join-Path $scratch 'go.json'
  $goData | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $goJSON -Encoding utf8
  & node "$PSScriptRoot\inventory.mjs" --frontend $frontend --out $bundlePath --report $Report --go-tool $tool --go-data $goJSON --provenance $provenance
  if ($LASTEXITCODE) { throw 'Licence inventory generation failed' }
} finally {
  $env:CGO_ENABLED, $env:GOFLAGS = $savedCGO, $savedFlags
  if ($junction -and (Test-Path -LiteralPath $junction)) { Remove-Item -LiteralPath $junction -Force }
  $resolved = [IO.Path]::GetFullPath($scratch)
  if ([IO.Path]::GetDirectoryName($resolved) -ne [IO.Path]::GetTempPath().TrimEnd('\') -or [IO.Path]::GetFileName($resolved) -notmatch '^bubo-gui-licenses-[a-f0-9]{32}$') { throw 'Refusing cleanup outside inventory scratch' }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}
