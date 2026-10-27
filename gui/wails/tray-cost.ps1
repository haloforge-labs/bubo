param(
    [string]$EvidenceDir = 'C:/bubo/evidence/capture-02/j8b-design-fix/wails-proof',
    [switch]$SkipScan
)
$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\bubo\tools\go-1.27.2\go\bin;' + $env:PATH
$env:GOPATH = 'C:\bubo\run\gui-go\gopath'
$env:GOCACHE = 'C:\bubo\run\gui-go\cache'
$env:CGO_ENABLED = '0'
$env:GOFLAGS = '-mod=readonly'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$utf8 = [Text.UTF8Encoding]::new($false)
$evidence = [IO.Path]::GetFullPath($EvidenceDir)
[IO.Directory]::CreateDirectory($evidence) | Out-Null
function Save-LF([string]$Name, [object[]]$Lines) {
    [IO.File]::WriteAllText((Join-Path $evidence $Name), (($Lines | ForEach-Object { [string]$_ }) -join "`n") + "`n", $utf8)
}
# Two isolated source snapshots keep the frontend and backend identical. The
# only source difference is the native tray adapter versus an unsupported stub.
# Neither executable is launched, and no tray/window registration is performed.
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-ffff'
$buildRoot = Join-Path $evidence "tray-cost-build-$stamp"
[IO.Directory]::CreateDirectory($buildRoot) | Out-Null
$snapshot = Join-Path $buildRoot 'with-tray'
[IO.Directory]::CreateDirectory($snapshot) | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -File | Where-Object {
    ($_.Extension -eq '.go' -and $_.Name -notlike '*_test.go') -or $_.Name -in @('go.mod','go.sum')
} | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $snapshot }
New-Item -ItemType Directory -Force -Path (Join-Path $snapshot 'frontend') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'frontend/dist') -Destination (Join-Path $snapshot 'frontend') -Recurse
$appPath = ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../app'))).Replace('\','/')
$supervisorPath = ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../supervisor'))).Replace('\','/')
$modPath = Join-Path $snapshot 'go.mod'
$modText = [IO.File]::ReadAllText($modPath).Replace('=> ../app', "=> $appPath").Replace('=> ../supervisor', "=> $supervisorPath")
[IO.File]::WriteAllText($modPath, $modText.Replace("`r`n","`n"), $utf8)
$baseline = Join-Path $buildRoot 'without-tray'
Copy-Item -LiteralPath $snapshot -Destination $baseline -Recurse
$stub = @'
//go:build windows

package main

import (
    "errors"
    "bubo/gui/supervisor"
)

func startNativeTray(func(string), func() supervisor.State) (func(), error) {
    return nil, errors.New("system tray is excluded from this cost baseline")
}
'@
[IO.File]::WriteAllText((Join-Path $baseline 'tray_windows.go'), $stub.Replace("`r`n","`n") + "`n", $utf8)
$sourceHashes = @(Get-ChildItem -LiteralPath $snapshot -Recurse -File | Sort-Object FullName | ForEach-Object {
    [ordered]@{file=$_.FullName.Substring($snapshot.Length + 1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
})
Save-LF 'tray-source-snapshot.json' @($sourceHashes | ConvertTo-Json -Depth 4)
$items = @()
foreach ($variant in @(
    @{Name='with-tray';Source=$snapshot;Exe='Bubo.Control.Web.Tray.exe'},
    @{Name='without-tray';Source=$baseline;Exe='Bubo.Control.Web.NoTray.exe'}
)) {
    # Owner requires an explicit AC wait before builds, including this pure-Go
    # comparison. The snapshots above can be prepared while on battery.
    $ac = @(& pwsh -NoProfile -File 'C:/bubo/evidence/capture-02/wait-ac.ps1')
    Save-LF ("tray-cost-" + $variant.Name + '-ac.txt') $ac
    if ($LASTEXITCODE) { throw 'AC check failed' }
    Push-Location $variant.Source
    try {
        $exe = Join-Path $evidence $variant.Exe
        $output = @(& go build -tags 'production,wv2runtime.error' -trimpath '-ldflags=-H=windowsgui -s -w' -o $exe . 2>&1)
        $rc = $LASTEXITCODE
        Save-LF ("tray-cost-" + $variant.Name + '-build.txt') $output
        if ($rc) { throw "tray cost build failed ($rc): $output" }
        $items += [ordered]@{variant=$variant.Name;path=$exe;bytes=(Get-Item -LiteralPath $exe).Length;sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;launched=$false;scan='not run'}
    } finally { Pop-Location }
}
if (!$SkipScan) {
    $platform = 'C:/ProgramData/Microsoft/Windows Defender/Platform'
    $mp = @(Get-ChildItem -LiteralPath $platform -Directory | Sort-Object Name -Descending | ForEach-Object {
        Join-Path $_.FullName 'MpCmdRun.exe'
    } | Where-Object { Test-Path -LiteralPath $_ }) | Select-Object -First 1
    if (!$mp) { $mp = 'C:/Program Files/Windows Defender/MpCmdRun.exe' }
    if (!(Test-Path -LiteralPath $mp)) { throw 'MpCmdRun.exe not found' }
    foreach ($item in $items) {
        $scan = @(& $mp -Scan -ScanType 3 -File $item.path -DisableRemediation 2>&1)
        $rc = $LASTEXITCODE
        Save-LF ("tray-cost-" + $item.variant + '-defender.txt') @("Command: $mp -Scan -ScanType 3 -File $($item.path) -DisableRemediation", $scan, "Exit code: $rc")
        $item.scan_exit = $rc
        if ($rc) { $item.scan = 'failed'; throw "Defender exact-file scan failed for $($item.path): $rc" }
        $item.scan = 'passed'
    }
    # Get-MpThreatDetection reads the same stored detection records used by
    # Protection History. Match only our two exact executable paths.
    $hits = @(Get-MpThreatDetection | Where-Object {
        $resources = $_.Resources -join "`n"
        $match = $false
        foreach ($item in $items) {
            if ($resources -match [regex]::Escape($item.path)) { $match = $true }
        }
        $match
    })
    Save-LF 'tray-cost-protection-history.json' @([ordered]@{method='Get-MpThreatDetection exact path match';paths=@($items | ForEach-Object {$_.path});matching_detections=$hits.Count;records=$hits} | ConvertTo-Json -Depth 8)
    if ($hits.Count) { throw 'Protection History has a matching detection' }
}
$cost = [ordered]@{
    strategy='same-process Shell_NotifyIconW; pinned existing Go dependencies; no separate tray process'
    build_tags='production,wv2runtime.error';cgo_enabled=0
    native_tray_launched=$false;native_lifecycle_proven=$false
    source_snapshot=$snapshot;baseline_source=$baseline
    bytes_delta=($items[0].bytes - $items[1].bytes)
    variants=$items
    limitation='Build, static layout, raster and fake lifecycle proof only; Explorer/native menu/hide/restore behavior needs an authorized native run.'
}
Save-LF 'tray-cost.json' @($cost | ConvertTo-Json -Depth 8)
$cost | ConvertTo-Json -Depth 8 -Compress
