# Add the desktop application to an existing, immutable CLI runtime package.
# The source package and its original members are never changed.
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$RuntimePackage,
  [Parameter(Mandatory)][string]$GUIExe,
  [Parameter(Mandatory)][string]$LicenseBundle,
  [Parameter(Mandatory)][string]$Out,
  [Parameter(Mandatory)][string]$Notes,
  [Parameter(Mandatory)][bool]$RuntimeChanged,
  [Parameter(Mandatory)][ValidateRange(0, [long]::MaxValue)][long]$RuntimeDownloadBytes,
  [Parameter(Mandatory)][bool]$ModelsChanged,
  [Parameter(Mandatory)][AllowEmptyString()][string]$ModelsNeedsPrepare,
  [Parameter(Mandatory)][string]$MinUpdatableFrom,
  [string]$Version = '',
  [string]$Tag = '',
  [string]$Commit = '',
  [string]$Go = 'go',
  [switch]$StructureOnly
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..").Path
$Notes = (Resolve-Path -LiteralPath $Notes).Path
if ([string]::IsNullOrWhiteSpace([IO.File]::ReadAllText($Notes, [Text.Encoding]::UTF8))) { throw 'Notes must contain a nonempty user-facing release summary' }
$runtime = (Resolve-Path -LiteralPath $RuntimePackage).Path
$gui = (Resolve-Path -LiteralPath $GUIExe).Path
$licenses = (Resolve-Path -LiteralPath $LicenseBundle).Path
$outPath = [IO.Path]::GetFullPath($Out).TrimEnd('\')
if ($outPath.StartsWith($runtime + '\', [StringComparison]::OrdinalIgnoreCase) -or $outPath -eq $runtime) { throw 'Output must be outside the runtime package' }
$manifest = Get-Content -LiteralPath (Join-Path $runtime 'BUILD-MANIFEST.json') -Raw | ConvertFrom-Json
if (-not $Version) { $Version = $manifest.product.product_version }
if ($Version -notmatch '^\d+\.\d+\.\d+$' -or $Version -ne $manifest.product.product_version) { throw 'Version must match the runtime package' }
if (-not $StructureOnly -and ($manifest.qualification -or $manifest.cmake.ENGINE_ENABLE_HIP -ne 'ON')) { throw 'A real package needs a qualified HIP runtime build' }
if (-not $Commit) { throw 'Commit is required for prebuilt GUI provenance (Wails may omit VCS stamps)' }
# Check available Go VCS stamps; Wails may omit them, requiring the declaration.
$buildInfo = @(& $Go version -m $gui)
if ($LASTEXITCODE) { throw 'Cannot read GUI Go build information' }
$stamps = @($buildInfo | Where-Object { $_ -match 'vcs.revision=' })
if ($stamps.Count -and -not @($stamps | Where-Object { $_ -match "vcs.revision=$Commit$" }).Count) { throw 'GUI prebuilt revision does not match Commit' }
if ($Commit -notmatch '^[0-9a-f]{40}$') { throw 'Commit must be a full source revision' }
if (-not $StructureOnly -and @($buildInfo | Where-Object { $_ -match 'vcs.modified=true' }).Count) { throw 'GUI was built from a dirty tree' }
if (-not $StructureOnly) {
  # The runtime may be the gated package from an earlier commit; only GUI, docs and packaging tools may differ since.
  & "$PSScriptRoot\check-runtime-provenance.ps1" -RuntimeCommit ([string]$manifest.source.commit) -ReleaseCommit $Commit -Repo $root
}
if (-not $StructureOnly) {
  $headCommit = (& git -C $root rev-parse HEAD).Trim()
  if ($LASTEXITCODE -or $headCommit -ne $Commit) { throw 'Build the helper from the declared release checkout' }
  & git -C $root diff --quiet HEAD -- gui/updater
  if ($LASTEXITCODE -or @(& git -C $root ls-files --others --exclude-standard -- gui/updater).Count) { throw 'Commit helper source changes before real package assembly' }
}
$runtimeSource = @(& git -C $root show "$($manifest.source.commit):src/cli/prepare/runtime_location.hpp") -join "`n"
if ($LASTEXITCODE -or $runtimeSource -notmatch 'kRuntimeId\s*=\s*"([^"]+)"') { throw 'CLI runtime ID is unavailable at the packaged runtime source revision' }
$runtimeID = $matches[1]
Push-Location (Join-Path $root 'gui\updater')
$savedCGO = $env:CGO_ENABLED
try {
  $env:CGO_ENABLED = '0'
  $layoutJSON = & $Go run ./cmd/bubo-release -PackageInfo
  if ($LASTEXITCODE) { throw 'Cannot read updater package layout' }
  $layout = $layoutJSON | ConvertFrom-Json
  $name = "bubo-v$Version-windows-x64-gfx1151"
  $stage = Join-Path $outPath $name
  $zip = Join-Path $outPath "$name.zip"
  if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)) { throw 'Output already exists; choose an empty output directory' }
  New-Item -ItemType Directory -Force $stage | Out-Null
  $originals = @(Get-ChildItem -LiteralPath $runtime -File -Recurse | ForEach-Object {
    [pscustomobject]@{path=[IO.Path]::GetRelativePath($runtime, $_.FullName); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
  })
  foreach ($file in $originals) {
    $destination = Join-Path $stage $file.path
    New-Item -ItemType Directory -Force ([IO.Path]::GetDirectoryName($destination)) | Out-Null
    Copy-Item -LiteralPath (Join-Path $runtime $file.path) -Destination $destination
  }
  foreach ($rel in @($layout.gui, $layout.helper)) { if (Test-Path -LiteralPath (Join-Path $stage $rel)) { throw "Runtime already contains $rel" } }
  Copy-Item -LiteralPath $gui -Destination (Join-Path $stage $layout.gui)
  & $Go build '-ldflags=-H=windowsgui' -o (Join-Path $stage $layout.helper) ./cmd/bubo-update
  if ($LASTEXITCODE) { throw 'Updater helper build failed' }
  if (-not (Test-Path -LiteralPath (Join-Path $licenses 'THIRD-PARTY-GUI.md'))) { throw 'Generate the GUI licence bundle first' }
  $noticeDir = Join-Path $stage 'licenses\gui'
  New-Item -ItemType Directory -Force $noticeDir | Out-Null
  Copy-Item -Path (Join-Path $licenses '*') -Destination $noticeDir -Recurse
  Add-Content -LiteralPath (Join-Path $stage 'THIRD-PARTY.txt') -Value "`r`nGUI and updater`r`n---------------`r`nGenerated inventory: licenses/gui/THIRD-PARTY-GUI.md (source: docs/release/THIRD-PARTY-GUI.md).`r`nComplete GUI dependency licence texts accompany that inventory."
  if ($StructureOnly) { 'structure only, CPU engine, not shippable' | Set-Content -LiteralPath (Join-Path $stage 'STRUCTURE-ONLY.txt') }
  & "$root\tools\release\build-manifest.ps1" -PackageFolder $stage -ExtendExisting -ExtraFiles @($layout.gui, $layout.helper)
  $manifest = Get-Content -LiteralPath (Join-Path $stage 'BUILD-MANIFEST.json') -Raw | ConvertFrom-Json
  $manifest | Add-Member -NotePropertyName gui -NotePropertyValue ([ordered]@{commit=$Commit; runtime_id=$runtimeID; go_version=(& $Go version)})
  $manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $stage 'BUILD-MANIFEST.json') -Encoding utf8
  $preserved = 0
  foreach ($file in $originals) {
    if ($file.path -in @('BUILD-MANIFEST.json', 'THIRD-PARTY.txt')) { continue }
    if ((Get-FileHash -LiteralPath (Join-Path $stage $file.path) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Existing member changed: $($file.path)" }
    $preserved++
  }
  Write-Output "PASS: existing runtime members byte-identical=$preserved; only BUILD-MANIFEST.json and THIRD-PARTY.txt extended"
  & pwsh -NoProfile -File "$root\tools\release\check-package.ps1" -PackageFolder $stage -RequireGUI -Go $Go
  if ($LASTEXITCODE) { throw 'Assembled package gate failed' }
  & (Join-Path $env:SystemRoot 'System32\tar.exe') -a -c -f $zip -C $outPath $name
  if ($LASTEXITCODE) { throw 'ZIP creation failed' }
  & pwsh -NoProfile -File "$root\tools\release\check-package.ps1" -PackageZip $zip -RequireGUI -Go $Go
  if ($LASTEXITCODE) { throw 'ZIP shipped-file gate failed' }
  # Read the running clock before generating timestamped release metadata.
  $released = (& 'C:\Program Files\Git\usr\bin\date.exe' -u '+%Y-%m-%dT%H:%M:%SZ').Trim()
  if ($LASTEXITCODE) { throw 'Cannot read release time' }
  & $Go run ./cmd/bubo-release -Zip $zip -Out (Join-Path $outPath 'bubo-release.json') -Version $Version -Tag $(if ($Tag) {$Tag} else {"v$Version"}) -Commit $Commit -MinUpdatableFrom $MinUpdatableFrom -Notes $Notes -Released $released "-RuntimeChanged=$($RuntimeChanged.ToString().ToLowerInvariant())" "-RuntimeDownloadBytes=$RuntimeDownloadBytes" "-ModelsChanged=$($ModelsChanged.ToString().ToLowerInvariant())" "-ModelsNeedsPrepare=$ModelsNeedsPrepare"
  if ($LASTEXITCODE) { throw 'Release manifest generation failed' }
  $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
  "$hash  $name.zip" | Set-Content -LiteralPath (Join-Path $outPath 'SHA256SUMS.txt') -Encoding ascii
  Get-ChildItem -LiteralPath (Join-Path $stage 'bin') -File | ForEach-Object { Write-Output "$($_.Name)`t$($_.Length)`t$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())" }
  Write-Output "PASS: assembled $zip"
} finally { $env:CGO_ENABLED = $savedCGO; Pop-Location }
