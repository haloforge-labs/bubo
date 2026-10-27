# Headless, structure-only proof. No C++ build, GPU, model or server is used.
param(
  [Parameter(Mandatory)][string]$RuntimeTemplate,
  [Parameter(Mandatory)][string]$GUIExe,
  [Parameter(Mandatory)][string]$GUICommit,
  [Parameter(Mandatory)][string]$LicenseBundle,
  [Parameter(Mandatory)][string]$CPUBuild,
  [Parameter(Mandatory)][string]$Out,
  [Parameter(Mandatory)][string]$Evidence,
  [string]$Go = 'go'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..").Path
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('bubo-package-proof-' + [guid]::NewGuid().ToString('N'))
$seed = Join-Path $scratch 'runtime'
New-Item -ItemType Directory -Force $seed, $Evidence | Out-Null
$savedManifest, $savedZip = $env:BUBO_RELEASE_MANIFEST, $env:BUBO_RELEASE_ZIP
try {
  $template = (Resolve-Path -LiteralPath $RuntimeTemplate).Path
  $templateManifest = Get-Content -LiteralPath (Join-Path $template 'BUILD-MANIFEST.json') -Raw | ConvertFrom-Json
  foreach ($file in Get-ChildItem -LiteralPath $template -Recurse -File) {
    $relative = [IO.Path]::GetRelativePath($template, $file.FullName)
    if ($relative -eq 'BUILD-MANIFEST.json') { continue }
    # A developer may have prepared AMD files beside this template's CLI.
    # Only the original manifested bootstrap binaries belong in the seed.
    if ($relative -like 'bin\*' -and $relative.Replace('\','/') -notin $templateManifest.files.path) { continue }
    $dest = Join-Path $seed $relative
    New-Item -ItemType Directory -Force ([IO.Path]::GetDirectoryName($dest)) | Out-Null
    if ($relative -match '^bin\\[^\\]+\.(exe|dll)$') {
      $cpu = Join-Path $CPUBuild $file.Name
      if (-not (Test-Path -LiteralPath $cpu -PathType Leaf)) { throw "CPU package member unavailable: $cpu" }
      New-Item -ItemType HardLink -Path $dest -Target $cpu | Out-Null
    } else {
      # Immutable seed data shares bytes, reducing SSD writes. Nothing writes
      # these links; the assembler copies them into its one output package.
      New-Item -ItemType HardLink -Path $dest -Target $file.FullName | Out-Null
    }
  }
  & "$root\tools\release\build-manifest.ps1" -BuildFolder $CPUBuild -PackageFolder $seed -StructureOnly
  $notes = Join-Path $scratch 'notes.txt'
  'structure only, CPU engine, not shippable' | Set-Content -LiteralPath $notes -Encoding utf8
  & "$PSScriptRoot\assemble-app.ps1" -RuntimePackage $seed -GUIExe $GUIExe -Commit $GUICommit -LicenseBundle $LicenseBundle -Out $Out -Notes $notes -RuntimeChanged:$false -RuntimeDownloadBytes 0 -ModelsChanged:$false -ModelsNeedsPrepare '' -MinUpdatableFrom '0.0.0' -StructureOnly -Go $Go *>&1 | Tee-Object (Join-Path $Evidence 'assemble.log')
  $metadata = Get-Content -LiteralPath (Join-Path $Out 'bubo-release.json') -Raw | ConvertFrom-Json
  $stage = Join-Path $Out $metadata.assets[0].top_folder
  $env:BUBO_RELEASE_MANIFEST = Join-Path $Out 'bubo-release.json'
  $env:BUBO_RELEASE_ZIP = Join-Path $Out $metadata.assets[0].name
  Push-Location (Join-Path $root 'gui\updater')
  try {
    & $Go vet ./...
    if ($LASTEXITCODE) { throw 'Updater vet failed' }
    $lines = @(& $Go test -json -count=1 -run '^(TestGeneratedReleasePackage|TestReleaseGeneratorRejectsInconsistentMetadata)$' .)
    $code = $LASTEXITCODE
    $lines | Set-Content -LiteralPath (Join-Path $Evidence 'package-tests.jsonl') -Encoding utf8
    $events = @($lines | ForEach-Object { $_ | ConvertFrom-Json })
    $passed = @($events | Where-Object { $_.Action -eq 'pass' -and $_.Test }).Count
    $failed = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test }).Count
    $skipped = @($events | Where-Object { $_.Action -eq 'skip' -and $_.Test }).Count
    if ($code -or $failed -or $skipped) { $lines | Write-Output; throw 'Generated package validation failed' }
    Write-Output "PASS: updater package tests passed=$passed failed=$failed skipped=$skipped; vet packages=3"
    $layout = (& $Go run ./cmd/bubo-release -PackageInfo) | ConvertFrom-Json
    if ($LASTEXITCODE) { throw 'Cannot read required package names' }
  } finally { Pop-Location }
  $helper = Join-Path $stage $layout.helper
  $hidden = Join-Path $scratch 'held-helper.exe'
  Move-Item -LiteralPath $helper -Destination $hidden
  try {
    $negative = @(& pwsh -NoProfile -File "$root\tools\release\check-package.ps1" -PackageFolder $stage -RequireGUI -Go $Go 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -eq 0 -or ($negative -join "`n") -notmatch [regex]::Escape($layout.helper)) { throw 'Missing-helper negative gate did not reject the expected member' }
    $negative | Set-Content -LiteralPath (Join-Path $Evidence 'missing-helper.log') -Encoding utf8
    Write-Output 'PASS: check-package rejects missing helper (1 negative case)'
  } finally { Move-Item -LiteralPath $hidden -Destination $helper }
  & pwsh -NoProfile -File "$root\tools\release\check-package.ps1" -PackageFolder $stage -RequireGUI -Go $Go
  if ($LASTEXITCODE) { throw 'Restored package gate failed' }
} finally {
  $env:BUBO_RELEASE_MANIFEST, $env:BUBO_RELEASE_ZIP = $savedManifest, $savedZip
  $resolved = [IO.Path]::GetFullPath($scratch)
  if ([IO.Path]::GetDirectoryName($resolved) -ne [IO.Path]::GetTempPath().TrimEnd('\') -or [IO.Path]::GetFileName($resolved) -notmatch '^bubo-package-proof-[a-f0-9]{32}$') { throw 'Refusing cleanup outside proof scratch' }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}
