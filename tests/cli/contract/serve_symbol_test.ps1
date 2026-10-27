# Execute the actual package test-symbol predicate on independently written
# files, then inspect the production bootstrap/control archive when built.
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$BuildRoot,
  [string]$ProductionLibrary,
  [string]$ProductionImage
)
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$source = Get-Content -LiteralPath (Join-Path $repository 'tools\release\check-package.ps1') -Raw
$match = [regex]::Match($source,
  '(?m)^\s*if \(\$imageText -match ''[^'']+''\) \{[\s\S]*?^\s*\}')
if (!$match.Success -or !$match.Value.Contains('fixture override symbol in release binary')) {
  throw 'The actual package fixture-symbol predicate is missing.'
}
$predicate = [ScriptBlock]::Create($match.Value)
$buildDirectory = [IO.Path]::GetFullPath($BuildRoot)
[IO.Directory]::CreateDirectory($buildDirectory) | Out-Null
$passed = 0
$markers = @(
  'FakeServeBackend',
  'BUBO_SERVE_TESTING',
  '--fake-backend',
  '--fake-hang-stop',
  '--fake-load-delay-ms',
  '--fake-stop-delay-ms',
  '--fake-event-burst',
  '--fake-log-burst',
  'bubo-test-only-fake'
)
foreach ($marker in $markers) {
  $path = Join-Path $buildDirectory ('marker-' + $passed + '.bin')
  [IO.File]::WriteAllBytes($path, [Text.Encoding]::ASCII.GetBytes("fixture`0$marker`0end"))
  $file = [pscustomobject]@{ FullName = $path }
  $imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($path))
  $errors = [Collections.Generic.List[string]]::new()
  & $predicate
  if ($errors.Count -ne 1 -or !$errors[0].Contains('fixture override symbol')) {
    throw "The production package gate accepted test marker: $marker"
  }
  ++$passed
  Write-Output "PASS package_rejects_$marker"
}

$cleanPath = Join-Path $buildDirectory 'production-clean.bin'
[IO.File]::WriteAllBytes($cleanPath, [Text.Encoding]::ASCII.GetBytes(
  'Bubo.Control. starting loading ready stopping stopped serve --jsonl status --json'))
$file = [pscustomobject]@{ FullName = $cleanPath }
$imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($cleanPath))
$errors = [Collections.Generic.List[string]]::new()
& $predicate
if ($errors.Count -ne 0) { throw 'The package gate rejected ordinary production control strings.' }
++$passed
Write-Output 'PASS package_accepts_production_control_strings'

foreach ($artifact in @($ProductionLibrary, $ProductionImage)) {
  if ([string]::IsNullOrEmpty($artifact)) { continue }
  if (!(Test-Path -LiteralPath $artifact -PathType Leaf)) {
    throw "Requested production artifact was not built: $artifact"
  }
  $file = [pscustomobject]@{ FullName = [IO.Path]::GetFullPath($artifact) }
  $imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($file.FullName))
  $errors = [Collections.Generic.List[string]]::new()
  & $predicate
  if ($errors.Count -ne 0) { throw $errors[0] }
  ++$passed
  Write-Output "PASS production_artifact_has_no_fixture_symbols $($file.FullName)"
}
if ([string]::IsNullOrEmpty($ProductionImage)) {
  Write-Output 'NOT RUN production engine/package image scan; pass -ProductionImage after the production build.'
}
Write-Output "Serve test-symbol gate: $passed passed, 0 failed"
